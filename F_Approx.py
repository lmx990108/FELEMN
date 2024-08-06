import importlib
import json
import os
import random
import time
from importlib import import_module
from time import perf_counter
from torch.optim import Adam, SGD
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

def fetchTrainData(batch_size, dataset, until=None, offset=0,):

    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['Gdataloader']]))
    train_data_index = np.arange(0, datasetfile["nb_train"])

    if until is None or until > len(train_data_index):
        until = train_data_index.shape[0]
    limit = offset
    while limit <= until - batch_size:
        limit += batch_size
        indices = train_data_index[limit - batch_size:limit]
        yield dataloader.load(indices=indices, category="train")
    if limit < until:
        indices = train_data_index[limit:until]
        yield dataloader.load(indices=indices, category="train")


def train(model="LR", epochs=10, batch=128, dropout_rate=0.4, learning_rate=0.001, optimizer="sgd",
           dataset="datasets/purchase/datasetfile"):
    model_lib = import_module("architectures.{}".format(model))

    # Retrive dataset metadata.
    with open(dataset) as f:
        datasetfile = json.loads(f.read())

    nb_classes = datasetfile["nb_classes"]
    input_shape = tuple(datasetfile["original_input_shape"])
    # Use Gpu if available
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # pylint: disable=no-member

    model = model_lib.Model(input_shape=input_shape, nb_classes=nb_classes, dropout_rate=dropout_rate)
    model.to(device)

    loss_fn = CrossEntropyLoss()
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    else:
        raise "Unsupported optimizer"

    start_epoch = 0
    # Actual training.
    train_time = 0.0

    for epoch in range(start_epoch, epochs):
        for data, labels in fetchTrainData(batch_size=batch, dataset=dataset, until=None):
            # Convert data to torch format and send to selected device.
            gpu_data = torch.from_numpy(data).to(
                device
            )  # pylint: disable=no-member
            gpu_labels = torch.from_numpy(labels).to(
                device
            )  # pylint: disable=no-member

            forward_start_time = perf_counter()

            # Perform basic training step.
            logits = model(gpu_data)

            loss = loss_fn(logits, gpu_labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_time += perf_counter() - forward_start_time
    torch.save(model.state_dict(), "G/training.pt")
    return model

def compute_gradient(model, data, labels):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    loss_fn = CrossEntropyLoss()
    data = torch.from_numpy(data).to(device)
    labels = torch.from_numpy(labels).to(device)

    model.zero_grad()
    logits = model(data)
    loss = loss_fn(logits, labels)
    loss.backward()
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.clone().detach())
        else:
            gradients.append(None)
    return gradients

def cpmpute_gradient_b(model, data, labels, b=None):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    loss_fn = CrossEntropyLoss(reduction='sum')
    data = torch.from_numpy(data).to(device)
    labels = torch.from_numpy(labels).to(device)

    model.zero_grad()
    logits = model(data)
    flattened_params = torch.cat([p.flatten() for p in model.parameters()])
    b_tensor = torch.tensor(b, dtype=flattened_params.dtype, device=flattened_params.device)
    extra_term = (b_tensor * flattened_params).sum()
    loss_cross = loss_fn(logits, labels)
    loss = loss_cross + extra_term
    loss.backward()
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.clone().detach())
        else:
            gradients.append(None)
    return gradients


def test(model_name, dataset, batch, output_type="argmax"):
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['Gdataloader']]))

    nb_classes = datasetfile["nb_classes"]
    nb_test = datasetfile["nb_test"]
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.load_state_dict(torch.load(model_name))

    model.eval()
    outputs = np.empty((0, nb_classes))
    all_predictions = []
    for data, _ in fetchTestData(dataset=dataset, batch_size=batch):
        gpu_images = torch.from_numpy(data).to(device)  # pylint: disable=no-member
        logits = model(gpu_images)

        if output_type == "softmax":
            predictions = torch.softmax(logits, dim=1).to("cpu").detach().numpy()
        else:
            predictions = torch.argmax(logits, dim=1).to("cpu").detach().numpy()

            # Collect all predictions
        all_predictions.append(predictions)

        # Concatenate all predictions
    outputs = np.concatenate(all_predictions, axis=0)

    # Load true labels
    test_indices = range(int(nb_test))
    _, test_labels = dataloader.load(indices=test_indices, category='test')

    if output_type == "softmax":
        predicted_classes = np.argmax(outputs, axis=1)
    else:
        predicted_classes = outputs

    # Ensure test_labels is a 1D array
    test_labels = np.array(test_labels).flatten()


    accuracy = np.mean(predicted_classes == test_labels)

    return accuracy


def fetchTestData(dataset, batch_size):
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['Gdataloader']]))

    limit = 0
    while limit <= datasetfile['nb_test'] - batch_size:
        limit += batch_size
        yield dataloader.load(indices=np.arange(limit - batch_size, limit),  category='test')
    if limit < datasetfile['nb_test']:
        yield dataloader.load(indices=np.arange(limit, datasetfile['nb_test']), category='test')


def get_theta_A_w(unlearn_feature, Batch,Dataset):
    start_time = time.time()
    accumulated_gradient_diff = None
    for data, labels in fetchTrainData(batch_size=Batch, dataset=Dataset, until=None):

        grad_z = compute_gradient(model=model, data=data, labels=labels)
        data[:, unlearn_feature] = 0
        grad_delta_z = compute_gradient(model=model, data=data, labels=labels)
        grad_diff = [gz - gdz for gz, gdz in zip(grad_z, grad_delta_z)]
        # print(grad_diff)
        if accumulated_gradient_diff is None:
            accumulated_gradient_diff = grad_diff
        else:
            accumulated_gradient_diff = [agd + gd for agd, gd in zip(accumulated_gradient_diff, grad_diff)]
    with torch.no_grad():
        for param, grad_diff in zip(model.parameters(), accumulated_gradient_diff):
            if grad_diff is not None:
                param -= unlearning_rate * grad_diff

    torch.save(model.state_dict(), "G/unlearning_after_unlearning.pt")

    model.load_state_dict(torch.load("G/unlearning_after_unlearning.pt"))
    total_accumulated_gradient = None
    for data, labels in fetchTrainData(batch_size=Batch, dataset=Dataset, until=None):
        data[:, unlearn_feature] = 0
        grad = compute_gradient(model=model, data=data, labels=labels)

        if total_accumulated_gradient is None:
            total_accumulated_gradient = grad
        else:
            total_accumulated_gradient = [tag + g for tag, g in zip(total_accumulated_gradient, grad)]
    flattened_gradients = torch.cat([g.flatten() for g in total_accumulated_gradient])

    norm_2 = torch.norm(flattened_gradients, p=2)
    return norm_2
def get_total_param_count(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params


def get_G(index):
    start_time = time.time()
    accumulated_gradient_diff = None
    for data, labels in fetchTrainData(batch_size=Batch, dataset=Dataset, until=None):

        grad_z = compute_gradient(model=model, data=data, labels=labels)
        data[:, index] = 0
        grad_delta_z = compute_gradient(model=model, data=data, labels=labels)
        grad_diff = [gz - gdz for gz, gdz in zip(grad_z, grad_delta_z)]
        # print(grad_diff)
        if accumulated_gradient_diff is None:
            accumulated_gradient_diff = grad_diff
        else:
            accumulated_gradient_diff = [agd + gd for agd, gd in zip(accumulated_gradient_diff, grad_diff)]

    with torch.no_grad():
        for param, grad_diff in zip(model.parameters(), accumulated_gradient_diff):
            if grad_diff is not None:
                param -= unlearning_rate * grad_diff
    unlearn_time = time.time() - start_time

    torch.save(model.state_dict(), "G/unlearning_after.pt")
    return unlearn_time

def unlearning(model="LR", epochs=10, batch=128, dropout_rate=0.2, learning_rate=0.001, optimizer="sgd",
           dataset="datasets/purchase/datasetfile", b=None):
    model_lib = import_module("architectures.{}".format(model))
    # Retrive dataset metadata.
    with open(dataset) as f:
        datasetfile = json.loads(f.read())

    nb_classes = datasetfile["nb_classes"]
    input_shape = tuple(datasetfile["original_input_shape"])

    # Use Gpu if available
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # pylint: disable=no-member

    model = model_lib.Model(input_shape=input_shape, nb_classes=nb_classes, dropout_rate=dropout_rate)
    model.to(device)
    loss_fn = CrossEntropyLoss(reduction='sum')
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    else:
        raise "Unsupported optimizer"

    start_epoch = 0
    # Actual training.
    train_time = 0.0

    for epoch in range(start_epoch, epochs):
        # epoch_start_time = time()

        for data, labels in fetchTrainData(batch_size=batch, dataset=dataset, until=None):
            # Convert data to torch format and send to selected device.
            gpu_data = torch.from_numpy(data).to(
                device
            )  # pylint: disable=no-member
            gpu_labels = torch.from_numpy(labels).to(
                device
            )  # pylint: disable=no-member

            flattened_params = torch.cat([p.flatten() for p in model.parameters()])

            forward_start_time = perf_counter()
            b_tensor = torch.tensor(b, dtype=flattened_params.dtype, device=flattened_params.device)
            extra_term = (b_tensor * flattened_params).sum()

            # Perform basic training step.
            logits = model(gpu_data)
            loss_cross = loss_fn(logits, gpu_labels)
            loss = loss_cross + extra_term
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_time += perf_counter() - forward_start_time
    torch.save(model.state_dict(), "G/unlearning_before.pt")
    return model, train_time

if __name__ == "__main__":
    Model = "MLP"
    Epoch = 10
    Opt = "adam"
    Dataset = "datasets/adult/datasetfile"
    Batch = 128

    unlearning_rate = 0.0001
    epsilon = 0.1
    delta = 0.01
    if not os.path.isdir(f"G"):
        os.makedirs(f"G")
    model = train(model=Model, epochs=Epoch, optimizer=Opt, dataset=Dataset, batch=Batch)
    total_param_count = get_total_param_count(model)
    with open(Dataset) as f:
        datasetfile = json.loads(f.read())
    input_shape = datasetfile["original_input_shape"][0]

    unlearning_features = random.sample(range(int(datasetfile["original_input_shape"][0])), 2)
    w_1 = get_theta_A_w(unlearn_feature=unlearning_features[0], Batch=Batch, Dataset=Dataset)
    w_2 = get_theta_A_w(unlearn_feature=unlearning_features[1], Batch=Batch, Dataset=Dataset)
    beta = (w_1 + w_2) / 2
    c = np.sqrt(2*np.log(1.5/delta))
    sigma = beta*c/epsilon
    sigma = sigma.cpu().item() if isinstance(sigma, torch.Tensor) else sigma

    b = np.random.normal(0, sigma, total_param_count) if sigma != 0 else np.zeros(total_param_count)

    model, train_time = unlearning(model=Model, epochs=Epoch, optimizer=Opt, dataset=Dataset, batch=Batch, b=b)
    train_acc = test(model_name="G/unlearning_before.pt", dataset=Dataset, batch=Batch)
    index = random.sample(range(int(datasetfile["original_input_shape"][0])), 1)
    unlearn_time = get_G(index=index)
    unlearn_acc = test(model_name="G/unlearning_after.pt", dataset=Dataset, batch=Batch)
    with open("F-Approx.txt", "a") as file:
        file.write("dataset---------{}\n".format(Dataset))
        file.write(str(train_acc) + "\n")
        file.write(str(unlearn_acc) + "\n")
        file.write(str(train_time) + "\n")
        file.write(str(unlearn_time) + "\n")


