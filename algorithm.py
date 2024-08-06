import json
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, SGD
from function import sizeOfShard, getFeatureHash, fetchTrainBatch, fetchTestBatch, fetchTest
from time import perf_counter
import os
from importlib import import_module
import shutil
import numpy as np
from torch.nn.functional import one_hot


def train_and_test(model="MLP", train=False, retrain=False, retrain_dir="method", test_batch=False, test_single=False, epochs=10, batch=128,
                   dropout_rate=0.2, learning_rate=0.001, optimizer="sgd", output_type="softmax", shard_num=5,
                   shard_index=0, dataset="datasets/purchase/datasetfile", test_index=0):
    # Import the architecture.
    model_lib = import_module("architectures.{}".format(model))

    # Retrive dataset metadata.
    with open(dataset) as f:
        datasetfile = json.loads(f.read())

    input = np.load("{}/{}/featurefile/shard_{}.npy".format(retrain_dir, shard_num, shard_index))
    input_shape = input.shape
    nb_classes = datasetfile["nb_classes"]

    # Use Gpu if available
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # pylint: disable=no-member
    # print('using device: ', device)

    # Instantiate model and send to selected device.
    model = model_lib.Model(input_shape=input_shape, nb_classes=nb_classes, dropout_rate=dropout_rate)
    model.to(device)

    # Instantiate loss and optimizer
    loss_fn = CrossEntropyLoss()
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    else:
        raise "Unsupported optimizer"

    if train:

        # Get feature hash
        feature_hash = getFeatureHash(dire=retrain_dir, shard_num=shard_num, shard=shard_index)
        start_epoch = 0
        # Actual training.
        train_time = 0.0

        for epoch in range(start_epoch, epochs):

            for data, labels in fetchTrainBatch(dire=retrain_dir, shard_num=shard_num, shard=shard_index,
                                                batch_size=batch, dataset=dataset, until=None):
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

        # When training is complete, save the parameters.
        torch.save(
            model.state_dict(),
            "{}/{}/cache/{}.pt".format(retrain_dir, shard_num, feature_hash)
        )

        source_file = "{}/{}/cache/{}.pt".format(retrain_dir, shard_num, feature_hash)
        destination_path = "{}/{}/cache/shard_{}.pt".format(retrain_dir, shard_num, shard_index)
        if os.path.exists(destination_path):
            os.unlink(destination_path)
        shutil.copy2(source_file, destination_path)
        if not os.path.exists(
                "{}/{}/times/shard_{}.time".format(retrain_dir, shard_num, shard_index)
        ):
            with open(
                    "{}/{}/times/shard_{}.time".format(retrain_dir, shard_num, shard_index), "w"
            ) as f:
                f.write("{}\n".format(train_time))

    if retrain:
        feature_hash = getFeatureHash(dire=retrain_dir, shard_num=shard_num, shard=shard_index)
        if not os.path.exists(
                "{}/{}/cache/{}.pt".format(retrain_dir, shard_num, feature_hash)
        ):
            start_epoch = 0
            # Actual training.
            train_time = 0.0

            for epoch in range(start_epoch, epochs):

                for data, labels in fetchTrainBatch(dire=retrain_dir, shard_num=shard_num, shard=shard_index,
                                                    batch_size=batch, dataset=dataset, until=None):
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
            torch.save(
                model.state_dict(),
                "{}/{}/cache/{}.pt".format(retrain_dir, shard_num, feature_hash)
            )

            source_file = "{}/{}/cache/{}.pt".format(retrain_dir, shard_num, feature_hash)
            destination_path = "{}/{}/cache/shard_{}.pt".format(retrain_dir, shard_num, shard_index)
            if os.path.exists(destination_path):
                os.unlink(destination_path)
            shutil.copy2(source_file, destination_path)
            if not os.path.exists("{}/{}/times/retrain_shard_{}.time".format(retrain_dir, shard_num, shard_index)):

                with open(
                        "{}/{}/times/retrain_shard_{}.time".format(retrain_dir, shard_num, shard_index), "w"
                ) as f:
                    f.write("{}\n".format(train_time))
            else:
                with open(
                        "{}/{}/times/retrain_shard_{}.time".format(retrain_dir, shard_num, shard_index), "a"
                ) as f:
                    f.write("{}\n".format(train_time))

    if test_batch:
        # Load model weights.
        model.load_state_dict(
            torch.load(
                "{}/{}/cache/shard_{}.pt".format(
                    retrain_dir, shard_num, shard_index
                )
            )
        )

        model.eval()

        # Compute predictions batch per batch.
        outputs = np.empty((0, nb_classes))
        for data, _ in fetchTestBatch(dire=retrain_dir, shard_num=shard_num, shard=shard_index, dataset=dataset, batch_size=batch):

            # Convert data to torch format and send to selected device.
            gpu_images = torch.from_numpy(data).to(device)  # pylint: disable=no-member

            if output_type == "softmax":
                # Actual batch prediction.
                logits = model(gpu_images)
                predictions = torch.softmax(logits, dim=1).to("cpu")  # Send back to cpu.

                # Convert back to numpy and concatenate with previous batches.
                outputs = np.concatenate((outputs, predictions.detach().numpy()))


            else:
                # Actual batch prediction.
                logits = model(gpu_images)
                predictions = torch.argmax(logits, dim=1)  # pylint: disable=no-member

                # Convert to one hot, send back to cpu, convert back to numpy and concatenate with previous batches.
                out = one_hot(predictions, nb_classes).to("cpu")
                outputs = np.concatenate((outputs, out.numpy()))

        # Save outputs in numpy format.
        outputs = np.array(outputs)
        np.save(
            "{}/{}/outputs/shard_{}.npy".format(
                retrain_dir, shard_num, shard_index,
            ),
            outputs,
        )

    if test_single:
        # Load model weights.
        model.load_state_dict(
            torch.load(
                "{}/{}/cache/shard_{}.pt".format(
                    retrain_dir, shard_num, shard_index
                )
            )
        )

        model.eval()
        for data, _ in fetchTest(dire=retrain_dir, shard_num=shard_num, shard=shard_index, dataset=dataset, index=test_index):
            # Convert data to torch format and send to selected device.
            gpu_data = torch.from_numpy(data).to(device)

            if output_type == "softmax":
                logits = model(gpu_data)
                predictions = torch.softmax(logits, dim=1).to("cpu")  # Send back to cpu.
            else:
                # Actual batch prediction.
                logits = model(gpu_data)
                predictions = torch.argmax(logits, dim=1)  # pylint: disable=no-member

                predictions = one_hot(predictions, nb_classes).to("cpu")

                # Save outputs in numpy format.
            predictions_array = predictions.detach().numpy()
            return predictions_array



