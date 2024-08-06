import json
import random
import numpy as np
import os
from aggregation import aggregation
from algorithm import train_and_test
from function import delete_directory_if_exists


def max_corr_distribution(shard_num, dataset="datasets/adult/datasetfile", feature_remain=None, feature_sample=None,
                          max_feature=0):
    features = {}
    for i in range(shard_num):
        features[i] = [feature_sample[i]]

    base_directory = "/".join(dataset.split("/")[:2])
    file_path = os.path.join(base_directory, "mutual_information.npy")
    corr_matrix = np.load(file_path)
    for idx, k in enumerate(feature_remain):
        corr_array = {}
        valid_keys = list(features.keys())
        for i in valid_keys:
            corr_sum = 0
            for m in features[i]:
                corr_sum += corr_matrix[k][m]
            corr_mean = corr_sum / len(features[i])
            corr_array[i] = corr_mean
        max_corr_mean_index = max(corr_array, key=corr_array.get)
        features[max_corr_mean_index].append(k)
        if len(features[max_corr_mean_index]) >= max_feature:
            value = features.pop(max_corr_mean_index)
            np.save("method/{}/featurefile/shard_{}.npy".format(shard_num, max_corr_mean_index), value)
    if features:
        for key, value in features.items():
            np.save(f"method/{shard_num}/featurefile/shard_{key}.npy", value)


def max_unlearn(Model, Shard_num, Epoch, Opt, Dataset, Batch, feature_first, unlearn_feature):
    directory_path = f"method/{Shard_num}"
    delete_directory_if_exists(directory_path)

    if not os.path.isdir(f"method/{Shard_num}"):
        os.makedirs(f"method/{Shard_num}")
        os.makedirs(f"method/{Shard_num}/cache")
        os.makedirs(f"method/{Shard_num}/times")
        os.makedirs(f"method/{Shard_num}/outputs")
        os.makedirs(f"method/{Shard_num}/featurefile")

    with open(Dataset) as f:
        datasetfile = json.loads(f.read())


    partition = np.split(
        np.arange(0, datasetfile["nb_train"]),
        [
            t * (datasetfile["nb_train"] // Shard_num)
            for t in range(1, Shard_num)
        ],
    )
    partition_array = np.asarray(partition, dtype=object)
    np.save("method/{}/splitfile.npy".format(Shard_num), partition_array, allow_pickle=True)

    original_input_shape = int(datasetfile["original_input_shape"][0])
    full_feature_index = list(range(int(original_input_shape)))
    feature_remain = list(set(full_feature_index) - set(feature_first))
    max_feature = original_input_shape // Shard_num + 1

    max_corr_distribution(shard_num=Shard_num, dataset=Dataset, feature_remain=feature_remain,
                          feature_sample=feature_first, max_feature=max_feature)


    unlearned_features_dict = {}


    for sh in range(Shard_num):

        train_and_test(model=Model, train=True, epochs=Epoch, batch=Batch, optimizer=Opt, shard_num=Shard_num,
                       shard_index=sh, dataset=Dataset)
        unlearned_features_dict[sh] = []
        train_and_test(test_batch=True, dataset=Dataset, batch=Batch, shard_index=sh, shard_num=Shard_num, model=Model)
    train_accuracy = aggregation(dire="method", dataset=Dataset, shards=Shard_num)

    if unlearn_feature in feature_first:
        feature_replace = random.choice(feature_remain)
        feature_remain.remove(feature_replace)
        index = feature_first.index(unlearn_feature)
        feature_first[index] = feature_replace
        max_corr_distribution(shard_num=Shard_num, dataset=Dataset, feature_remain=feature_remain,
                              feature_sample=feature_first, max_feature=max_feature)

    else:
        feature_remain.remove(unlearn_feature)
        max_corr_distribution(shard_num=Shard_num, dataset=Dataset, feature_remain=feature_remain,
                              feature_sample=feature_first, max_feature=max_feature)
    for k in range(Shard_num):
        train_and_test(model=Model, retrain=True, epochs=Epoch, batch=Batch, optimizer=Opt, shard_num=Shard_num,
                       shard_index=k, dataset=Dataset)
        train_and_test(test_batch=True, dataset=Dataset, batch=Batch, shard_index=k, shard_num=Shard_num, model=Model)
    retrain_accuracy = aggregation(dire="method", dataset=Dataset, shards=Shard_num)

    total_time = 0
    retrain_time = 0
    folder_path = "method/{}/times".format(Shard_num)
    i = 0
    for file_name in os.listdir(folder_path):
        if file_name.startswith("shard_"):
            time_file_path = os.path.join(folder_path, file_name)
            with open(time_file_path, "r") as time_file:
                time_value = float(time_file.read())
                total_time += time_value

        elif file_name.startswith("retrain_"):
            i += 1
            time_file_path = os.path.join(folder_path, file_name)
            with open(time_file_path, "r") as time_file:
                time_values = time_file.readlines()
                for time_value in time_values:
                    retrain_time += float(time_value)
    with open("GreedyMax.results.txt", "a") as file:
        file.write("dataset---------{}\n".format(Dataset))
        file.write("shard_number: {}\n".format(Shard_num))
        file.write(str(train_accuracy) + "\n")
        file.write(str(retrain_accuracy) + "\n")
        file.write(str(total_time) + "\n")
        file.write(str(retrain_time) + "\n")

if __name__ == "__main__":
    Model = "MLP"
    Shard_num = 4
    Epoch = 10
    Opt = "adam"
    Dataset = "datasets/adult/datasetfile"
    Batch = 128

    with open(Dataset) as f:
        datasetfile = json.loads(f.read())
    unlearn_feature = random.randint(0, int(datasetfile["original_input_shape"][0]) - 1)
    original_input_shape = int(datasetfile["original_input_shape"][0])
    full_feature_index = list(range(int(original_input_shape)))
    feature_first = random.sample(full_feature_index, Shard_num)

    max_unlearn(Model=Model, Shard_num=Shard_num, Epoch=Epoch, Opt=Opt, Dataset=Dataset, Batch=Batch,
                feature_first=feature_first, unlearn_feature=unlearn_feature)