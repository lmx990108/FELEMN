import os
import numpy as np
from function import delete_directory_if_exists, unlearnOfFeature
from distribution import distribution_overlap
from algorithm import train_and_test
from aggregation import aggregation




def FELEMN(Model, Shard_num, Epoch, Opt, Dataset, Batch, unlearn_feature):
    directory_path = f"method/{Shard_num}"  # 要删除的目录路径
    delete_directory_if_exists(directory_path)

    if not os.path.isdir(f"method/{Shard_num}"):
        os.makedirs(f"method/{Shard_num}")
        os.makedirs(f"method/{Shard_num}/cache")
        os.makedirs(f"method/{Shard_num}/times")
        os.makedirs(f"method/{Shard_num}/outputs")
        os.makedirs(f"method/{Shard_num}/featurefile")


    distribution_overlap(shard_num=Shard_num, dataset=Dataset)
    unlearn = []
    unlearned_features_dict = {}


    for sh in range(Shard_num):
        train_and_test(model=Model, train=True, epochs=Epoch, batch=Batch, optimizer=Opt, shard_num=Shard_num,
                       shard_index=sh, dataset=Dataset)
        unlearned_features_dict[sh] = []
        train_and_test(test_batch=True, dataset=Dataset, batch=Batch, shard_index=sh, shard_num=Shard_num, model=Model)
        selected_features = np.load("method/{}/featurefile/shard_{}.npy".format(Shard_num, sh))
        if unlearn_feature in selected_features:
            unlearned_features_dict[sh].append(unlearn_feature)

    train_accuracy = aggregation(dataset=Dataset, shards=Shard_num)

    for j in range(Shard_num):
        if unlearned_features_dict[j] and j not in unlearn:
            unlearn.append(j)


    for i in unlearn:
        unlearnOfFeature(dataset=Dataset, shard_num=Shard_num, shard=i, unlearn_value=[unlearn_feature], dire="method")
        train_and_test(model=Model, retrain=True, epochs=Epoch, batch=Batch, optimizer=Opt, shard_num=Shard_num,
                       shard_index=i, dataset=Dataset)
    for k in range(Shard_num):
        train_and_test(train=False, test_batch=True, dataset=Dataset, batch=Batch, shard_index=k,
                       shard_num=Shard_num, model=Model)
    retrain_accuracy = aggregation(dataset=Dataset, shards=Shard_num)

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
    with open("FELEMN.results.txt", "a") as file:
        file.write("dataset---------{}\n".format(Dataset))
        file.write("shard_number: {}\n".format(Shard_num))
        file.write(str(train_accuracy) + "\n")
        file.write(str(retrain_accuracy) + "\n")
        file.write(str(total_time) + "\n")
        file.write(str(retrain_time) + "\n")
