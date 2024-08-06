import os
import numpy as np
from function import unlearnOfFeature, delete_directory_if_exists
from algorithm import train_and_test
import time


def OBO_hardvoting(Model, Shard_num, Epoch, Opt, Dataset, Batch, request_list, test_labels):

    notUnlearnShards = []
    unlearnShards = []
    outputs = []
    unlearnedFeaturesIndex = {}

    for sh in range(Shard_num):
        unlearnedFeaturesIndex[sh] = []
    i = 0
    total_test_time = 0
    total_processing_time = 0

    while 0 <= i < len(request_list):
        start_time = time.time()
        request = request_list[i]

        if request["type"] == "test":
            a_start_time = time.time()

            argmax_sum = None
            for j in range(Shard_num):
                output = train_and_test(model=Model, test_single=True, dataset=Dataset, batch=Batch, shard_index=j,
                                        shard_num=Shard_num, test_index=request["value"], retrain_dir="OBO_hardvoting", output_type="argmax")
                if argmax_sum is None:
                    argmax_sum = output
                else:
                    argmax_sum += output

            argmax_sum = np.sum(argmax_sum, axis=0)
            label = np.argmax(argmax_sum)
            outputs.append(label)
            i += 1
            single_test_time = time.time() - a_start_time
            total_test_time += single_test_time


        if request["type"] == "unlearn":
            unlearn_feature = request["value"]
            for sh in range(Shard_num):
                selected_features = np.load("OBO_hardvoting/{}/featurefile/shard_{}.npy".format(Shard_num, sh))
                if unlearn_feature in selected_features:
                    unlearnedFeaturesIndex[sh].append(unlearn_feature)
            for sh in range(Shard_num):
                if not unlearnedFeaturesIndex[sh] and sh not in notUnlearnShards:
                    notUnlearnShards.append(sh)
                if unlearnedFeaturesIndex[sh] and sh not in unlearnShards:
                    unlearnShards.append(sh)
            notUnlearnShards.sort()
            unlearnShards.sort()
            for j in unlearnShards:
                unlearnOfFeature(dataset=Dataset, shard_num=Shard_num, shard=j,
                                 unlearn_value=unlearnedFeaturesIndex[j], dire="OBO_hardvoting")
                train_and_test(model=Model, epochs=Epoch, batch=Batch, optimizer=Opt, shard_num=Shard_num, shard_index=j,
                               dataset=Dataset, retrain=True, retrain_dir="OBO_hardvoting")
                unlearnedFeaturesIndex[j] = []


            notUnlearnShards = []
            unlearnShards = []
            i += 1

        total_processing_time += time.time() - start_time
    accuracy = np.sum(outputs == test_labels) / len(test_labels)


    total_time = 0
    retrain_time = 0
    folder_path = "OBO_hardvoting/{}/times".format(Shard_num)

    for file_name in os.listdir(folder_path):
        if file_name.startswith("shard_"):
            time_file_path = os.path.join(folder_path, file_name)
            with open(time_file_path, "r") as time_file:
                time_value = float(time_file.read())
                total_time += time_value

        elif file_name.startswith("retrain_"):
            time_file_path = os.path.join(folder_path, file_name)
            with open(time_file_path, "r") as time_file:
                time_values = time_file.readlines()
                for time_value in time_values:
                    retrain_time += float(time_value)
    with open("OBO_hardvoting.txt", "a") as file:
        file.write("dataset---------{}\n".format(Dataset))
        file.write("shard_number: {}\n".format(Shard_num))
        file.write(str(accuracy) + "\n")
        file.write(str(total_time) + "\n")
        file.write(str(retrain_time) + "\n")
        file.write(str(total_test_time) + "\n")
        file.write(str(total_processing_time) + "\n")

    directory_path = "OBO_hardvoting/{}".format(Shard_num)  # 要删除的目录路径
    delete_directory_if_exists(directory_path)