
import os
import numpy as np
from function import unlearnOfFeature, delete_directory_if_exists
from algorithm import train_and_test
import time


def retrain(unlearnIndex, unlearnShards, notUnlearnShards, unlearnedFeaturesIndex, Model, Epoch, Batch, Dataset,
            Shard_num, Opt):

    unlearnOfFeature(dire="S_FELEMN_softvoting", dataset=Dataset, shard_num=Shard_num, shard=unlearnIndex,
                     unlearn_value=unlearnedFeaturesIndex[unlearnIndex])
    train_and_test(model=Model, epochs=Epoch, batch=Batch, retrain=True, optimizer=Opt, shard_num=Shard_num,
                   shard_index=unlearnIndex, dataset=Dataset, retrain_dir="S_FELEMN_softvoting")
    unlearnShards.remove(unlearnIndex)
    notUnlearnShards.append(unlearnIndex)
    unlearnedFeaturesIndex[unlearnIndex] = []
    return unlearnShards, notUnlearnShards, unlearnedFeaturesIndex


def test(notUnlearnShards, Model, Epoch, Batch, Dataset, Shard_num, request):
    softmax_sum = None
    for j in notUnlearnShards:
        output = train_and_test(model=Model, epochs=Epoch, batch=Batch, test_single=True, dataset=Dataset, shard_index=j,
                                shard_num=Shard_num, test_index=request["value"], retrain_dir="S_FELEMN_softvoting")
        if softmax_sum is None:
            softmax_sum = output
        else:
            softmax_sum += output

    softmax_sum = np.sum(softmax_sum, axis=0)
    return softmax_sum


def S_FELEMN_softvoting(Model, Shard_num, Epoch, Opt, Dataset, Batch, request_list, test_labels):

    notUnlearnShards = []
    unlearnShards = []
    outputs = []
    unlearnedFeaturesIndex = {}

    for sh in range(Shard_num):
        unlearnedFeaturesIndex[sh] = []

    i = 0
    start_time = time.time()
    while 0 <= i < len(request_list):
        request = request_list[i]
        if request["type"] == "test":
            for sh in range(Shard_num):
                if not unlearnedFeaturesIndex[sh] and sh not in notUnlearnShards:
                    notUnlearnShards.append(sh)
                if unlearnedFeaturesIndex[sh] and sh not in unlearnShards:
                    unlearnShards.append(sh)
            notUnlearnShards.sort()
            unlearnShards.sort()

            if unlearnShards:
                if len(notUnlearnShards) <= Shard_num / 2:
                    for j in unlearnShards:
                        unlearnShards, notUnlearnShards, unlearnedFeaturesIndex = retrain(unlearnIndex=j, unlearnShards=unlearnShards,
                                                                                          notUnlearnShards=notUnlearnShards,
                                                                                          unlearnedFeaturesIndex=unlearnedFeaturesIndex,
                                                                                          Model=Model, Epoch=Epoch,
                                                                                          Batch=Batch, Dataset=Dataset,
                                                                                          Shard_num=Shard_num, Opt=Opt)
                        break
                else:
                    softmax_sum = test(notUnlearnShards=notUnlearnShards, Model=Model, Epoch=Epoch, Batch=Batch,
                                       Dataset=Dataset, Shard_num=Shard_num, request=request)
                    sorted_indices = np.argsort(softmax_sum)
                    max_index = sorted_indices[-1]
                    second_max_index = sorted_indices[-2]
                    diff = softmax_sum[max_index] - softmax_sum[second_max_index]
                    if diff >= len(unlearnShards):
                        outputs.append(max_index)
                        i += 1
                    else:
                        for j in unlearnShards:
                            unlearnShards, notUnlearnShards, unlearnedFeaturesIndex = retrain(unlearnIndex=j, unlearnShards=unlearnShards,
                                                                                              notUnlearnShards=notUnlearnShards,
                                                                                              unlearnedFeaturesIndex=unlearnedFeaturesIndex,
                                                                                              Model=Model, Epoch=Epoch,
                                                                                              Batch=Batch,Dataset=Dataset,
                                                                                              Shard_num=Shard_num, Opt=Opt)
                            break
            else:
                softmax_sum = test(notUnlearnShards=notUnlearnShards, Model=Model, Epoch=Epoch, Batch=Batch,
                                   Dataset=Dataset, Shard_num=Shard_num, request=request)
                max_index = np.argmax(softmax_sum)
                outputs.append(max_index)
                i += 1

        if request["type"] == "unlearn":
            unlearn_feature = request["value"]
            for sh in range(Shard_num):
                selected_features = np.load("S_FELEMN_softvoting/{}/featurefile/shard_{}.npy".format(Shard_num, sh))
                if unlearn_feature in selected_features:
                    unlearnedFeaturesIndex[sh].append(unlearn_feature)

                    notUnlearnShards = []
                    unlearnShards = []
            i += 1
        if i == len(request_list)-1 and unlearnShards != []:
            for j in unlearnShards:
                unlearnOfFeature(dataset=Dataset, shard_num=Shard_num, shard=j,
                                 unlearn_value=unlearnedFeaturesIndex[j], dire="S_FELEMN_softvoting")
                train_and_test(model=Model, epochs=Epoch, batch=Batch, optimizer=Opt, shard_num=Shard_num, shard_index=j,
                               dataset=Dataset, retrain=True, retrain_dir="S_FELEMN_softvoting")
    end_time = time.time()
    total_processing_time = end_time - start_time
    accuracy = np.sum(outputs == test_labels) / len(test_labels) # pylint: disable=unsubscriptable-object

    total_time = 0
    retrain_time = 0
    folder_path = "S_FELEMN_softvoting/{}/times".format(Shard_num)

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
    with open("S_FELEMN_softvoting.txt", "a") as file:
        file.write("dataset---------{}\n".format(Dataset))
        file.write("shard_number: {}\n".format(Shard_num))
        file.write(str(accuracy) + "\n")
        file.write(str(total_time) + "\n")
        file.write(str(retrain_time) + "\n")
        file.write(str(total_processing_time) + "\n")
    directory_path = "S_FELEMN_softvoting/{}".format(Shard_num)
    delete_directory_if_exists(directory_path)
