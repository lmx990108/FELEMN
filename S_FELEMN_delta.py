import copy
import json
import os
import numpy as np
from function import createRequests, unlearnOfFeature, delete_directory_if_exists, copy_directory_content
from distribution import distribution_overlap
from algorithm import train_and_test
import time


def retrain(unlearnIndex, unlearnShards, notUnlearnShards, unlearnedFeaturesIndex, Model, Epoch, Batch, Dataset,
            Shard_num, Opt):

    unlearnOfFeature(dire="method", dataset=Dataset, shard_num=Shard_num, shard=unlearnIndex,
                     unlearn_value=unlearnedFeaturesIndex[unlearnIndex])
    train_and_test(model=Model, epochs=Epoch, batch=Batch, retrain=True, optimizer=Opt, shard_num=Shard_num,
                   shard_index=unlearnIndex, dataset=Dataset, retrain_dir="method")
    unlearnShards.remove(unlearnIndex)
    notUnlearnShards.append(unlearnIndex)
    unlearnedFeaturesIndex[unlearnIndex] = []
    return unlearnShards, notUnlearnShards, unlearnedFeaturesIndex


def test(notUnlearnShards, Model, Epoch, Batch, Dataset, Shard_num, request):
    softmax_sum = None
    for j in notUnlearnShards:
        output = train_and_test(model=Model, epochs=Epoch, batch=Batch, test_single=True, dataset=Dataset, shard_index=j,
                                shard_num=Shard_num, test_index=request["value"], retrain_dir="method")
        if softmax_sum is None:
            softmax_sum = output
        else:
            softmax_sum += output
    softmax_sum = np.sum(softmax_sum, axis=0)
    return softmax_sum

def compute_total_corr(feature_correlation, original_input_shape):
    return sum(feature_correlation[j][k] for j in range(original_input_shape) for k in range(original_input_shape))


def replace_corr(notUnlearnShards, indexOfUnlearn, Shard_num, Dataset_name, unlearnedIndex, threshold):

    feature_correlation = np.load("datasets/{}/mutual_information.npy".format(Dataset_name))
    unlearn_common = []

    selected_features_un = np.load("method/{}/featurefile/shard_{}.npy".format(Shard_num, indexOfUnlearn))
    not_in_common = np.setdiff1d(selected_features_un, unlearnedIndex)
    k_len = len(not_in_common)
    for i in notUnlearnShards:
        selected_features_no = np.load("method/{}/featurefile/shard_{}.npy".format(Shard_num, i))
        correlations_sum = sum(feature_correlation[j][k] for j in not_in_common for k in selected_features_no)
        if correlations_sum > threshold * k_len:
            unlearn_common.append(i)

    return unlearn_common


def S_FELEMN_delta(Model, Shard_num, Epoch, Opt, Dataset, Batch, Dataset_name):
    directory_path = f"method/{Shard_num}"
    delete_directory_if_exists(directory_path)

    if not os.path.isdir(f"method/{Shard_num}"):
        os.makedirs(f"method/{Shard_num}")
        os.makedirs(f"method/{Shard_num}/cache")
        os.makedirs(f"method/{Shard_num}/times")
        os.makedirs(f"method/{Shard_num}/featurefile")

    distribution_overlap(shard_num=Shard_num, distribution="uniform", dataset=Dataset)
    notUnlearnShards = []
    unlearnShards = []
    outputs = []
    unlearnedFeaturesIndex = {}

    for sh in range(Shard_num):
        train_and_test(model=Model, epochs=Epoch, batch=Batch, optimizer=Opt, shard_num=Shard_num, shard_index=sh,
                       dataset=Dataset, train=True, retrain_dir="method")
        unlearnedFeaturesIndex[sh] = []
    request_list, test_labels = createRequests(dataset=Dataset, shards=Shard_num)
    copy_directory_content(f"method/{Shard_num}", f"OBO_softvoting/{Shard_num}")
    copy_directory_content(f"method/{Shard_num}", f"S_FELEMN_softvoting/{Shard_num}")



    with open(Dataset) as f:
        datasetfile = json.loads(f.read())
    original_input_shape = int(datasetfile["original_input_shape"][0])
    input_shape = int(datasetfile["input_shape"][0])
    feature_correlation = np.load(f"datasets/{Dataset_name}/mutual_information.npy")
    total_correlation = compute_total_corr(feature_correlation, original_input_shape)
    threshold = total_correlation * (input_shape) / (original_input_shape ** 2)

    replace_corr_cache = {}
    last_unlearnedFeaturesIndex = None

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
                        unlearnShards, notUnlearnShards, unlearnedFeaturesIndex = retrain(unlearnIndex=j,
                                                                                          unlearnShards=unlearnShards,
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
                        isFlag = True

                        if unlearnedFeaturesIndex != last_unlearnedFeaturesIndex:
                            replace_corr_cache = {
                                index: replace_corr(notUnlearnShards=notUnlearnShards, indexOfUnlearn=index,
                                                    Shard_num=Shard_num, Dataset_name=Dataset_name,
                                                    unlearnedIndex=unlearnedFeaturesIndex[index],
                                                    threshold=threshold)
                                for index in unlearnShards
                            }
                            last_unlearnedFeaturesIndex = copy.deepcopy(unlearnedFeaturesIndex)
                        for index, value in enumerate(unlearnShards):
                            unlearnCommonShards = replace_corr_cache[value]
                            if unlearnCommonShards:
                                similar_predict = test(notUnlearnShards=unlearnCommonShards, Model=Model, Epoch=Epoch,
                                                       Batch=Batch, Dataset=Dataset, Shard_num=Shard_num,
                                                       request=request)
                                average_prediction = similar_predict / len(
                                    unlearnCommonShards)
                                softmax_sum = softmax_sum + average_prediction
                                sort_softmax = np.argsort(softmax_sum)
                                max_indice = sort_softmax[-1]
                                second_max_indice = sort_softmax[-2]

                                if softmax_sum[max_indice] - softmax_sum[second_max_indice] >= len(unlearnShards) - index - 1:
                                    outputs.append(max_indice)
                                    i += 1

                                    isFlag = False
                                    break
                        if isFlag:
                            for k in unlearnShards:
                                unlearn, notUnlearnShards, unlearnedFeaturesIndex = retrain(unlearnIndex=k,
                                                                                            unlearnShards=unlearnShards,
                                                                                            notUnlearnShards=notUnlearnShards,
                                                                                            unlearnedFeaturesIndex=unlearnedFeaturesIndex,
                                                                                            Model=Model, Epoch=Epoch,
                                                                                            Batch=Batch, Dataset=Dataset,
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
                selected_features = np.load("method/{}/featurefile/shard_{}.npy".format(Shard_num, sh))
                if unlearn_feature in selected_features:
                    unlearnedFeaturesIndex[sh].append(unlearn_feature)

                    notUnlearnShards = []
                    unlearnShards = []
            i += 1

        if i == len(request_list) - 1 and unlearnShards != []:
            for j in unlearnShards:
                unlearnOfFeature(dataset=Dataset, shard_num=Shard_num, shard=j,
                                 unlearn_value=unlearnedFeaturesIndex[j], dire="method")
                train_and_test(model=Model, epochs=Epoch, batch=Batch, optimizer=Opt, shard_num=Shard_num,
                               shard_index=j, dataset=Dataset, retrain_dir="method")
    end_time = time.time()  # 记录结束时间
    total_processing_time = end_time - start_time
    accuracy = np.sum(outputs == test_labels) / len(test_labels)  # pylint: disable=unsubscriptable-object

    total_time = 0
    retrain_time = 0
    folder_path = "method/{}/times".format(Shard_num)

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
    with open("S_FELEMN_delta.txt", "a") as file:
        file.write("dataset---------{}\n".format(Dataset))
        file.write("shard_number: {}\n".format(Shard_num))
        file.write(str(accuracy) + "\n")
        file.write(str(total_time) + "\n")
        file.write(str(retrain_time) + "\n")
        file.write(str(total_processing_time) + "\n")
    directory_path = "method/{}".format(Shard_num)
    delete_directory_if_exists(directory_path)
    return request_list, test_labels











