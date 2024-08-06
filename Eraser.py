import os
import numpy as np

from function import unlearnOfFeature, delete_directory_if_exists
from algorithm import train_and_test
import time

def retrain(unlearnIndex, unlearnShards, notUnlearnShards, unlearnedFeaturesIndex, Model, Epoch, Batch, Dataset,
            Shard_num, Opt):

    unlearnOfFeature(dire="Eraser", dataset=Dataset, shard_num=Shard_num, shard=unlearnIndex,
                     unlearn_value=unlearnedFeaturesIndex[unlearnIndex])
    train_and_test(model=Model, epochs=Epoch, batch=Batch, retrain=True, optimizer=Opt, shard_num=Shard_num,
                   shard_index=unlearnIndex, dataset=Dataset, retrain_dir="Eraser", output_type="argmax")
    unlearnShards.remove(unlearnIndex)
    notUnlearnShards.append(unlearnIndex)
    unlearnedFeaturesIndex[unlearnIndex] = []
    return unlearnShards, notUnlearnShards, unlearnedFeaturesIndex


def test(notUnlearnShards, Model, Epoch, Batch, Dataset, Shard_num, request):
    argmax_sum = None
    for j in notUnlearnShards:
        output = train_and_test(model=Model, epochs=Epoch, batch=Batch, test_single=True, dataset=Dataset, shard_index=j,
                                shard_num=Shard_num, test_index=request["value"], retrain_dir="Eraser", output_type="argmax")
        if argmax_sum is None:
            argmax_sum = output
        else:
            argmax_sum += output
    argmax_sum = np.sum(argmax_sum, axis=0)
    return argmax_sum


def Eraser(Model, Shard_num, Epoch, Opt, Dataset, Batch, request_list, test_labels):

    notUnlearnShards = []
    unlearnShards = []
    outputs = []
    unlearnedFeaturesIndex = {}

    for sh in range(Shard_num):
        unlearnedFeaturesIndex[sh] = []

    total_test_time = 0
    total_processing_time = 0
    i = 0
    isFlag = True

    while 0 <= i < len(request_list):
        start_time = time.time()
        test2_time = 0
        test3_time = 0
        unlearn1_time = 0
        unlearn2_time = 0
        request = request_list[i]
        if request["type"] == "test":
            a_start_time = time.time()
            for sh in range(Shard_num):
                if not unlearnedFeaturesIndex[sh] and sh not in notUnlearnShards:
                    notUnlearnShards.append(sh)
                if unlearnedFeaturesIndex[sh] and sh not in unlearnShards:
                    unlearnShards.append(sh)
            notUnlearnShards.sort()
            unlearnShards.sort()
            test1_time = time.time() - a_start_time

            b_start_time = time.time()
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
                    unlearn1_time = time.time() - b_start_time
                else:
                    dirty_argmax_sum = test(notUnlearnShards=unlearnShards, Model=Model, Epoch=Epoch, Batch=Batch,
                                            Dataset=Dataset, Shard_num=Shard_num, request=request)
                    clean_argmax_sum = test(notUnlearnShards=notUnlearnShards, Model=Model, Epoch=Epoch, Batch=Batch,
                                            Dataset=Dataset, Shard_num=Shard_num, request=request)

                    all_argmax_sum = dirty_argmax_sum + clean_argmax_sum
                    sorted_indices = np.argsort(all_argmax_sum)
                    max_index = sorted_indices[-1]
                    test_label_dirty = None
                    testFlag = True
                    gama1 = dirty_argmax_sum[max_index]

                    for a in range(2, len(all_argmax_sum)+1):
                        k_index = sorted_indices[-a]
                        gama = all_argmax_sum[max_index] - all_argmax_sum[k_index]

                        gama3 = sum(dirty_argmax_sum) - dirty_argmax_sum[max_index] - dirty_argmax_sum[k_index]
                        if 2*gama1 + gama3 > gama:

                            for j in unlearnShards:
                                c_start_time = time.time()
                                unlearnShards, notUnlearnShards, unlearnedFeaturesIndex = retrain(unlearnIndex=j,
                                                                                                  unlearnShards=unlearnShards,
                                                                                                  notUnlearnShards=notUnlearnShards,
                                                                                                  unlearnedFeaturesIndex=unlearnedFeaturesIndex,
                                                                                                  Model=Model,
                                                                                                  Epoch=Epoch,
                                                                                                  Batch=Batch,
                                                                                                  Dataset=Dataset,
                                                                                                  Shard_num=Shard_num,
                                                                                                  Opt=Opt)
                                unlearn2_time += time.time() - c_start_time

                                output = train_and_test(model=Model, epochs=Epoch, batch=Batch, test_single=True, dataset=Dataset, shard_index=j,
                                                        shard_num=Shard_num, test_index=request["value"], retrain_dir="method", output_type="argmax")
                                if test_label_dirty is None:
                                    test_label_dirty = output
                                else:
                                    test_label_dirty += output

                            test_label_dirty = np.sum(test_label_dirty, axis=0)

                            testFlag = False
                            isFlag = False
                            break
                    if testFlag == True:
                        outputs.append(max_index)
                        i += 1
                test2_time = time.time() - b_start_time - unlearn1_time - unlearn2_time

            else:
                d_start_time = time.time()
                if isFlag:
                    argmax_sum = test(notUnlearnShards=notUnlearnShards, Model=Model, Epoch=Epoch, Batch=Batch,
                                      Dataset=Dataset, Shard_num=Shard_num, request=request)
                    max_index = np.argmax(argmax_sum)
                    outputs.append(max_index)
                    i += 1
                else:
                    argmax_sum = clean_argmax_sum + test_label_dirty
                    isFlag = True
                    argmax_sum = np.sum(argmax_sum, axis=0)
                    max_index = np.argmax(argmax_sum)
                    outputs.append(max_index)
                    i += 1

                test3_time = time.time() - d_start_time
            single_test_time = test1_time + test2_time + test3_time

            total_test_time += single_test_time


        if request["type"] == "unlearn":
            unlearn_feature = request["value"]
            for sh in range(Shard_num):
                selected_features = np.load("Eraser/{}/featurefile/shard_{}.npy".format(Shard_num, sh))
                if unlearn_feature in selected_features:
                    unlearnedFeaturesIndex[sh].append(unlearn_feature)

                    notUnlearnShards = []
                    unlearnShards = []
            i += 1
        if i == len(request_list) - 1 and unlearnShards != []:
            for j in unlearnShards:
                unlearnOfFeature(dataset=Dataset, shard_num=Shard_num, shard=j,
                                 unlearn_value=unlearnedFeaturesIndex[j], dire="Eraser")
                train_and_test(model=Model, epochs=Epoch, batch=Batch, optimizer=Opt, shard_num=Shard_num, shard_index=j,
                               dataset=Dataset, retrain=True, retrain_dir="Eraser")

        total_processing_time += time.time() - start_time
    accuracy = np.sum(outputs == test_labels) / len(test_labels) # pylint: disable=unsubscriptable-object


    total_time = 0
    retrain_time = 0
    folder_path = "Eraser/{}/times".format(Shard_num)

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
    with open("Eraser.txt", "a") as file:
        file.write("dataset---------{}\n".format(Dataset))
        file.write("shard_number: {}\n".format(Shard_num))
        file.write(str(accuracy) + "\n")
        file.write(str(total_time) + "\n")
        file.write(str(retrain_time) + "\n")
        file.write(str(total_test_time) + "\n")
        file.write(str(total_processing_time) + "\n")
    directory_path = "Eraser/{}".format(Shard_num)
    delete_directory_if_exists(directory_path)



