import random
import numpy as np
import json
import os

def distribution_non_overlap(shard_num=0, distribution="uniform", dataset="datasets/purchase/datasetfile"):
    # Load dataset metadata.
    with open(dataset) as f:
        datasetfile = json.loads(f.read())

    if shard_num is not None:
        # Randomly selecting a subset of features for each shard.
        original_input_shape = int(datasetfile["original_input_shape"][0])
        input_shape = original_input_shape // shard_num  # 选择特征数量是由分片数量决定的 而不是指定数量
        all_features = list(range(original_input_shape))
        random.shuffle(all_features)
        features_re = all_features  # 缓存剩余未分组的特征
        for sh in range(shard_num):
            if not os.path.exists(
                "method/{}/featurefile/shard_{}.npy".format(shard_num, sh)
            ):
                if len(features_re) >= input_shape:
                    selected_features = features_re[:input_shape]
                else:
                    selected_features = features_re
                selected_features.sort()
                remaining_features = np.setdiff1d(all_features, selected_features)
                remaining_features.sort()
                np.save("method/{}/featurefile/shard_{}.npy".format(shard_num,sh), selected_features)
                np.save("method/{}/featurefile/remaining_feature_{}.npy".format(shard_num, sh),remaining_features)
                features_re = features_re[input_shape:]
        # If distribution is uniform, split without optimizing.
        if distribution == "uniform":
            partition = np.split(
                np.arange(0, datasetfile["nb_train"]),
                [
                    t*(datasetfile["nb_train"]//shard_num)
                    for t in range(1, shard_num)
                ],
            )
            partition_array = np.asarray(partition, dtype=object)
            np.save("method/{}/splitfile.npy".format(shard_num), partition_array, allow_pickle=True)

    else:
        partition = np.load(
            "method/{}/splitfile.npy".format(shard_num), allow_pickle=True
        )
        for shard in range(partition.shape[0]):
            selected_features = np.load("method/{}/featurefile/shard_{}.npy".format(shard_num, shard))
            np.save("method/{}/featurefile/feature-shard_{}.npy".format(shard_num, shard), selected_features)


def distribution_overlap(shard_num=0, distribution="uniform", dataset="datasets/purchase/datasetfile"):

    # Load dataset metadata.
    with open(dataset) as f:
        datasetfile = json.loads(f.read())

    if shard_num is not None:
        # Randomly selecting a subset of features for each shard.
        original_input_shape = int(datasetfile["original_input_shape"][0])
        input_shape = int(datasetfile["input_shape"][0])

        for sh in range(shard_num):
            if not os.path.exists(
                "method/{}/featurefile/shard_{}.npy".format(shard_num, sh)
            ):
                all_features = list(range(original_input_shape))
                selected_features = random.sample(all_features, input_shape)
                selected_features.sort()
                remaining_features = np.setdiff1d(all_features, selected_features)
                remaining_features.sort()
                np.save("method/{}/featurefile/shard_{}.npy".format(shard_num,sh), selected_features)
                np.save("method/{}/featurefile/remaining_feature_{}.npy".format(shard_num, sh),remaining_features)

        # If distribution is uniform, split without optimizing.
        if distribution == "uniform":
            partition = np.split(
                np.arange(0, datasetfile["nb_train"]),
                [
                    t*(datasetfile["nb_train"]//shard_num)
                    for t in range(1, shard_num)
                ],
            )
            partition_array = np.asarray(partition, dtype=object)
            np.save("method/{}/splitfile.npy".format(shard_num), partition_array, allow_pickle=True)

    else:
        partition = np.load(
            "method/{}/splitfile.npy".format(shard_num), allow_pickle=True
        )
        for shard in range(partition.shape[0]):
            selected_features = np.load("method/{}/featurefile/shard_{}.npy".format(shard_num, shard))
            np.save("method/{}/featurefile/feature-shard_{}.npy".format(shard_num, shard), selected_features)  #lmx 存储每个shard最终训练时选择的特征

