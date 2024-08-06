import hashlib
import os
import shutil

import numpy as np
import importlib
import json
import random

def sizeOfShard(dire, shard_num, shard):
    """

    Returns the size (in number of points) of the shard before any unlearning request.
    """
    shards = np.load('{}/{}/splitfile.npy'.format(dire, shard_num), allow_pickle=True)

    return shards[shard].shape[0]


def getFeatureHash(dire, shard_num, shard):
    """

    Returns a hash of the corresponding feature indices
    """
    feature_shard = np.load('{}/{}/featurefile/shard_{}.npy'.format(dire, shard_num, shard))
    data_str = feature_shard.tostring()
    feature_hash = hashlib.sha256(data_str).hexdigest()
    return feature_hash


def fetchTrainBatch(dire, shard_num, shard, batch_size, dataset, offset=0, until=None):
    """

    Generator returning batches of points in the shard that are not in the requests
    with specified batch_size from the specified dataset.
    """
    shards = np.load('{}/{}/splitfile.npy'.format(dire, shard_num), allow_pickle=True)

    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

    if until is None or until > shards[shard].shape[0]:
        until = shards[shard].shape[0]

    limit = offset
    while limit <= until - batch_size:
        limit += batch_size
        indices = shards[shard][limit - batch_size:limit]
        yield dataloader.load(dire=dire, indices=indices, shards=shard_num, shard=shard)
    if limit < until:
        indices = shards[shard][limit:until]
        yield dataloader.load(dire=dire, indices=indices, shards=shard_num, shard=shard)


def fetchTestBatch(dire, shard_num, shard, dataset, batch_size):
    """

    Generator returning batches of points from the specified test dataset
    with specified batch_size.
    """

    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

    limit = 0
    while limit <= datasetfile['nb_test'] - batch_size:
        limit += batch_size
        yield dataloader.load(dire=dire, indices=np.arange(limit-batch_size, limit), shards=shard_num, shard=shard, category='test')
    if limit < datasetfile['nb_test']:
        yield dataloader.load(dire=dire, indices=np.arange(limit, datasetfile['nb_test']), shards=shard_num, shard=shard, category='test')


def fetchTest(shard_num, shard, dataset, index, dire):
    """
    Generator returning a single data point from the specified test dataset
    with a batch_size of 1.
    """
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))


    yield dataloader.load(dire=dire, indices=np.array([index]), shards=shard_num, shard=shard, category='test')


def createRequests(dataset, shards):
    """

    Generate unlearn request and test.
    """
    # Load dataset metadata.
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

    # Generate unlearn request and test
    request_list = []
    unlearn_num = 10
    test_num = 1000
    shard_indice = random.sample(range(shards), 3)
    shard_0_data = np.load("method/{}/featurefile/shard_{}.npy".format(shards, shard_indice[0])).tolist()
    unlearn_request = random.sample(shard_0_data, 5)

    shard_1_data = np.load("method/{}/featurefile/shard_{}.npy".format(shards, shard_indice[1])).tolist()
    unlearn_request.extend(random.sample(shard_1_data, 3))

    shard_2_data = np.load("method/{}/featurefile/shard_{}.npy".format(shards, shard_indice[2])).tolist()
    unlearn_request.extend(random.sample(shard_2_data, 2))
    random.shuffle(unlearn_request)

    nb_test = datasetfile["nb_test"]
    test_indices = random.sample(range(int(nb_test)), test_num)
    for index in test_indices:
        t_request = {"type": "test", "value": index}
        request_list.append(t_request)

    insert_indices = random.sample(range(test_num), unlearn_num)
    insert_indices.sort()
    for i, insert_index in enumerate(insert_indices):
        u_request = {"type": "unlearn", "value": unlearn_request[i]}
        request_list.insert(insert_index, u_request)

    _, test_labels = dataloader.load(dire="method", indices=test_indices, shards=shards, shard=0, category='test')
    return request_list, test_labels


def unlearnOfFeature(dire, dataset, shard_num, shard, unlearn_value):
    """

    replace the revoke feature from the remaining features
    """

    # Load dataset metadata.
    with open(dataset) as f:
        datasetfile = json.loads(f.read())

    selected_feature = np.load("{}/{}/featurefile/shard_{}.npy".format(dire, shard_num, shard))
    remaining_features = np.load("{}/{}/featurefile/remaining_feature_{}.npy".format(dire, shard_num, shard))
    values_to_replace = np.intersect1d(selected_feature, unlearn_value)
    if values_to_replace.size > 0:
        selected_remaining_feature = np.random.choice(remaining_features, size=values_to_replace.size, replace=False)
        difsection = np.setdiff1d(selected_feature, values_to_replace)
        features = np.concatenate((difsection, selected_remaining_feature))
        features.sort()
        remianings = np.setdiff1d(remaining_features, selected_remaining_feature)
        np.save("{}/{}/featurefile/shard_{}.npy".format(dire, shard_num,shard), features)
        np.save("{}/{}/featurefile/remaining_feature_{}.npy".format(dire, shard_num, shard), remianings)


def delete_directory_if_exists(directory):
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)

        except Exception as e:
            print(f"Error deleting directory: {e}")
    else:
        print(f"Directory '{directory}' not found.")


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def copy_directory_content(src, dst):
    ensure_directory_exists(dst)
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        if os.path.isdir(src_item):
            ensure_directory_exists(dst_item)  # 确保子目录存在
            copy_directory_content(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)
