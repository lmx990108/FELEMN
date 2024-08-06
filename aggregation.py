import numpy as np
import json
import os
import importlib

def aggregation(dire="method", strategy="uniform", dataset="datasets/purchase/datasetfile",
                shards=4):
    # Load dataset metadata.
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module(
        ".".join(dataset.split("/")[:-1] + [datasetfile["dataloader"]])
    )

    # Output files used for the vote.
    filenames = ["shard_{}.npy".format(i) for i in range(shards)]#加载所有分片的输出

# Concatenate output files.
    outputs = []
    for filename in filenames:
        outputs.append(
            np.load(
                os.path.join("{}/{}/outputs".format(dire, shards), filename),
                allow_pickle=True,
            )
        )
    outputs = np.array(outputs)

    if strategy == "uniform":  # 将所有shard权重设置为相等的值
        weights = (
                1 / outputs.shape[0] * np.ones((outputs.shape[0],))
        )  # pylint: disable=unsubscriptable-object

    elif strategy.startswith("models:"):  # models：指定特定的模型进行投票
        models = np.array(strategy.split(":")[1].split(",")).astype(int)
        weights = np.zeros((outputs.shape[0],))  # pylint: disable=unsubscriptable-object
        weights[models] = 1 / models.shape[0]  # pylint: disable=unsubscriptable-object
    elif strategy == "proportional":
        split = np.load(
            "{}/{}/splitfile.npy".format(dire, shards), allow_pickle=True
        )
        weights = np.array([shard.shape[0] for shard in split])

    # Tensor contraction of outputs and weights (on the shard dimension). 根据投票策略计算模型的准确率
    votes = np.argmax(
        np.tensordot(weights.reshape(1, weights.shape[0]), outputs, axes=1), axis=2
    ).reshape(
        (outputs.shape[1],)
    )  # pylint: disable=unsubscriptable-object

    # Load labels.
    _, labels = dataloader.load(dire=dire, indices=np.arange(datasetfile["nb_test"]), shards=shards, shard=0, category="test")

    accuracy = (
            np.where(votes == labels)[0].shape[0] / outputs.shape[1]
    )  # pylint: disable=unsubscriptable-object

    return accuracy
