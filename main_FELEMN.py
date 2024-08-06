import json
from FELEMN import FELEMN
import random

if __name__ == '__main__':

        Model = "MLP"
        Shard_num = 4
        Epoch = 10
        Opt = "adam"
        Dataset = "datasets/adult/datasetfile"
        Batch = 128

        with open(Dataset) as f:
            datasetfile = json.loads(f.read())
        unlearn_feature = random.randint(0, int(datasetfile["original_input_shape"][0]))
        FELEMN(Model=Model, Shard_num=Shard_num, Epoch=Epoch, Opt=Opt, Dataset=Dataset, Batch=Batch,
                          unlearn_feature=unlearn_feature)

