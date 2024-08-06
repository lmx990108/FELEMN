from S_FELEMN_hardvoting import S_FELEMN_hardvoting
from OBO_hardvoting import OBO_hardvoting
from Eraser import Eraser

if __name__ == '__main__':

    Model = "MLP"
    Shard_num = 4
    Epoch = 10
    Opt = "adam"
    Dataset = "datasets/adult/datasetfile"

    Batch = 128

    request_list, test_labels = S_FELEMN_hardvoting(Model=Model, Shard_num=Shard_num, Epoch=Epoch, Opt=Opt, Dataset=Dataset,
                                                    Batch=Batch)

    OBO_hardvoting(Model=Model, Shard_num=Shard_num, Epoch=Epoch, Opt=Opt, Dataset=Dataset, Batch=Batch,
                   request_list=request_list, test_labels=test_labels)

    Eraser(Model=Model, Shard_num=Shard_num, Epoch=Epoch, Opt=Opt, Dataset=Dataset, Batch=Batch,
           request_list=request_list, test_labels=test_labels)


