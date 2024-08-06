from S_FELEMN_delta import S_FELEMN_delta
from OBO_softvoting import OBO_softvoting
from S_FELEMN_softvoting import S_FELEMN_softvoting


if __name__ == '__main__':

    Model = "MLP"
    Shard_num = 4
    Epoch = 10
    Opt = "adam"
    Dataset = "datasets/adult/datasetfile"
    Dataset_name = "adult"
    Batch = 128

    request_list, test_labels = S_FELEMN_delta(Model=Model, Shard_num=Shard_num, Epoch=Epoch, Opt=Opt, Dataset=Dataset,
                                               Batch=Batch, Dataset_name=Dataset_name)

    OBO_softvoting(Model=Model, Shard_num=Shard_num, Epoch=Epoch, Opt=Opt, Dataset=Dataset, Batch=Batch,
                   request_list=request_list, test_labels=test_labels)
    S_FELEMN_softvoting(Model=Model, Shard_num=Shard_num, Epoch=Epoch, Opt=Opt, Dataset=Dataset, Batch=Batch,
                        request_list=request_list, test_labels=test_labels)


