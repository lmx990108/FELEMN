import numpy as np
import os


pwd = os.path.dirname(os.path.realpath(__file__))



train_data = np.load(os.path.join(pwd, 'adult_train.npy'), allow_pickle=True)
test_data = np.load(os.path.join(pwd, 'adult_test.npy'), allow_pickle=True)

X_train = train_data[:,:-1].astype(np.float32)
X_test = test_data[:,:-1].astype(np.float32)
y_train = train_data[:,-1].astype(np.int64)
y_test = test_data[:,-1].astype(np.int64)



def load(indices, shards, shard, dire, category='train'):
    containers_directory = os.path.abspath(os.path.join(pwd, '..', '..'))
    feature_indices_path = os.path.join(containers_directory, '{}'.format(dire), str(shards), 'featurefile',
                                        'shard_{}.npy'.format(shard))
    indices = np.array(indices, dtype=int)
    if category == 'train':
        feature_indices = np.load(feature_indices_path)
        X_train_selected = X_train[indices]
        selected_features = X_train_selected[:, feature_indices]
        y_train_selected = y_train[indices]
        return selected_features, y_train_selected
    elif category == 'test':
        feature_indices = np.load(feature_indices_path)
        X_test_selected = X_test[indices]
        selected_features = X_test_selected[:, feature_indices]
        return selected_features, y_test[indices]
