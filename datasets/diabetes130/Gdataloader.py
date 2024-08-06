import numpy as np
import os


pwd = os.path.dirname(os.path.realpath(__file__))


train_data = np.load(os.path.join(pwd, 'diabetes130_train.npy'), allow_pickle=True)
test_data = np.load(os.path.join(pwd, 'diabetes130_test.npy'), allow_pickle=True)

X_train = train_data[:, :-1].astype(np.float32)
X_test = test_data[:, :-1].astype(np.float32)
y_train = train_data[:, -1].astype(np.int64)
y_test = test_data[:, -1].astype(np.int64)

def load(indices, category='train'):
    if category == 'train':
            return X_train[indices], y_train[indices]
    if category == 'test':
        return X_test[indices], y_test[indices]