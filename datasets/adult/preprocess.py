"""
Preprocesses dataset but keep continuous variables.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

def dataset_specific(random_state, test_size):

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']

    # retrieve dataset
    data = pd.read_csv('adult.data', header=None, names=columns)
    print("Original data shape:", data.shape)


    # categorize attributes
    # data = data.dropna()
    # data = data.replace('?', pd.NA).dropna()

    label = ['label']
    numeric = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical = list(set(columns) - set(numeric) - set(label))
    # print('label', label)
    # print('numeric', numeric)
    # print('categorical', categorical)

    return data, label, numeric, categorical


def main(random_state=1, test_size=0.2):

    data, label, numeric, categorical = dataset_specific(random_state=random_state,
                                                         test_size=test_size)
    X = data.drop(columns=label)
    y = data[label]

    # binarize inputs
    ct = ColumnTransformer([('kbd', KBinsDiscretizer(n_bins=10, encode='onehot', strategy='uniform'), numeric),
                            ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)])



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # binarize outputs
    X_train_processed = ct.fit_transform(X_train)
    X_test_processed = ct.transform(X_test)

    # 将标签列转换为数值
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # add labels
    train_data = np.hstack([X_train_processed, y_train_encoded.reshape(-1, 1)])
    test_data = np.hstack([X_test_processed, y_test_encoded.reshape(-1, 1)])
    #lmx 归一化
    # minmax_scaler = MinMaxScaler()
    # train_normalized = minmax_scaler.fit_transform(train)
    # test_normalized = minmax_scaler.transform(test)

    print(train_data.shape[0])
    print(test_data.shape[0])
    print(test_data.shape[1])
    print(len(np.unique(y)))

    # print('\ntrain:\n{}, dtype: {}'.format(train_data, train_data.dtype))
    # print('train.shape: {}, label sum: {}'.format(train_data.shape, train_data[:, -1].sum()))
    #
    # print('\ntest:\n{}, dtype: {}'.format(test_data, test_data.dtype))
    # print('test.shape: {}, label sum: {}'.format(test_data.shape, test_data[:, -1].sum()))
    #
    # # save to numpy format
    # print('saving...')
    np.save('adult_train.npy', train_data)
    np.save('adult_test.npy', test_data)


if __name__ == '__main__':
    main()
