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

    columns = ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id',
               'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty',
               'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency',
               'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
               'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
               'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
               'tolazamide','examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
               'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed',
               'label']

    # retrieve dataset
    data = pd.read_csv('diabetic_data.csv', header=None, names=columns)
    # print("Original data shape:", data.shape)


    # remove select columns
    remove_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult']
    if len(remove_cols) > 0:
        data = data.drop(columns=remove_cols)
        # test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]


    # train_nan_rows = data[data.isnull().any(axis=1)]
    # print('train nan rows: {}'.format(len(train_nan_rows)))


    # categorize attributes
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    data = data.replace('?', pd.NA).dropna()

    label = ['label']
    numeric = diag_cols
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
    np.save('diabetes130_train.npy', train_data)
    np.save('diabetes130_test.npy', test_data)


if __name__ == '__main__':
    main()
