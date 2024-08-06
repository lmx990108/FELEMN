
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder


def main():
    # 1. 读取数据
    numeric = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    categorical = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                   35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    data = np.loadtxt('covtype.data', delimiter=',')
    # print(data)

    # 2. 拆分数据
    X = data[:, :-1]  # 特征
    y = data[:, -1] # 标签

    # print(f"数据标签范围: {y.min()} - {y.max()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ct = ColumnTransformer([
        ('kbd', KBinsDiscretizer(n_bins=10, encode='onehot', strategy='uniform'), numeric),
        ('passthrough', 'passthrough', categorical)  # 'passthrough' 直接通过未指定处理的列
    ])
    X_train_processed = ct.fit_transform(X_train)
    X_test_processed = ct.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # add labels
    train_data = np.hstack([X_train_processed, y_train_encoded.reshape(-1, 1)])
    test_data = np.hstack([X_test_processed, y_test_encoded.reshape(-1, 1)])

    # 3. 归一化处理
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # print(X_train)
    # print(y_train)
    # print(y_test)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    # X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    # y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    # y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # 4. 保存数据
    np.save('covtype_train.npy', train_data)
    np.save('covtype_test.npy', test_data)
    # np.save('covtype_y_train.npy', y_train)
    # np.save('covtype_y_test.npy', y_test)
    print(train_data.shape[0])
    print(test_data.shape[0])
    print(test_data.shape[1])
    print(len(np.unique(y)))

if __name__ == '__main__':
    main()