import numpy as np

from sklearn.metrics import mutual_info_score
data = np.load("datasets/purchase/purchase2_train.npy", allow_pickle=True)
data = data.reshape((1,))[0]
data = data['X'].astype(np.float32)


# Calculating the mutual information between features
def mutual_info(data):
    n_features = data.shape[1]
    mi_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            mi = mutual_info_score(data[:, i], data[:, j])
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    return mi_matrix


mi_matrix = mutual_info(data)
np.save("datasets/purchase/mutual_information.npy", mi_matrix)

