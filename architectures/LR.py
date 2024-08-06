from torch import nn


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_shape[0], nb_classes)  # 只有一个全连接层

    def forward(self, x):
        x = self.fc(x)
        return x  # 对于逻辑回归，不需要非线性激活函数

    # print("--------------------lr")
