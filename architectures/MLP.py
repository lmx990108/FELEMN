import random

import numpy as np
from torch import nn
from torch.nn import Module, Linear, ReLU

from torch.nn import init
import torch

class Model(Module): #pytorch的模型定义，使用了nn.Module类作为基类，实现了一个包含两层全连接层的神经网络模型
    def __init__(self, input_shape, nb_classes, *args, **kwargs): #输入形状 输出类别数作为参数传入 lmx 输入形状
        super(Model, self).__init__()#定义了两个全连接层self.fc1 self.fc2
        self.fc1 = Linear(input_shape[0], 128) #输入的特征维度，输出维度为128 lmx
        self.fc2 = Linear(128, nb_classes) #输入维度128 输出维度 最终的类别数

    def forward(self, x): #pytorch模型的前向传播函数，定义输入数据在模型中的流动方式 向前传递 还有反向传播
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)

        return x



# class Model(Module):
#     def __init__(self, input_shape, nb_classes, hidden_dim=64, *args, **kwargs):
#         super(Model, self).__init__()
#         self.fc1 = Linear(input_shape[0], hidden_dim, bias=False)
#         self.relu = ReLU()
#         self.fc2 = Linear(hidden_dim, nb_classes, bias=False)
#         init.xavier_normal_(self.fc1.weight)  # 使用默认的 Xavier 初始化
#         # print(input_shape)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x