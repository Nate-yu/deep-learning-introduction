import sys, os
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction")  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss
    
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1]) # 正确解标签

net = simpleNet()
""" print(net.W) # 权重参数

p = net.predict(x)
print(p)
print(np.argmax(p)) # 最大值的索引
print(net.loss(x,t)) """

f = lambda w: net.loss(x,t)
dW = numerical_gradient(f, net.W)
print(dW)