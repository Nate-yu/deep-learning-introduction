import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

# f(x)在x处的切线
def tangent_line(f,x):
    # 切线在x处的斜率
    k = numerical_diff(f,x) 
    print(k)

    # 切线截距
    b = f(x) - k*x

    # 返回一个lambda函数，它接受一个参数t，返回切线在t处的函数值。
    return lambda t: k*t + b

x = np.arange(0.0,20.0,0.1)
y = function_1(x)

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.xlabel("x")
plt.ylabel("f(x)")
# 曲线
plt.plot(x,y)
# 曲线的切线
plt.plot(x,y2)
plt.show()

