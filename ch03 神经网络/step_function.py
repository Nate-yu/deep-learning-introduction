import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype=np.int64)

x = np.arange(-5.0, 5.0, 0.1) # 在 −5.0 到 5.0 的范围内，以 0.1 为单位，生成NumPy数组（[-5.0, -4.9,..., 4.9]）
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1) # 指定y轴的范围
# plt.show()