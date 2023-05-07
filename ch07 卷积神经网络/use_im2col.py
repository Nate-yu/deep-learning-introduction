import sys
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction") # 为了导入父目录的文件而进行的设定
from common.util import im2col
import numpy as np

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1,5,5,stride=1,pad=0)
print(col1.shape) # (9,75)

x2 = np.random.rand(10,3,7,7)
col1 = im2col(x2,5,5,stride=1,pad=0)
print(col1.shape) # (90,75)
