<!-- TOC -->

- [1 Python知识预备](#1-python知识预备)
    - [1.1 Python的安装](#11-python的安装)
    - [1.2 Python解释器](#12-python解释器)
    - [1.3 Python脚本文件](#13-python脚本文件)
    - [1.4 NumPy](#14-numpy)
        - [1.4.1 导入NumPy](#141-导入numpy)
        - [1.4.2 NumPy数组](#142-numpy数组)
        - [1.4.3 NumPy的N维数组](#143-numpy的n维数组)
        - [1.4.4 广播](#144-广播)
        - [1.4.5 访问元素](#145-访问元素)
    - [1.5 Matplotlib](#15-matplotlib)
        - [1.5.1 绘制简单图形](#151-绘制简单图形)
        - [1.5.2 pyplot的功能](#152-pyplot的功能)
        - [1.5.3 显示图像](#153-显示图像)
- [2 感知机](#2-感知机)
    - [2.1 感知机的定义](#21-感知机的定义)
    - [2.2 简单逻辑电路](#22-简单逻辑电路)
    - [2.3 感知机的实现](#23-感知机的实现)
    - [2.4 感知机的局限性](#24-感知机的局限性)
    - [2.5 多层感知机](#25-多层感知机)
- [3 神经网络](#3-神经网络)
    - [3.1 从感知机到神经网络](#31-从感知机到神经网络)
    - [3.2 激活函数](#32-激活函数)
        - [3.2.1 sigmoid函数](#321-sigmoid函数)
        - [3.2.2 阶跃函数的实现与图像](#322-阶跃函数的实现与图像)
        - [3.2.3 sigmoid函数的实现与图像](#323-sigmoid函数的实现与图像)
        - [3.2.4 sigmoid函数和跃阶函数的比较](#324-sigmoid函数和跃阶函数的比较)
        - [3.2.5 ReLU函数](#325-relu函数)
    - [3.3 多维数组的运算](#33-多维数组的运算)
        - [3.3.1 多维数组](#331-多维数组)
        - [3.3.2 矩阵乘法](#332-矩阵乘法)
        - [3.3.3 神经网络的内积](#333-神经网络的内积)
    - [3.4 3层神经网络的实现](#34-3层神经网络的实现)
        - [3.4.1 符号确认](#341-符号确认)
        - [3.4.2 各层间信号传递的实现](#342-各层间信号传递的实现)
        - [3.4.3 代码实现小结](#343-代码实现小结)
    - [3.5 输出层设计](#35-输出层设计)
        - [3.5.1 恒等函数和softmax函数](#351-恒等函数和softmax函数)
        - [3.5.2 实现softmax函数时的注意事项](#352-实现softmax函数时的注意事项)
        - [3.5.3 softmax函数的特征](#353-softmax函数的特征)
        - [3.5.4 输出层的神经元数量](#354-输出层的神经元数量)
    - [3.6 手写数字识别](#36-手写数字识别)
        - [3.6.1 MNIST数据集](#361-mnist数据集)
        - [3.6.2 神经网络的推理处理](#362-神经网络的推理处理)
        - [3.6.3 批处理](#363-批处理)
- [4 神经网络的学习](#4-神经网络的学习)
    - [4.1 从数据中学习](#41-从数据中学习)
    - [4.2 损失函数](#42-损失函数)
        - [4.2.1 均方误差](#421-均方误差)
        - [4.2.2 交叉熵误差](#422-交叉熵误差)
        - [4.2.3 mini-batch学习](#423-mini-batch学习)
        - [4.2.4 mini-batch版交叉熵误差的实现](#424-mini-batch版交叉熵误差的实现)
    - [4.3 数值微分](#43-数值微分)
        - [4.3.1 导数](#431-导数)
        - [4.3.2 数值微分的例子](#432-数值微分的例子)
        - [4.3.3 偏导数](#433-偏导数)
    - [4.4 梯度](#44-梯度)
        - [4.4.1 梯度法](#441-梯度法)
        - [4.4.2 神经网络的梯度](#442-神经网络的梯度)
    - [4.5 学习算法的实现](#45-学习算法的实现)
        - [4.5.1 2层神经网络的类](#451-2层神经网络的类)
        - [4.5.2 mini-batch的实现](#452-mini-batch的实现)
        - [4.5.3 基于测试数据的评价](#453-基于测试数据的评价)
- [5 误差反向传播法](#5-误差反向传播法)
    - [5.1 计算图](#51-计算图)
        - [5.1.1 用计算图求解](#511-用计算图求解)
        - [5.1.2 局部计算](#512-局部计算)
        - [5.1.3 为何用计算图解题](#513-为何用计算图解题)

<!-- /TOC -->
# 1 Python知识预备
## 1.1 Python的安装
将使用的编程语言与库。

1. Python 3.x
2. NumPy（用于数值计算）
3. Matplotlib（将实验结果可视化）

Anaconda发行版。<br />[Anaconda 环境配置](https://www.yuque.com/abiny/wikclb/xp0imzu4wxouo4ag?view=doc_embed)

## 1.2 Python解释器
检查Python版本。打开终端，输入命令`python --version`，该命令会输出已安装的Python的版本信息<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681301360773-629be852-18e1-477b-844b-d42eba2dd227.png#averageHue=%23191919&clientId=u63205d68-842a-4&from=paste&height=132&id=u5e8dd190&name=image.png&originHeight=165&originWidth=535&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13408&status=done&style=none&taskId=uc9962723-718a-4fe5-861b-930e308b9de&title=&width=428)

输入命令`python`即可启动Python解释器<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681371465530-27d84266-ec22-4300-bf9e-d3c280031f56.png#averageHue=%23181818&clientId=u63116624-3e14-4&from=paste&height=168&id=ub0056b3e&name=image.png&originHeight=210&originWidth=1207&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20155&status=done&style=none&taskId=u7ec7fc0d-05d9-4c1e-ad31-ca573c7879d&title=&width=965.6)
> 关闭Python解释器时，Linux或Mac OS X的情况下输入Ctrl-D（按住Ctrl，再按D键）；Windows的情况下输入Ctrl-Z，然后按Enter键


## 1.3 Python脚本文件
新建一个`test.py`的文件，打开终端，`cd`到文件所在目录，使用`python test.py`即可执行Python程序

类：如果用户自己定义类的话，就可以自己创建数据类型。此外，也可以定义原创的方法（类的函数）和属性。Python中使用`class`关键字来定义类，类需要遵循以下格式
```python
class 类名：
	def __init__(self, 参数, ...): # 构造函数
        ...
    def 方法名1(self, 参数, ...): # 方法1
        ...
    def 方法名2(self, 参数, ...): # 方法2
        ...
```
这里有一个特殊的`__init__`方法，这是进行初始化的方法，也称为构造函数（constructor），只在生成类的实例时被调用一次。此外，在方法的第一个参数中明确地写入表示自身（自身的实例）的`self`是Python的一个特点。<br />具体代码实例：
```python
class Man:
    def __init__(self,name):
        self.name = name
        print("Initialized!")
    
    def hello(self):
        print("Hello "+ self.name + "!")

    def goodbye(self):
        print("Good-bye" + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()
```
从终端运行`man.py`<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681373379728-3ee6f331-4d85-4cb4-809c-f358c3cf9a5b.png#averageHue=%23191919&clientId=ubed99078-1978-4&from=paste&height=184&id=u0d72e8ff&name=image.png&originHeight=230&originWidth=1373&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=25821&status=done&style=none&taskId=uf6eccc05-0df9-4fb8-b313-d4a887f3a97&title=&width=1098.4)<br />这里我们定义了一个新类`Man`。上面的例子中，类Man生成了实例（对象）`m`。类Man的构造函数（初始化方法）会接收参数name，然后用这个参数初始化实例变量`self.name`。实例变量是存储在各个实例中的变量。Python 中可以像 self.name 这样，通过在 self 后面添加属性名来生成或访问实例变量。

## 1.4 NumPy
### 1.4.1 导入NumPy
`import numpy as np`<br />Python中使用`import`语句来导入库，这里的import numpy as np，直译的话就是“将numpy作为np导入”的意思。

### 1.4.2 NumPy数组
要生成NumPy数组，需要使用`np.array()`方法。`np.array()`接收Python列表作为参数，生成NumPy数组（numpy.ndarray）。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681374024600-e5872ac5-3042-4662-a2eb-36d5507395ce.png#averageHue=%23181818&clientId=ubed99078-1978-4&from=paste&height=134&id=u657cb9b2&name=image.png&originHeight=168&originWidth=412&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12039&status=done&style=none&taskId=uc6aa6201-c093-4347-bfa2-02f965d5969&title=&width=329.6)

下面是NumPy数组的算术运算的例子<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681374620471-07ec64ba-5417-4554-80fc-c0a47b282375.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=307&id=ua6b08fab&name=image.png&originHeight=384&originWidth=491&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27767&status=done&style=none&taskId=u5d3e7b50-cfa2-4a05-a2a7-5a01f09c7dc&title=&width=392.8)<br />这里需要注意的是，数组x和数组y的元素个数是相同的（两者均是元素个数为3的一维数组）。当x和y的元素个数相同时，可以对各个元素进行算术运算。如果元素个数不同，程序就会报错，所以元素个数保持一致非常重要。<br />NumPy数组与单一数值组合起来进行运算，需要在NumPy数组的各个元素和标量之间进行运算。这个功能也被称为“广播”。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681374771976-5fea14fa-504d-433f-9340-179615eb98e7.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=65&id=u0927b63c&name=image.png&originHeight=81&originWidth=306&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3835&status=done&style=none&taskId=u1066fda5-4862-491d-9dba-576cad53257&title=&width=244.8)

### 1.4.3 NumPy的N维数组
NumPy不仅可以生成一维数组（排成一列的数组），也可以生成多维数组。比如，可以生成如下的二维数组（矩阵）。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375015403-36f28ea5-0736-499d-b375-fb912683cb47.png#averageHue=%23151515&clientId=ubed99078-1978-4&from=paste&height=174&id=u6c5f5567&name=image.png&originHeight=217&originWidth=417&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12102&status=done&style=none&taskId=ueba25889-9dde-466b-88b4-a6f62ea1ffe&title=&width=333.6)<br />这里生成了一个2 × 2的矩阵A。另外，矩阵A的形状可以通过shape查看，矩阵元素的数据类型可以通过dtype查看。下面则是矩阵的算术运算<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375166295-b60edf62-f418-4c62-b0dc-a52163afba0f.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=158&id=ue8e8b997&name=image.png&originHeight=198&originWidth=390&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10618&status=done&style=none&taskId=u890432bc-1633-4231-bc54-c0c6c793f74&title=&width=312)<br />和数组的算术运算一样，矩阵的算术运算也可以在相同形状的矩阵间以对应元素的方式进行。并且，也可以通过标量（单一数值）对矩阵进行算术运算。这也是基于广播的功能。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375310039-18d8e7dc-2729-418c-90fa-fb832448714c.png#averageHue=%23181818&clientId=ubed99078-1978-4&from=paste&height=139&id=u3c252100&name=image.png&originHeight=174&originWidth=250&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=6559&status=done&style=none&taskId=ua283a115-47df-43c5-9da9-866681103ee&title=&width=200)

### 1.4.4 广播
> NumPy中，形状不同的数组之间也可以进行运算。之前的例子中，在2×2的矩阵A和标量10之间进行了乘法运算。

广播的实例<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375493160-ed6e6cf1-d24e-4a64-b9c5-ae0745edb32f.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=159&id=ud39f1815&name=image.png&originHeight=199&originWidth=350&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=9843&status=done&style=none&taskId=ub78ed560-4634-428b-bcbb-c74712c1851&title=&width=280)<br />在此运算中，一维数组B被“巧妙地”变成了和二位数组A相同的形状，然后再以对应元素的方式进行运算。综上，因为NumPy有广播功能，所以不同形状的数组之间也可以顺利地进行运算。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375562896-42e8b3fb-e43f-4ae8-b1af-0cd701b4c249.png#averageHue=%23414141&clientId=ubed99078-1978-4&from=paste&height=165&id=u21a9d845&name=image.png&originHeight=206&originWidth=1127&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24647&status=done&style=none&taskId=u7c400a42-15b9-47c6-89a9-f6d60d83ef9&title=&width=901.6)

### 1.4.5 访问元素
元素的索引从0开始。对各个元素的访问可按如下方式进行。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375748313-095cc8b4-682e-4b29-abc5-3826d00246e8.png#averageHue=%23131313&clientId=ubed99078-1978-4&from=paste&height=197&id=u8400083c&name=image.png&originHeight=246&originWidth=583&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13215&status=done&style=none&taskId=u4a0178db-cdbb-4a61-abe3-fcc0050f749&title=&width=466.4)<br />也可以使用`for`语句访问各个元素<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375823856-f20f51d2-a5b5-429d-afda-386525c7d60d.png#averageHue=%23131313&clientId=ubed99078-1978-4&from=paste&height=141&id=udc255b20&name=image.png&originHeight=176&originWidth=314&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=5267&status=done&style=none&taskId=u6fef30ca-7a7f-4600-835c-872dbf6d085&title=&width=251.2)

NumPy还可以使用数组访问各个元素。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375945066-25657b72-77f9-403d-ba25-29ae844905d8.png#averageHue=%23181818&clientId=ubed99078-1978-4&from=paste&height=120&id=u29252f3f&name=image.png&originHeight=150&originWidth=636&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14919&status=done&style=none&taskId=u7e197bb1-173c-4575-918f-934b2e76d62&title=&width=508.8)<br />运用这个标记法，可以获取满足一定条件的元素。例如，要从X中抽出大于15的元素，可以写成如下形式。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375993391-ae902c46-2e9d-49e7-b9a7-fc401218be99.png#averageHue=%23151515&clientId=ubed99078-1978-4&from=paste&height=100&id=uc256161c&name=image.png&originHeight=125&originWidth=609&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=7257&status=done&style=none&taskId=u14637be8-95c0-4ed4-b8c8-5f0b0bdd076&title=&width=487.2)<br />对NumPy数组使用不等号运算符等（上例中是X > 15）,结果会得到一个布尔型的数组。

## 1.5 Matplotlib
> Matplotlib是用于绘制图形的库，使用Matplotlib可以轻松地绘制图形和实现数据的可视化。

### 1.5.1 绘制简单图形
可以使用matplotlib的pyplot模块绘制图形。以下是一个绘制sin函数曲线的例子。
```python
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.arange(0,6,0.1) # 以0.1为单位，生成0到6的数据
y = np.sin(x)

# 绘制图形
plt.plot(x,y)
plt.show()
```
这里使用NumPy的arange方法生成了[0, 0.1, 0.2, ..., 5.8, 5.9]的数据，将其设为x。对x的各个元素，应用NumPy的sin函数`np.sin()`，将x、y的数据传给`plt.plot()`方法，然后绘制图形。最后，通过`plt.show()`显示图形。运行上述代码后，就会显示如图所示的图形。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681376893281-47d41de9-437f-46e0-930a-130f5731a341.png#averageHue=%23fcfcfc&clientId=ubed99078-1978-4&from=paste&height=480&id=u9352eb6c&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=30810&status=done&style=none&taskId=u5d403bbc-1237-4703-acbb-19293443517&title=&width=640)

### 1.5.2 pyplot的功能
在刚才的sin函数的图形中，我们尝试追加cos函数的图形，并尝试使用pyplot的添加标题和x轴标签名等其他功能。
```python
import numpy as np
import matplotlib.pyplot as plt


# 生成数据
x = np.arange(0,6,0.1) # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)


# 绘制图形
plt.plot(x,y1,label = "sin")
plt.plot(x,y2,linestyle = "--", label = "cos") # 用虚线绘制
plt.xlabel("x") # x轴标签
plt.ylabel("y") # y轴标签
plt.title('sin & cos') # 标题
plt.legend() # 添加图例
plt.show()
```
结果如图所示，图的标题、轴的标签名都被标出来了。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681376866898-0bf5e266-bbee-4cbe-bcea-9b5e57e75f4d.png#averageHue=%23fcfbfb&clientId=ubed99078-1978-4&from=paste&height=450&id=u8466cb98&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=46740&status=done&style=none&taskId=u07da9fda-3049-440e-b6ff-348a1cf2d2c&title=&width=600)

### 1.5.3 显示图像
pyplot中还提供了用于显示图像的方法imshow()。另外，可以使用matplotlib.image模块的imread()方法读入图像。代码实例如下。
```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('dataset/dog.jpg') # 读入图像
plt.imshow(img)

plt.show()
```
运行上述代码后，会显示如下图像<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681377913536-0e6c03eb-4d8f-4ce1-a4af-b1bd0c92fd1d.png#averageHue=%23dbc4b1&clientId=ubed99078-1978-4&from=paste&height=480&id=u0518a560&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=335581&status=done&style=none&taskId=u0de96ec0-6444-48b4-966c-e3c23bc4472&title=&width=640)<br />因为我的Python解释器运行在根目录下，且图片在dataset下，故图片路径为'dataset/dog.jpg'。

# 2 感知机
## 2.1 感知机的定义
> 感知机接收多个输入信号，输出一个信号。

和实际的电流不同的是，感知机的信号只有“流/不流”（1/0）两种取值。0对应“不传递信号”，1对应“传递信号”。如下图所示是一个接收两个输入信号的感知机的例子。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681440819573-d2627755-1569-47c2-9fe4-df8fea50fd57.png#averageHue=%23414141&clientId=u3972c055-bc56-4&from=paste&height=419&id=u6bf328c9&name=image.png&originHeight=524&originWidth=1493&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=35135&status=done&style=none&taskId=u9214393e-87a7-4342-ba07-825d3de6f98&title=&width=1194.4)<br />其中，x1、x2是输入信号，y 是输出信号，w1、w2是权重。图中的圆称为“神经元”或者“节点”。输入信号被送往神经元时，会被分别乘以固定的权重（w1x1、w2x2）。神经元会计算传送过来的信号的总和，只有当这个总和超过了某个界限值时，才会输出1。这也称为“神经元被激活” 。这里将这个界限值称为阈值，用符号θ表示。<br />上述即为感知机的运行原理，用数学公式表示即为如下：<br />![](https://cdn.nlark.com/yuque/__latex/b5414eef4b7284e5f0a28b65fa4257db.svg#card=math&code=y%3D%20%5Cbegin%7Bcases%7D0%20%5Cquad%20%28w_1x_1%20%2B%20w_2x_2%5Cle%5Ctheta%29%5C%5C%201%5Cquad%20%28w_1x_1%2Bw_2x_2%3E%5Ctheta%29%5Cend%7Bcases%7D&id=UYdgq)<br />感知机的多个输入信号都有各自固有的权重，这些权重发挥着控制各个信号的重要性的作用。即权重越大，对应该权重的信号的重要性就越高。

## 2.2 简单逻辑电路
> 与门：与门是有两个输入和一个输出的门电路。

下表这种输入信号和输出信号的对应表称为“真值表”。与门仅在两个输入均为1输出1，其他时候则输出0。

| ![](https://cdn.nlark.com/yuque/__latex/0e8831d88c93179dbe6c8b5e3678ca20.svg#card=math&code=x_1&id=dGM7P) | ![](https://cdn.nlark.com/yuque/__latex/b526050a1759d2db5c1ae7e883a48312.svg#card=math&code=x_2&id=eSeEX) | ![](https://cdn.nlark.com/yuque/__latex/947eb82adf8e060dd9e14a2ced68c45a.svg#card=math&code=y%0A&id=AxfoL) |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |


> 与非门：与非门就是颠倒了与门的输出。

用真值表表示的话，如下表所示，仅当x1和x2同时为1时输出0，其他时候则输出1。

| ![](https://cdn.nlark.com/yuque/__latex/0e8831d88c93179dbe6c8b5e3678ca20.svg#card=math&code=x_1&id=wTv9h) | ![](https://cdn.nlark.com/yuque/__latex/b526050a1759d2db5c1ae7e883a48312.svg#card=math&code=x_2&id=Z60st) | ![](https://cdn.nlark.com/yuque/__latex/947eb82adf8e060dd9e14a2ced68c45a.svg#card=math&code=y%0A&id=Rha2W) |
| --- | --- | --- |
| 0 | 0 | 1 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |


> 或门：或门是“只要有一个输入信号是1，输出就为1”的逻辑电路。

| ![](https://cdn.nlark.com/yuque/__latex/0e8831d88c93179dbe6c8b5e3678ca20.svg#card=math&code=x_1&id=ZQxqx) | ![](https://cdn.nlark.com/yuque/__latex/b526050a1759d2db5c1ae7e883a48312.svg#card=math&code=x_2&id=c8kdd) | ![](https://cdn.nlark.com/yuque/__latex/947eb82adf8e060dd9e14a2ced68c45a.svg#card=math&code=y%0A&id=dRINF) |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 1 |


这里决定感知机参数的并不是计算机，而是我们人。我们看着真值表这种“训练数据”，人工考虑（想到）了参数的值。而机器学习的课题就是将这个决定参数值的工作交由计算机自动进行。学习是确定合适的参数的过程，而人要做的是思考感知机的构造（模型），并把训练数据交给计算机。

## 2.3 感知机的实现
实现与门逻辑电路
```python
def AND(x1,x2):
    w1,w2,theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
print(AND(0,0)) # 输出0
print(AND(1,0)) # 输出0
print(AND(0,1)) # 输出0
print(AND(1,1)) # 输出1
```
定义一个接收参数x1和x2的AND函数，在函数内初始化参数w1, w2, theta，当输入的加权总和超过阈值时返回1，否则返回0。

将 θ 换成 -b ，改写数学公式：<br />![](https://cdn.nlark.com/yuque/__latex/801b525f01e66b7bdc6abb731efed4d1.svg#card=math&code=y%3D%20%5Cbegin%7Bcases%7D0%20%5Cquad%20%28b%2Bw_1x_1%20%2B%20w_2x_2%5Cle0%29%5C%5C%201%5Cquad%20%28b%2Bw_1x_1%2Bw_2x_2%3E0%29%5Cend%7Bcases%7D&id=WUYmS),此处，b称为**偏置**，w1和w2称为**权重**。<br />感知机会计算输入信号和权重的乘积，然后加上偏置，如果这个值大于0则输出1，否则输出0。使用NumPy逐一确认结果。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681458245530-4e527aac-d5b0-4ef4-815b-04498c6f7ac9.png#averageHue=%231b1b1b&clientId=u21cf038b-070e-4&from=paste&height=230&id=u32b8a377&name=image.png&originHeight=287&originWidth=323&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14235&status=done&style=none&taskId=u4efd5878-b72b-4457-9aca-bfc566e3d57&title=&width=258.4)<br />在NumPy数组的乘法运算中，当两个数组的元素个数相同时，各个元素分别相乘，因此`w*x`的结果就是它们的各个元素分别相乘（[0, 1] * [0.5, 0.5] => [0, 0.5]）。之后，`np.sum(w*x)`再计算相乘后的各个元素的总和。最后再把偏置加到这个加权总和上。

使用权重和偏置实现与门逻辑电路
```python
import numpy as np


def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
输出：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681462775874-b8b92d7b-64d8-4d15-af91-679f8b225cde.png#averageHue=%23222c32&clientId=u21cf038b-070e-4&from=paste&height=81&id=u1cfbea84&name=image.png&originHeight=101&originWidth=240&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3928&status=done&style=none&taskId=u662b8860-04e1-438e-bad7-b0646028804&title=&width=192)

这里把−θ命名为偏置b，但是请注意，偏置和权重w1、w2的作用是不一样的。具体地说，w1和w2是控制输入信号的重要性的参数，而偏置是调整神经元被激活的容易程度（输出信号为 1 的程度）的参数。比如，若 b 为−0.1，则只要输入信号的加权总和超过0.1，神经元就会被激活。但是如果 b 为−20.0，则输入信号的加权总和必须超过20.0，神经元才会被激活。

实现与非门和或门
```python
import numpy as np


def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = NAND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
输出：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681462830103-0a9733a1-1836-4b63-b831-9a647f5df3ca.png#averageHue=%23212b30&clientId=u21cf038b-070e-4&from=paste&height=84&id=u835799bf&name=image.png&originHeight=105&originWidth=279&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=4034&status=done&style=none&taskId=u255bca7d-21ee-4bbb-9a26-ba2a1a2515a&title=&width=223.2)

```python
import numpy as np


def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = OR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
输出：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681462714163-a8d450fa-dcb7-4379-8e6b-585ba74bc547.png#averageHue=%23222c31&clientId=u21cf038b-070e-4&from=paste&height=80&id=u597a2d12&name=image.png&originHeight=100&originWidth=239&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3962&status=done&style=none&taskId=ud777c716-6348-4356-a22f-f94c4e60754&title=&width=191.2)<br />与门、与非门、或门区别只在于权重参数的值，因此，在与非门和或门的实现中，仅设置权重和偏置的值这一点和与门的实现不同。

## 2.4 感知机的局限性
> 异或门：异或门也被称为逻辑异或电路，仅当x1或x2中的一方为1时，才会输出1（“异或”是拒绝其他的意思）。

| ![](https://cdn.nlark.com/yuque/__latex/0e8831d88c93179dbe6c8b5e3678ca20.svg#card=math&code=x_1&id=QtMmL) | ![](https://cdn.nlark.com/yuque/__latex/b526050a1759d2db5c1ae7e883a48312.svg#card=math&code=x_2&id=lFtGD) | ![](https://cdn.nlark.com/yuque/__latex/947eb82adf8e060dd9e14a2ced68c45a.svg#card=math&code=y%0A&id=Z9G2O) |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

用前面的感知机是无法实现异或门的。感知机的局限性就在于它只能表示一条直线分割的空间。

## 2.5 多层感知机
通过已有门电路的组合：<br />异或门的制作方法有很多，其中之一就是组合我们前面做好的与门、与非门、或门进行配置。与门，与非门，或门用如下图的符号表示<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681472867117-0a56e39e-d4f8-4f59-8e22-04e15c093b6a.png#averageHue=%23414141&clientId=u21cf038b-070e-4&from=paste&height=167&id=ue80250b0&name=image.png&originHeight=209&originWidth=906&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16198&status=done&style=none&taskId=ud6112002-d08f-405b-ae78-95dc944dd28&title=&width=724.8)<br />通过组合感知机（叠加层）就可以实现异或门。异或门可以通过如下所示配置来实现，这里，x1和x2表示输入信号，y表示输出信号，x1和x2是与非门和或门的输入，而与非门和或门的输出则是与门的输入。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681473004376-abd7ade6-d849-4a66-81fb-5bc2fc6adc8c.png#averageHue=%23414141&clientId=u21cf038b-070e-4&from=paste&height=193&id=u01e5cafb&name=image.png&originHeight=241&originWidth=701&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20588&status=done&style=none&taskId=u605abcec-265e-4d61-982f-8d0f5264084&title=&width=560.8)<br />验证正确性，把s1作为与非门的输出，把s2作为或门的输出，填入真值表

| ![](https://cdn.nlark.com/yuque/__latex/0e8831d88c93179dbe6c8b5e3678ca20.svg#card=math&code=x_1&id=LGOAi) | ![](https://cdn.nlark.com/yuque/__latex/b526050a1759d2db5c1ae7e883a48312.svg#card=math&code=x_2&id=YzGbC) | ![](https://cdn.nlark.com/yuque/__latex/ca3cc99a77092b4a870a2ed346911759.svg#card=math&code=s_1&id=HfXlR) | ![](https://cdn.nlark.com/yuque/__latex/1ec65b375dec2920064daf545cb0476a.svg#card=math&code=s_2&id=ZauFg) | ![](https://cdn.nlark.com/yuque/__latex/947eb82adf8e060dd9e14a2ced68c45a.svg#card=math&code=y%0A&id=naEwl) |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | 0 | 0 |
| 1 | 0 | 1 | 1 | 1 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 1 | 0 | 1 | 0 |


实现异或门
```python
from and_gate import AND
from or_gate import OR
from nand_gate import NAND


def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y


if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
输出：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681473328701-19216ab6-3511-4f32-bc11-73fc03b34e10.png#averageHue=%23212a30&clientId=u21cf038b-070e-4&from=paste&height=81&id=ubd76c574&name=image.png&originHeight=101&originWidth=303&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=4237&status=done&style=none&taskId=uf5797a2c-dd42-4300-b131-327259eb8f6&title=&width=242.4)

用感知机的方表示方法（明确显示神经元）来表示这个异或门<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681473381307-ae7c0818-dcb4-4350-929e-db0bd17e57fe.png#averageHue=%23414141&clientId=u21cf038b-070e-4&from=paste&height=404&id=ueb31e869&name=image.png&originHeight=505&originWidth=883&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=49078&status=done&style=none&taskId=u29fe963c-dba1-463d-a9a2-3282706f889&title=&width=706.4)<br />异或门是一种多层结构的神经网络。这里，将最左边的一列称为第0层，中间的一列称为第1层，最右边的一列称为第2层。实际上，与门、或门是单层感知机，而异或门是2层感知机。叠加了多层的感知机也称为多层感知机（multi-layered perceptron）。<br />在如上图所示的2层感知机中，先在第0层和第1层的神经元之间进行信号的传送和接收，然后在第1层和第2层之间进行信号的传送和接收，具体如下所示。

1. 第0层的两个神经元接收输入信号，并将信号发送至第1层的神经元。
2. 第1层的神经元将信号发送至第2层的神经元，第2层的神经元输出y。

# 3 神经网络
> 神经网络的一个重要性质是它可以自动地从数据中学习到合适的权重参数。

## 3.1 从感知机到神经网络
如下图所示，把最左边的一列称为**输入层**，最右边的一列称为**输出层**，中间的一列称为**中间层**。中间层有时候也被称为隐藏层。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681561346694-117eab84-9dbd-4db3-8220-3d159886a29d.png#averageHue=%23424242&clientId=u664bfb51-2e30-4&from=paste&height=594&id=ue568955d&name=image.png&originHeight=742&originWidth=921&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=89516&status=done&style=none&taskId=u60bc8de4-886e-4cb5-8ac4-108055ab60e&title=&width=736.8)

简化感知机数学式：![](https://cdn.nlark.com/yuque/__latex/8fc00128696670950893256c60f16a21.svg#card=math&code=y%3Dh%28b%2Bw_1x_1%2Bw_2x_2%29&id=G7WZT)，我们用一个函数来表示这种分情况的动作（超过0则输出1，否则输出0）。<br />![](https://cdn.nlark.com/yuque/__latex/ce5f91ff43c8c12fcb79e91caf39a7ab.svg#card=math&code=h%28x%29%20%3D%20%5Cbegin%7Bcases%7D%200%20%5Cquad%20%28x%20%5Cle%200%29%20%5C%5C%201%20%5Cquad%20%28x%3E0%29%5Cend%7Bcases%7D&id=GoWvF)<br />输入信号的总和会被函数h(x)转换，转换后的值就是输出y。h（x）函数会将输入信号的总和转换为输出信号，这种函数一般称为激活函数（activation function）。其作用在于决定如何来激活输入信号的总和。<br />进一步来改进上式：<br />![](https://cdn.nlark.com/yuque/__latex/c0f2fc874d8df61947b105ef97705458.svg#card=math&code=%281%29%20%5Cquad%20a%20%3D%20b%20%2B%20w_1x_1%20%2B%20w_2x_2%20%0A&id=OLjcb)<br />![](https://cdn.nlark.com/yuque/__latex/ef7e54eae38bcb740b79e4a7f316c7f2.svg#card=math&code=%282%29%20%5Cquad%20y%20%3D%20h%28x%29%0A&id=zPld6)<br />首先，式（1）计算加权输入信号和偏置的总和，记为a。然后，式（2）用h()函数将a转换为输出y。下图为明确显示激活函数的计算过程。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681562541410-54919415-68b7-42d2-9f18-249e7b918385.png#averageHue=%23424242&clientId=u664bfb51-2e30-4&from=paste&height=307&id=u6f8e6ecf&name=image.png&originHeight=613&originWidth=689&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=43209&status=done&style=none&taskId=u07b64979-c9bc-4e91-bfb8-0c8817a31cd&title=&width=345)信号的加权总和为节点a，然后节点a被激活函数h()转换成节点y。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681562849199-b81a45ea-bca0-42c8-b842-b840280fb2f5.png#averageHue=%23414141&clientId=u664bfb51-2e30-4&from=paste&height=238&id=u0c9793c0&name=image.png&originHeight=298&originWidth=1316&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=38865&status=done&style=none&taskId=uc31b44f3-42b7-4582-8c50-cf0603cb52f&title=&width=1052.8)<br />左图是一般的神经元的图，右图是在神经元内部明确显示激活函数的计算过程的图（a表示输入信号的总和，h()表示激活函数，y表示输出）

> “朴素感知机”是指单层网络，指的是激活函数使用了阶跃函数 A 的模型。“多层感知机”是指神经网络，即使用 sigmoid 函数等平滑的激活函数的多层网络。

## 3.2 激活函数
> 阶跃函数：以阈值为界，一旦输入超过阈值，就切换输出。这样的函数称为“阶跃函数”

因此，可以说感知机中使用了阶跃函数作为激活函数。也就是说，在激活函数的众多候选函数中，感知机使用了阶跃函数。
### 3.2.1 sigmoid函数
神经网络中经常使用的一个激活函数就是sigmoid函数：![](https://cdn.nlark.com/yuque/__latex/1be52342616f44d85f79cc919cbc5401.svg#card=math&code=h%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D&id=ANbZM)。神经网络中用sigmoid函数作为激活函数，进行信号的转换，转换后的信号被传送给下一个神经元。神经元的多层<br />连接的构造、信号的传递方法等，基本上和感知机是一样的。

### 3.2.2 阶跃函数的实现与图像
当输入超过0时，输出1，否则输出0。可以像下面这样简单地实现阶跃函数。
```python
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
```

将其修改为支持NumPy数组的实现。
```python
def step_function(x):
    y = x > 0
    return y.astype(np.int)
```

对上面的代码进行解读：对NumPy数组进行不等号运算后，数组的各个元素都会进行不等号运算，生成一个布尔型数组，大于0的被转换为True，小于等于0的被转换为False，从而形成一个新的布尔型数组y。但我们想要的跃阶函数是会输出int型的0或1的函数，因此需要把数组y的元素类型从布尔型转换为int型。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681611582389-7bf9f2e8-bd77-4c28-8d3f-81fed8672778.png#averageHue=%2330343c&clientId=ufb91a362-4377-4&from=paste&height=217&id=uc36410ff&name=image.png&originHeight=271&originWidth=530&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=19501&status=done&style=none&taskId=u4af4b33c-87af-4ad6-a74d-03e5707ee3b&title=&width=424)<br />如上所示，可以用astype()方法转换NumPy数组的类型。astype()方法通过参数指定期望的类型，这个例子中是np.int64型。Python中将布尔型转换为int型后，True会转换为1，False会转换为0。

用图来表示上面定义的阶跃函数，为此需要使用matplotlib库。
```python
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x>0, dtype=np.int64)


x = np.arange(-5.0, 5.0, 0.1) # 在 −5.0 到 5.0 的范围内，以 0.1 为单位，生成NumPy数组（[-5.0, -4.9,..., 4.9]）
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1) # 指定y轴的范围
plt.show()
```
	step_function()以该NumPy数组为参数，对数组的各个元素执行阶跃函数运算，并以数组形式返回运算结果。对数组x、y进行绘图，结果如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681612000678-69066074-31e3-4ba1-a84b-e509b46c4726.png#averageHue=%23fcfcfc&clientId=ufb91a362-4377-4&from=paste&height=480&id=ud36ec270&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12443&status=done&style=none&taskId=u0479c1a1-3b0e-4544-a25a-5b38578db28&title=&width=640)其值呈阶梯式变化，所以称为阶跃函数。

### 3.2.3 sigmoid函数的实现与图像
用Python表示sigmoid函数如下
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
如果在这个sigmoid函数中输入一个NumPy数组，则结果如下所示<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681635280332-bd2a8352-4cd4-4a04-b509-ce2c717358c1.png#averageHue=%2331353d&clientId=ufb91a362-4377-4&from=paste&height=170&id=u332bef14&name=image.png&originHeight=212&originWidth=593&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17342&status=done&style=none&taskId=u04174489-7b40-407b-ae1e-91de5761b90&title=&width=474.4)<br />根据NumPy 的广播功能，如果在标量和NumPy数组之间进行运算，则标量会和NumPy数组的各个元素进行运算。

画出sigmoid函数如下。
```python
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681635664998-f0b2f0c8-f623-4c5c-96cb-3f1e3e99cd2e.png#averageHue=%23fcfcfc&clientId=ufb91a362-4377-4&from=paste&height=480&id=u4ceb15e4&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20800&status=done&style=none&taskId=ud6d74a35-2e41-48fe-9a16-74a08588d17&title=&width=640)

### 3.2.4 sigmoid函数和跃阶函数的比较

1. sigmoid函数是一条平滑的曲线，随着输入发生连续性的变化。而阶跃函数以0为界，输出发生急剧性的变化。二者图像比较如下。
```python
import numpy as np
import matplotlib.pylab as plt
from sigmoid import sigmoid
from step_function import step_function


x = np.arange(-5.0,5.0,0.1)
y1 = sigmoid(x)
y2 = step_function(x)


plt.plot(x,y1)
plt.plot(x,y2,'k--')
plt.ylim(-0.1,1.1)
plt.show()
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681636235482-b59fc910-2e0c-4f38-a6a6-dacf95aaf11e.png#averageHue=%23fbfbfb&clientId=ufb91a362-4377-4&from=paste&height=480&id=u8ff07ac1&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24907&status=done&style=none&taskId=u41657456-2262-416d-a15d-34d59d8d628&title=&width=640)虚线为阶跃函数。

2. 相对于阶跃函数只能返回0或1，sigmoid函数可以返回实数，也就是说，感知机中神经元之间的流动的是0或1的二元信号，而神经网络中流动的是连续的实数值信号。
3. 从宏观上看，二者具有相似形状，输出的值取决于输入信号的重要性
4. 不管输入信号有多小，或者多大，输出信号都在0到1之间
5. 二者均为非线性函数

### 3.2.5 ReLU函数
> ReLU函数在输入大于0时，直接输出该值；在输入小于等于0时，输出0

ReLU函数可以标识为下面的式子。<br />![](https://cdn.nlark.com/yuque/__latex/6cec56607874f29ef62dc3fe9f0670b1.svg#card=math&code=h%28x%29%20%3D%20%5Cbegin%7Bcases%7D%20x%20%5Cquad%20%28x%20%3E%200%29%20%5C%5C%200%20%5Cquad%20%28x%20%5Cle%200%29%20%5Cend%7Bcases%7D&id=p11G9)<br />ReLU函数可用如下代码实现。
```python
def relu(x):
    return np.maximum(0,x)
```

maximum函数会从输入的数值中选择较大的那个值进行输出。ReLU函数图像如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681636976693-e2e19e10-84ec-4dbd-aa3c-41873ae5d997.png#averageHue=%23fdfdfd&clientId=ufb91a362-4377-4&from=paste&height=480&id=u8a5fe94b&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14548&status=done&style=none&taskId=ucabee685-461f-466e-893d-95cacf93588&title=&width=640)

## 3.3 多维数组的运算
### 3.3.1 多维数组
用NumPy来生成多维数组。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681643307918-46f3a580-8bbc-437e-87d2-5ab3e4fc4ad1.png#averageHue=%232e323a&clientId=u59ebbd16-e1e7-4&from=paste&height=216&id=u54284b0f&name=image.png&originHeight=270&originWidth=470&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15105&status=done&style=none&taskId=u1f38e056-e47c-4e0b-a5cc-bef5bd3670b&title=&width=376)<br />数组的维数可以通过`np.dim()`函数获得，数组的形状可以通过实例变量`shape`获得。在上面的例子中，A是一维数组，由4个元素构成，`A.shape`的结果是一个元组。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681643517943-81f59bae-02fb-44c2-ad37-27ccbe406e3e.png#averageHue=%232d3139&clientId=u59ebbd16-e1e7-4&from=paste&height=202&id=uc7ffd61c&name=image.png&originHeight=253&originWidth=504&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12638&status=done&style=none&taskId=ud02b3a9b-e85d-4652-ac68-064ca238cb2&title=&width=403.2)<br />这里生成了一个 3 X 2 的数组B。其第一个维度有3个元素，第二个维度有2个元素。另外，第一个维度对应第0维，第二个维度对应第1维。

### 3.3.2 矩阵乘法
利用NumPy实现矩阵乘法。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681643820556-218862d9-e6a5-4447-8350-3920dc81d6f9.png#averageHue=%2331353d&clientId=u59ebbd16-e1e7-4&from=paste&height=196&id=ucb8029d9&name=image.png&originHeight=245&originWidth=440&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16972&status=done&style=none&taskId=u2a8c35e1-e137-408f-ac05-bf03705ed5a&title=&width=352)<br />这里的A、B均为2 X 2的矩阵，其乘积可以通过NumPy的`np.dot()`函数计算。`np.dot()`接收两个NumPy数组作为参数，并返回数组的乘积。<br />注意：在两个矩阵相乘时，矩阵A的第1维和矩阵B的第0维元素个数必须一致。另外，当A是二维矩阵，B是一维数组时，对应维度的元素个数要保持一致的原则依然成立。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681645792389-f62ce452-e9da-462b-829a-2ec930281e98.png#averageHue=%232f333b&clientId=u59ebbd16-e1e7-4&from=paste&height=176&id=udd9d4a41&name=image.png&originHeight=220&originWidth=558&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16623&status=done&style=none&taskId=u9995ea01-5894-4381-af63-f0cf4796f7c&title=&width=446.4)

### 3.3.3 神经网络的内积
以如下图所示的简单神经网络为对象（省略了偏置和激活函数，只有权重），使用NumPy矩阵来实现神经网络。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681645943756-99d872ff-d923-4331-bc0a-4403f74712f9.png#averageHue=%23414141&clientId=u59ebbd16-e1e7-4&from=paste&height=348&id=u74601007&name=image.png&originHeight=435&originWidth=1034&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=52363&status=done&style=none&taskId=u35db73ca-5808-4e4d-ab21-f505b6ffc8b&title=&width=827.2)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681646153811-f01fbcb5-51d7-423f-b9dd-88040812fcc5.png#averageHue=%232e323a&clientId=u59ebbd16-e1e7-4&from=paste&height=255&id=uabae394d&name=image.png&originHeight=319&originWidth=507&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20911&status=done&style=none&taskId=uc09c5e89-8704-4e5e-9a6e-ad504b6c0be&title=&width=405.6)<br />如上所示，使用np.dot可以一次性计算出Y的结果。

## 3.4 3层神经网络的实现
3层神经网络示意图<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681718849833-835a8ff6-5e78-4e74-b979-244ac9ec3a9e.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=349&id=ub68b167d&name=image.png&originHeight=436&originWidth=843&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=58696&status=done&style=none&taskId=ue9602726-24da-4a04-a051-0607c21680c&title=&width=674.4)<br />输入层有2个神经元，第1个隐藏层有3个神经元，第2个隐藏层有2个神经元，输出层有2个神经元

### 3.4.1 符号确认
权重符号如下<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681719977951-16438890-0340-4ff1-808d-51617a665b8b.png#averageHue=%23414141&clientId=u05e7a439-d16b-4&from=paste&height=307&id=ufa409226&name=image.png&originHeight=384&originWidth=830&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=50531&status=done&style=none&taskId=u1428a267-6e8d-48c9-bb10-9e9ae8a31a6&title=&width=664)<br />权重和隐藏层的神经元的右上角有一个“(1)”，表示权重和神经元的层号（即第1层的权重、第1层的神经元）。此外，权重的右下角有两个数字，它们是后一层的神经元和前一层的神经元的索引号。![](https://cdn.nlark.com/yuque/__latex/08c3265a2052ba6be0de989d959b7eff.svg#card=math&code=w_%7B12%7D%5E%7B%281%29%7D&id=spxxh)表示前一层的第2个神经元![](https://cdn.nlark.com/yuque/__latex/9929b4550bf80849e3bbd9bdace8be77.svg#card=math&code=x_2%0A&id=FIRSh)到后一层的第1个神经元![](https://cdn.nlark.com/yuque/__latex/8e3351610d813c64e18b3c901bafc333.svg#card=math&code=a_1%5E%7B%281%29%7D&id=nxfxC)的权重。权重右下角按照“后一层的索引号、前一层的索引号”的顺序排列。

### 3.4.2 各层间信号传递的实现
下面是输入层到第1层的第一个神经元的信号传递过程<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681720475570-d9853296-24ca-46af-bc82-72792b2d10f2.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=419&id=ue9118699&name=image.png&originHeight=524&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=90058&status=done&style=none&taskId=u30f68141-080c-407c-8406-fa898e751bb&title=&width=688)<br />上图中增加了表示偏置的神经元“1”。偏置的右下角的索引号只有一个，因为前一层的偏置神经元只有一个。<br />下面用数学式表示![](https://cdn.nlark.com/yuque/__latex/8e3351610d813c64e18b3c901bafc333.svg#card=math&code=a_1%5E%7B%281%29%7D&id=yagQ5)通过加权信号和偏置的和按如下方式进行计算：![](https://cdn.nlark.com/yuque/__latex/c77460825d40735a9a69fb0a277be610.svg#card=math&code=a_1%5E%7B%281%29%7D%20%3D%20w_%7B11%7D%5E%7B1%7Dx_1%20%2B%20w_%7B12%7D%5E%7B%281%29%7Dx_2%20%2B%20b_1%5E%7B%281%29%7D%0A&id=YXvL6)。此外，如果使用矩阵的乘法运算，则可以将第1层的加权表示成下面的式子：![](https://cdn.nlark.com/yuque/__latex/25e7c3556ab706b9587766023a7ee384.svg#card=math&code=%5Cbm%20%7BA%7D%5E%7B%281%29%7D%20%3D%20%5Cbm%7BXW%7D%5E%7B%281%29%7D%20%2B%20%5Cbm%7BB%7D%5E%7B%281%29%7D&id=Ir6RH)，其中各元素如下所示<br />![](https://cdn.nlark.com/yuque/__latex/3f100ca5bf0cb8a5bae7dadc67adcd1d.svg#card=math&code=%5Cbm%7BA%7D%5E%7B%281%29%7D%20%3D%20%5Cbegin%7Bpmatrix%7Da_1%5E%7B%281%29%7D%20%26%20a_2%5E%7B%281%29%7D%20%26%20a_3%5E%7B%281%29%7D%5Cend%7Bpmatrix%7D%EF%BC%8C%5Cbm%7BX%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%20x_1%20%26%20x_2%20%5Cend%7Bpmatrix%7D%EF%BC%8C%5Cbm%7BB%7D%5E%7B%281%29%7D%3D%5Cbegin%7Bpmatrix%7D%20b_1%5E%7B%281%29%7D%20%26%20b_2%5E%7B%281%29%7D%20%26%20b_3%5E%7B%281%29%7D%20%5Cend%7Bpmatrix%7D%EF%BC%8CW%5E%7B%281%29%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%20w_%7B11%7D%5E%7B%281%29%7D%20%26w_%7B21%7D%5E%7B%281%29%7D%20%26%20w_%7B31%7D%5E%7B%281%29%7D%20%5C%5C%20w_%7B12%7D%5E%7B%281%29%7D%20%26w_%7B22%7D%5E%7B%281%29%7D%20%26%20w_%7B32%7D%5E%7B%281%29%7D%5Cend%7Bpmatrix%7D&id=ogM9q)

下面来用NumPy多维数组实现上面矩阵的乘法运算。（将输入信号、权重、偏置设置成任意值）
```python
X = np.array([1.0,0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape) # (2, 3)
print(X.shape)  # (2,)
print(B1.shape) # (3,)

A1 = np.dot(X, W1) + B1
```
W1为2 X 3的数组，X是元素个数为2的一维数组。第1层中激活函数的计算过程如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681722975139-248aaf98-4c04-4ddd-9ce1-5662001cbc99.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=496&id=u46daa2a3&name=image.png&originHeight=620&originWidth=856&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=128100&status=done&style=none&taskId=u19e7490d-cc79-4934-a5c8-4afb1c35bc3&title=&width=684.8)<br />如上图所示，隐藏层的加权和（加权信号和偏置的总和）用a表示，被激活函数转换后的信号用z表示。此外，图纸h()表示激活函数。这里使用sigmoid函数，用Python实现如下所示。
```python
Z1 = sigmoid(A1)

print(A1) # [0.3, 0.7, 1.1]
print(Z1) # [0.57444252, 0.66818777, 0.75026011]
```

下面来实现第1层到第2层的信号传递。图示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681723225327-ee1fd5bc-c0cc-450b-be54-39ee5527611c.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=464&id=udff0d667&name=image.png&originHeight=580&originWidth=851&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=132079&status=done&style=none&taskId=u20182c7a-da76-4713-b685-db4f5ba8666&title=&width=680.8)<br />除了第1层的输出（Z1）变成了第2层的输入这一点以外，这个实现和刚才的代码完全相同。
```python
X = np.array([1.0,0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
```

最后是第2层到输出层的信号传递。输出层的实现也和之前的实现基本相同。不过，最后的激活函数和之前的隐藏层有所不同。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681723450946-8a274b97-b915-41c9-9e43-ea328454b13c.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=433&id=uf7d73495&name=image.png&originHeight=541&originWidth=846&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=123080&status=done&style=none&taskId=ubff74a00-47a9-4d8a-8854-de57a5d174a&title=&width=676.8)
```python
def identity_function(x):
    return x
    
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3

Y = identity_function(A3) # 或者Y = A3
```
这里我们定义了`identity_function()`函数（也称为“恒等函数”），并将其作为输出层的激活函数。恒等函数会将输入按原样输出，这里这样实现只是为了和之前的流程保持统一。
> 一般地，回归问题可以使用恒等函数，二元分类问题可以使用sigmoid函数，多元分类问题可以使用softmax函数。


### 3.4.3 代码实现小结
```python
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, w2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, w3) + b3
	y = identity_function(a3)

	return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [0.31682708  0.69627909]
```
上面定义了`init_network()`和`forward()`函数。`init_network()`函数会进行权重和偏置的初始化，并将它们保存在字典变量network中。这个字典变量network中保存了每一层所需的参数（权重和偏置）。`forward()`函数（表示的是从输入到输出方向的传递处理）中则封装了将输入信号转换为输出信号的处理过程。

## 3.5 输出层设计
### 3.5.1 恒等函数和softmax函数
> 恒等函数会将输入按原样输出，对于输入的信息，不加以任何改动地直接输出。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681735934458-b2cd7956-824d-4541-9c3a-a88ee9095782.png#averageHue=%23414141&clientId=u05e7a439-d16b-4&from=paste&height=224&id=ud3f43a09&name=image.png&originHeight=280&originWidth=359&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18699&status=done&style=none&taskId=u3ff15381-92bf-44bb-a205-44776367db3&title=&width=287.2)

分类问题中使用的softmax函数可以用下面的数学式表示。<br />![](https://cdn.nlark.com/yuque/__latex/0134ce5b3fbc81c86adc88f5c68176b4.svg#card=math&code=y_k%20%3D%20%5Cfrac%7Be%5E%7Ba_k%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%7D%7D&id=rAlb0)<br />这个式子表示假设输出层共有n个神经元，计算第k个神经元的输出![](https://cdn.nlark.com/yuque/__latex/48e6989aee378b0671dcbc11187f8dd6.svg#card=math&code=y_k&id=LP6cI)。分子是输入信号![](https://cdn.nlark.com/yuque/__latex/eb1130b86c0023f2fc1466c5f4664eb9.svg#card=math&code=a_k&id=t4PXG)的指数函数，分母是所有输入信号的指数函数的和。softmax函数的图示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681736253756-5d6e7f44-0786-4110-8295-74d023641600.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=225&id=u1bb2d82b&name=image.png&originHeight=281&originWidth=295&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24835&status=done&style=none&taskId=u8af574ca-7cec-4c2f-8a56-8e5df98d5fb&title=&width=236)<br />softmax函数的输出通过箭头与所有的输入信号相连。从上面的数学式可以看出，输出层的各个神经元都受到所有输入信号的影响。

用Python解释器实现softmax函数如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681736653152-203f8efc-7a75-4824-9c75-2b02a03a99e0.png#averageHue=%23373b43&clientId=u05e7a439-d16b-4&from=paste&height=236&id=u841780ba&name=image.png&originHeight=295&originWidth=464&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27633&status=done&style=none&taskId=ub3278667-81bf-4c75-a14f-15eacb9af39&title=&width=371.2)<br />将其封装为`softmax()`函数。
```python
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
	return y
```

### 3.5.2 实现softmax函数时的注意事项
为了防止指数计算时的溢出，softmax函数的实现可以如下改进。<br />![](https://cdn.nlark.com/yuque/__latex/d2ef4ae5f3b96de651302b78138bfba3.svg#card=math&code=y_k%20%3D%20%5Cfrac%7Be%5E%7Ba_k%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%7D%7D%20%3D%20%5Cfrac%7BCe%5E%7Ba_k%7D%7D%7BC%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%7D%7D%20%3D%20%5Cfrac%7Be%5E%7Ba_k%2BlnC%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%2BlnC%7D%7D%20%3D%20%5Cfrac%7Be%5E%7Ba_k%2BC%27%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%2BC%27%7D%7D%28%E5%85%B6%E4%B8%AD%EF%BC%8CC%27%3DlnC%29&id=qNFH8)

这里的`C'`可以使用任何值，但是为了防止溢出，一般会使用输入信号中的最大值。具体实例如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681737161062-116991db-71ec-475a-a3dc-28c978f761db.png#averageHue=%23343840&clientId=u05e7a439-d16b-4&from=paste&height=214&id=u50e56a02&name=image.png&originHeight=267&originWidth=831&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=30728&status=done&style=none&taskId=u61af6bc7-eb73-434e-aab0-04cae6a74f1&title=&width=664.8)<br />如该例所示，通过减去输入信号中的最大值（上例中的c），我们发现原本为nan（not a number，不确定）的地方，现在被正确计算了。综上，softmax函数可以优化如下。
```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
	return y
```

### 3.5.3 softmax函数的特征
使用`softmax()`函数可以按如下方式计算神经网络的输出。
```python
import numpy as np

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
sum_y = np.sum(y)
print(sum_y)
```
输出结果：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681737713902-efca57fb-a496-4530-bdd2-d62233921ba6.png#averageHue=%23253036&clientId=u05e7a439-d16b-4&from=paste&height=70&id=u06743c5d&name=image.png&originHeight=87&originWidth=862&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16499&status=done&style=none&taskId=uce62d9c4-c8d4-4f65-8139-02889311301&title=&width=689.6)<br />如上所示，softmax函数的输出是0.0到1.0之间的实数。并且，softmax函数的输出值的总和是1。输出总和为1是softmax函数的一个重要性质。正因为有了这个性质，我们才可以把softmax函数的输出解释为“概率”。

一般而言，神经网络只把输出值最大的神经元所对应的类别作为识别结果。并且，即便使用softmax函数，输出值最大的神经元的位置也不会变。因此，神经网络在进行分类时，输出层的softmax函数可以省略。

### 3.5.4 输出层的神经元数量
> 输出层的神经元数量需要根据待解决的问题来决定。对于分类问题，输出层的神经元数量一般设定为类别的数量。

比如，对于某个输入图像，预测是图中的数字0到9中的哪一个的问题（10类别分类问题），可以像下图这样，将输出层的神经元设定为10个。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681737983461-d7d91e85-9f59-4e2f-8e87-c93efc5d1208.png#averageHue=%23414141&clientId=u05e7a439-d16b-4&from=paste&height=339&id=u0c7f65eb&name=image.png&originHeight=424&originWidth=777&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=39871&status=done&style=none&taskId=u21ab90e8-8975-462a-a855-eea2fcb9bac&title=&width=621.6)<br />如上图所示，输出层的神经元从上往下依次对应数字0, 1, . . ., 9。此外，图中输出层的神经元的值用不同的灰度表示。这个例子中神经元y2颜色最深，输出的值最大。这表明这个神经网络预测的是y2对应的类别，也就是“2”。

## 3.6 手写数字识别
下面来实现神经网络的“推理处理”。这个推理处理也称为神经网络的前向传播（forward propagation）。

### 3.6.1 MNIST数据集
> 这里使用的数据集是MNIST手写数字图像集。MNIST是机器学习领域最有名的数据集之一，被应用于从简单的实验到发表的论文研究等各种场合。

MNIST数据集是由0到9的数字图像构成的。训练图像有6万张，测试图像有1万张，这些图像可以用于学习和推理。MNIST的图像数据是28像素 × 28像素的灰度图像（1通道），各个像素的取值在0到255之间。每个图像数据都相应地标有“7”“2”“1”等标签。

使用`mnist.py`中的`load_mnist()`函数读入MNIST数据。
```python
import sys,os
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction") # 为了导入父目录的文件而进行的设定
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)


# 输出各个数据的形状
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000,)
```
首先，为了导入父目录中的文件，进行相应的设定（笔者的VSCode只能使用绝对路径，上述代码中的绝对路径可以用`os.pardir`代替`sys.path.append(os.pardir)`语句实际上是把父目录deep-learning-from-scratch加入到sys.path（Python的搜索模块的路径集）中，从而可以导入deep-learning-from-scratch下的任何目录（包括dataset目录）中的任何文件。）。然后导入dataset/mnist.py中的load_mnist函数。最后，使用load_mnist函数，读入MNIST数据集。<br />第一次调用load_mnist函数时，因为要下载MNIST数据集，所以需要接入网络。第2次及以后的调用只需读入保存在本地的文件（pickle文件）即可，因此处理所需的时间非常短。

load_mnist函数以“(训练图像,训练标签)，(测试图像，测试标签)”的形式返回读入的MNIST数据。此外，还可以像`load_mnist(normalize=True, flatten=True, one_hot_label=False)`这 样，设 置 3 个 参 数。

1. 第 1 个参数`normalize`设置是否将输入图像正规化为0.0～1.0的值。如果将该参数设置为False，则输入图像的像素会保持原来的0～255。
2. 第 2 个参数`flatten`设置是否展开输入图像（变成一维数组）。如果将该参数设置为False，则输入图像为1 × 28 × 28的三维数组；若设置为True，则输入图像会保存为由784个元素构成的一维数组。
3. 第 3 个参数`one_hot_label`设置是否将标签保存为one-hot表示（one-hot representation）。one-hot表示是仅正确解标签为1，其余皆为0的数组，就像[0,0,1,0,0,0,0,0,0,0]这样。当one_hot_label为False时，只是像7、2这样简单保存正确解标签；当one_hot_label为True时，标签则保存为one-hot表示。

下面来尝试显示MNIST图像，同时也确认一下数据。图像的显示使用PIL（Python Image Library）模块。
```python
import sys,os
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction") # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 把保存为NumPy数组的图像数据转换为PIL用的数据对象
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print(label) # 5


print(img.shape) # (784,)
img = img.reshape(28,28) # 把图像的性质变为原来的尺寸
print(img.shape) # (28, 28)


img_show(img)
```
执行上述代码之后，训练图像的第一张就会显示出来，如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681783390429-ef5bff05-c964-4200-812e-8b907f0b5b8a.png#averageHue=%23efefef&clientId=ude4bc88e-ee5e-4&from=paste&height=340&id=u4870c339&name=image.png&originHeight=425&originWidth=744&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=22445&status=done&style=none&taskId=u84f709ae-bf4b-4de8-889a-2af235f1105&title=&width=595.2)<br />需要注意的是，flatten=True时读入的图像是以一列（一维）NumPy数组的形式保存的。因此，显示图像时，需要把它变为原来的28像素 × 28像素的形状。可以通过reshape()方法的参数指定期望的形状，更改NumPy数组的形状。此外，还需要把保存为NumPy数组的图像数据转换为PIL用的数据对象，这个转换处理由`Image.fromarray()`来完成。而`np.uint8(img)`是将图像数据类型转换为8位无符号整数类型的函数，可以减少内存占用并提高计算速度。

### 3.6.2 神经网络的推理处理
神经网络的输入层有784个神经元，输出层有10个神经元。输入层的784这个数字来源于图像大小的28 × 28 = 784，输出层的10这个数字来源于10类别分类（数字0到9，共10类别）。此外，这个神经网络有2个隐藏层，第1个隐藏层有50个神经元，第2个隐藏层有100个神经元。这个50和100可以设置为任何值。<br />导入依赖库
```python
import sys, os
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction")  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
```

实现 3 个函数。
```python
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("ch03 神经网络/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network


def predict(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']


    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y
```
`init_network()`会读入保存在pickle文件sample_weight.pkl中的学习到的权重参数。这个文件中以字典变量的形式保存了权重和偏置参数。

用这3个函数来实现神经网络的推理处理。然后，评价它的识别精度（accuracy），即能在多大程度上正确分类。
```python
x, t = get_data()
network = init_network()


accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
```
输出：`Accuracy: 0.9352`（这表示有93.52 %的数据被正确分类了。）<br />具体流程：

1. 首先获得MNIST数据集，生成网络
2. 接着，用for语句逐一取出保存在x中的图像数据，用`predict()`函数进行分类。`predict()`函数以NumPy数组的形式输出各个标签对应的概率。比如输出[0.1, 0.3, 0.2, ..., 0.04]的数组，该数组表示“0”的概率为0.1，“1”的概率为0.3，等等
3. 然后，我们取出这个概率列表中的最大值的索引（第几个元素的概率最高），作为预测结果。可以用`np.argmax(x)`函数取出数组中的最大值的索引，`np.argmax(x)`将获取被赋给参数x的数组中的最大值元素的索引
4. 最后，比较神经网络所预测的答案和正确解标签，将回答正确的概率作为识别精度。

另外，在这个例子中，我们把 load_mnist 函数的参数 normalize 设置成了True。将 normalize 设置成 True 后，函数内部会进行转换，将图像的各个像素值除以255，使得数据的值在0.0～1.0的范围内。像这样把数据限定到某个范围内的处理称为正规化（normalization）。<br />此外，对神经网络的输入数据进行某种既定的转换称为预处理（pre-processing）。这里，作为对输入图像的一种预处理，我们进行了正规化。

### 3.6.3 批处理
考虑打包输入多张图像的情形。比如，我们想用`predict()`函数一次性打包处理100张图像。为此，可以把x的形状改为100 × 784，将100张图像打包作为输入数据。如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681786186999-c5cea2fe-954d-4b11-bdc4-7eab687cdf5d.png#averageHue=%23414141&clientId=ude4bc88e-ee5e-4&from=paste&height=128&id=u2e989359&name=image.png&originHeight=160&originWidth=848&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15741&status=done&style=none&taskId=uea038880-8f44-4242-922d-1565b36e08d&title=&width=678.4)<br />输入数据的形状为 100 × 784，输出数据的形状为100 × 10。这表示输入的100张图像的结果被一次性输出了。这种打包式的输入数据称为批（batch）。进行基于批处理的代码实现如下。
```python
x,t = get_data()
network = init_network()


batch_size = 100 # 批处理数量
accuracy_cnt = 0


for i in range(0, len(x), batch_size):
    x_batch = x[i: i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i: i+batch_size])


print("Accuracy: "+str(float(accuracy_cnt) / len(x)))
```

1. `range()`函数：若指定为`range(start, end)`，则会生成一个由start到end-1之间的整数构成的列表。若像`range(start, end, step)`这样指定3个整数，则生成的列表中的下一个元素会增加step指定的值。（相当于是步长）
2. 在`range()`函数生成的列表的基础上，通过`x[i:i+batch_size]`从输入数据中抽出批数据。`x[i:i+batch_n]`会取出从第i个到第i+batch_n个之间的数据。本例中是像x[0:100]、x[100:200]……这样，从头开始以100为单位将数据提取为批数据。
3. 通过argmax()获取值最大的元素的索引。不过这里需要注意的是，我们给定了参数axis=1。这指定了在100 × 10的数组中，沿着第1维方向（以第1维为轴）找到值最大的元素的索引（第0维对应第1个维度）。

举例如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681786802220-92325511-2ac0-49d3-a200-da05ffce1848.png#averageHue=%232e323a&clientId=ude4bc88e-ee5e-4&from=paste&height=131&id=u9e849c30&name=image.png&originHeight=164&originWidth=1052&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15812&status=done&style=none&taskId=u88c3d31d-1485-4dc2-b55d-4bf934873ba&title=&width=841.6)

4. 最后，我们比较一下以批为单位进行分类的结果和实际的答案。为此，需要在NumPy数组之间使用比较运算符（==）生成由True/False构成的布尔型数组，并计算True的个数。

# 4 神经网络的学习
> 这里所说的“学习”是指从训练数据中自动获取最优权重参数的过程


## 4.1 从数据中学习
> 所谓“从数据中学习”，是指可以由数据自动决定权重参数的值。


下面是两种针对机器学习任务的方法，以及神经网络（深度学习）的方法。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682038464552-6d6b5be2-8cbb-4ba5-97b7-a4286e375cd2.png#averageHue=%23444444&clientId=u3bfb9bac-dc34-4&from=paste&height=315&id=u0600d5f5&name=image.png&originHeight=394&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=35557&status=done&style=none&taskId=uad14a795-3822-4859-83d8-116ed585ebf&title=&width=691.2)<br />如图所示，神经网络直接学习图像本身。在第二个方法，即利用特征量和机器学习的方法中，特征量仍是由人工设计的，而在神经网络中，连图像中包含的重要特征量也都是由机器来学习的。

神经网络的优点是对所有的问题都可以用同样的流程来解决。比如，不管要求解的问题是识别5，还是识别狗，抑或是识别人脸，神经网络都是通过不断地学习所提供的数据，尝试发现待求解的问题的模式。

机器学习中，一般将数据分为训练数据和测试数据两部分来进行学习和实验。

1. 使用训练数据进行学习，寻找最优的参数
2. 使用测试数据评价训练得到的模型的实际能力

训练数据也可以称为监督数据。

泛化能力：是指处理未被观察过的数据（不包含在训练数据中的数据）的能力

## 4.2 损失函数
> 神经网络的学习中所用的指标称为损失函数（loss function）。这个损失函数可以使用任意函数，但一般用**均方误差**和**交叉熵误差**


### 4.2.1 均方误差
均方误差（mean squared error）如下式所示。<br />![](https://cdn.nlark.com/yuque/__latex/fb2e6d20d833b215ddd11b8e594d10a3.svg#card=math&code=E%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Csum_k%28y_k-t_k%29%5E2&id=VV4yq)<br />在这里，![](https://cdn.nlark.com/yuque/__latex/48e6989aee378b0671dcbc11187f8dd6.svg#card=math&code=y_k&id=lvWsl)是表示神经网络的输出，![](https://cdn.nlark.com/yuque/__latex/a53f422097657e8d1a469427a6ef2fe4.svg#card=math&code=t_k&id=TFjCJ)表示监督数据，k表示数据的维数。均方误差会计算神经网络的输出和正确解监督数据的各个元素之差的平方，再求总和。下面用Python实现均方误差。
```python
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```
这里参数 y 和 t 是NumPy数组。

### 4.2.2 交叉熵误差
交叉熵误差（cross entropy error）如下式所示。<br />![](https://cdn.nlark.com/yuque/__latex/d230036acf9be5da1b6dd33a36fb1aff.svg#card=math&code=E%20%3D%20-%5Csum_kt_klny_k&id=WU60Y)<br />这里，ln表示以e为敌的自然对数，![](https://cdn.nlark.com/yuque/__latex/48e6989aee378b0671dcbc11187f8dd6.svg#card=math&code=y_k&id=bSF3E)是神经网络输出，![](https://cdn.nlark.com/yuque/__latex/a53f422097657e8d1a469427a6ef2fe4.svg#card=math&code=t_k&id=AimMJ)是正确解标签。并且，![](https://cdn.nlark.com/yuque/__latex/a53f422097657e8d1a469427a6ef2fe4.svg#card=math&code=t_k&id=pQQaa)中只有正确解标签的索引为1，其他均为0。

因此，交叉熵误差实际上只计算正确解标签的输出的自然对数。也就是说，交叉熵误差的值是由正确解标签所对应的输出结果决定的。<br />自然对数图像如下如所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682041946272-e35499c8-1b16-49ba-8e80-f69d8ee6dee4.png#averageHue=%23fcfcfc&clientId=u3bfb9bac-dc34-4&from=paste&height=480&id=u0668b88f&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18983&status=done&style=none&taskId=u902c5ef3-bd76-4a6a-9a67-8eab603bbdf&title=&width=640)<br />x等于1时，y为0；随着x向0靠近，y逐渐变小。因此正确解标签对应的输出越大，CEE的值越接近0；当输出为1时，CEE为0。下面用Python实现交叉熵误差。
```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```
这里，参数 y 和 t 是NumPy数组。函数内部在计算`np.log`时，加上了一个微小值delta，这是因为当出现`np.log(0)`时，会变为负无限大的`-inf`，这样就会导致后续计算无法进行。作为保护性对策，添加一个微小值可以防止负无限大的发生。

使用`cross_entropy_error(y, t)`进行一些简单的计算<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682044355295-44486e4f-b0b6-484a-ab7d-67a86408e6f7.png#averageHue=%2332363e&clientId=u3bfb9bac-dc34-4&from=paste&height=344&id=udfe6465d&name=image.png&originHeight=430&originWidth=841&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=42271&status=done&style=none&taskId=u189e2927-b556-4d1d-8654-ad2ef4d17ff&title=&width=672.8)<br />第一个例子中，正确解标签对应的输出为0.6，此时交叉熵误差大约为0.51。第二个例子中，正确解标签对应的输出为0.1的低值，此时的交叉熵误差大约为2.3。

### 4.2.3 mini-batch学习
所有训练数据损失函数的总和，以交叉熵误差为例，可以写成下面的式子。<br />![](https://cdn.nlark.com/yuque/__latex/3575c5274d7db760988043b84d24792c.svg#card=math&code=E%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%5Csum_n%5Csum_kt_%7Bnk%7Dlny_%7Bnk%7D&id=aJlS3)<br />这里，假设数据有N个，![](https://cdn.nlark.com/yuque/__latex/285f0cdbe031a4f7e44332ccda355f94.svg#card=math&code=t_%7Bnk%7D&id=x3Iua)表示第 n 个数据的第 k 个元素的值（![](https://cdn.nlark.com/yuque/__latex/3588824c2810f242b22a9536d98a2d2b.svg#card=math&code=y_%7Bnk%7D&id=q9jUM)是神经网络的输出，![](https://cdn.nlark.com/yuque/__latex/285f0cdbe031a4f7e44332ccda355f94.svg#card=math&code=t_%7Bnk%7D&id=LkfCN)是监督数据）。

神经网络的学习也是从训练数据中选出一批数据（称为mini-batch，小批量），然后对每个mini-batch进行学习。

1. 读入MNIST数据集
```python
import sys
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction") # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)
```

2. 使用`np.random.choice`随机抽取10笔数据
```python
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```
注：使用`np.random.choice()`可以从指定的数字中随机选择想要的数字。比如，`np.random.choice(60000, 10)`会从 0 到 59999 之间随机选择 10 个数字。

3. 之后，我们只需指定这些随机选出的索引，取出mini-batch，然后使用这个mini-batch计算损失函数即可。

### 4.2.4 mini-batch版交叉熵误差的实现
实现一个可以同时处理单个数据和批量数据（数据作为batch集中输入）两种情况的函数
```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
             
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```
这里，y 是神经网络的输出，t 是监督数据。y 的维度为1时，即求单个数据的交叉熵误差时，需要改变数据的形状。并且，当输入为mini-batch时，要用batch的个数进行正规化，计算单个数据的平均交叉熵误差。

此外，当监督数据是标签形式（非one-hot表示，而是像“2”“7”这样的标签）时，交叉熵误差可通过如下代码实现。
```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```
实现的要点是，由于 one-hot 表示中 t 为0的元素的交叉熵误差也为0，因此针对这些元素的计算可以忽略。换言之，如果可以获得神经网络在正确解标签处的输出，就可以计算交叉熵误差。因此，t 为 one-hot 表示时通过`t * np.log(y)`计算的地方，在t为标签形式时，可用`np.log( y[np.arange (batch_size), t] )`实现相同的处理（为了便于观察，这里省略了微小值1e-7）。<br />`np.arange (batch_size)`会生成一个从0到 batch_size-1 的数组。比如当 batch_size 为 5 时，np.arange(batch_size) 会生成一个 NumPy 数组[0, 1, 2, 3, 4]。因为 t 中标签是以[2, 7, 0, 9, 4]的形式存储的，所以 y[np.arange(batch_size), t] 能抽出各个数据的正确解标签对应的神经网络的输出（在这个例子中，y[np.arange(batch_size), t]会 生 成 NumPy 数 组[y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]]）。

## 4.3 数值微分
### 4.3.1 导数
实现数值微分（数值梯度）
```python
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)
```

### 4.3.2 数值微分的例子
尝试用数值微分对简单函数进行求导。<br />![](https://cdn.nlark.com/yuque/__latex/b385dfc9d0465063c27d9d1d49e373b4.svg#card=math&code=y%3D0.01x%5E2%2B0.1x&id=A9wa6)<br />用Python实现上式。
```python
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x
```

绘制上式函数图像。
```python
x = np.arange(0.0,20.0,0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()
```

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682063452294-da2838cd-7404-4d42-a435-537d670a78e5.png#averageHue=%23fcfcfc&clientId=uc6be4a61-1f40-4&from=paste&height=480&id=ue1f2c96f&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=21810&status=done&style=none&taskId=u51282eea-4a98-48e9-b3ff-ac7fa9d3851&title=&width=640)

用上面的数值微分的值作为斜率，画一条直线。
```python
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
```

x=5处的切线如下<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682065463885-63e55d0e-89df-498a-a85d-eab0b74e63d2.png#averageHue=%23fcfcfb&clientId=uc6be4a61-1f40-4&from=paste&height=480&id=uaf7b73c5&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27844&status=done&style=none&taskId=u9a56b707-5ac6-4de7-a399-a0e4e9141c4&title=&width=640)

### 4.3.3 偏导数
如下一个函数，有两个变量。<br />![](https://cdn.nlark.com/yuque/__latex/d40c8e25200796ee3e5d872374b72528.svg#card=math&code=f%28x_0%2Cx_1%29%20%3D%20x_0%5E2%2Bx_1%5E2&id=WWued)<br />用Python实现如下。
```python
def function_2(x):
    return x[0] ** 2 + x[1] ** 2
    # 或者 return np.sum(x**2)
```

函数图像如下图所示<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682065831378-bc7e84ed-0f91-4893-ab6b-ee4098fed154.png#averageHue=%23fafafa&clientId=uc6be4a61-1f40-4&from=paste&height=480&id=u37a8a44e&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=153055&status=done&style=none&taskId=ub0b458ab-4322-4eaf-a150-ecc43b5ff09&title=&width=640)

求偏导数<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682066162084-9c39604d-c0cf-4a2c-a2dc-8993901259ef.png#averageHue=%23414141&clientId=uc6be4a61-1f40-4&from=paste&height=366&id=ub8b83a22&name=image.png&originHeight=457&originWidth=612&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=66679&status=done&style=none&taskId=u82c2e35a-3609-4c31-bdd0-ca11a0accc6&title=&width=489.6)<br />偏导数和单变量的导数一样，都是求某个地方的斜率。偏导数需要将多个变量中的某一个变量定为目标变量，并将其他变量固定为某个值。

## 4.4 梯度
> 像![](https://cdn.nlark.com/yuque/__latex/5344300fcb7e4e996670140397506570.svg#card=math&code=%28%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_0%7D%2C%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D%29&id=pfAZZ)这样由全部变量的偏导数汇总而成的向量称为梯度（gradient）

梯度可以像下面这样实现。
```python
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x) # 生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 还原值

    return grad
```
函数`numerical_gradient(f, x)`中，参数`f`为函数，`x`为NumPy数组，该函数对NumPy数组`x`的各个元素求数值微分。

把![](https://cdn.nlark.com/yuque/__latex/d40c8e25200796ee3e5d872374b72528.svg#card=math&code=f%28x_0%2Cx_1%29%20%3D%20x_0%5E2%2Bx_1%5E2&id=HLKMt)的梯度画在图上。
```python
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad

def numerical_gradient(f,X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def tangent_line(f, x):
    k = numerical_gradient(f, x)
    print(k)
    b = f(x) - k*x
    return lambda t: k*t + b

if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)

    # 将x0和x1两个一维数组转换为二维数组X和Y，其中X和Y的行数分别等于x1和x0的长度
    X, Y = np.meshgrid(x0, x1)
    # 将X、Y数组展平为一维数组。这个操作的目的是为了将X和Y中的每个元素组合成一个二元组
    X = X.flatten()
    Y = Y.flatten()
    
    grad = numerical_gradient(function_2, np.array([X, Y]) )
    
    # 创建一个新的图形窗口
    plt.figure()
    # 用于绘制二维向量场，其中X和Y是网格点坐标，-grad[0]和-grad[1]是对应网格点处的梯度向量
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid() # 显示网格线
    plt.legend()
    plt.draw()
    plt.show()
```

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682068570155-649f67d5-8ec3-42ac-b59b-9b7476e022b3.png#averageHue=%23f6f6f6&clientId=uc6be4a61-1f40-4&from=paste&height=480&id=uce95b296&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=88822&status=done&style=none&taskId=u42af958e-b390-48c7-ba5a-b8f2d7ea542&title=&width=640)

### 4.4.1 梯度法
> 通过巧妙地使用梯度来寻找函数最小值（或者尽可能小的值）的方法就是梯度法。通过不断地沿梯度方向前进，
> 逐渐减小函数值的过程就是梯度法（gradient method）。

这里的梯度表示的是各点处的函数值减小最多的方向。寻找最小值的梯度法称为梯度下降法（gradient descent method），<br />寻找最大值的梯度法称为梯度上升法（gradient ascent method）。

下面用数学式来表示梯度法。<br />![](https://cdn.nlark.com/yuque/__latex/5cecbce9f0330b69f8ad835c8f7af65e.svg#card=math&code=x_0%20%3D%20x_0%20-%20%5Ceta%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_0%7D%20%5Cquad%20x_1%3Dx_1-%5Ceta%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D&id=qYYT5)<br />上式中，![](https://cdn.nlark.com/yuque/__latex/96b0657b0bd8ddf567b3f27d8ad467e6.svg#card=math&code=%5Ceta%0A&id=KwB2K)表示更新量，在神经网络的学习中，称为学习率（learning rate）。学习率决定在一次学习中，应该学习多少，以及在多大程度上更新参数。

用Python实现梯度下降法
```python
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    x_history = []


    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f,x)
        x -= lr * grad
        
    return x, np.array(x_history)
```
参数`f`是要进行最优化的函数，`init_x`是初始值，`lr`是学习率learning rate，`step_num`是梯度法的重复次数。`numerical_gradient(f,x)`会求函数的梯度，用该梯度乘以学习率得到的值进行更新操作，由`step_num`指定重复的次数。

下面用梯度法求函数![](https://cdn.nlark.com/yuque/__latex/c0983a6809880f591e5e1dfa55306a5b.svg#card=math&code=f%28x_0%2Bx_1%29%20%3D%20x_0%5E2%2Bx_1%5E2&id=CZcCn)的最小值
```python
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f,x)
        x -= lr * grad

    return x, np.array(x_history)

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot([-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
```

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682077455478-f1190231-f757-4e5f-9e3a-eb6e8e8ffa7d.png#averageHue=%23fcfcfc&clientId=u4287f989-f779-4&from=paste&height=480&id=u62cd216a&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16645&status=done&style=none&taskId=uabc156d8-43a3-4c54-8ee0-d7c6800a936&title=&width=640)

注意：实验结果表明，学习率过大的话，会发散成一个很大的值；反过来，学习率过小的话，基本上没怎么更新就结束了。也就是说，设定合适的学习率是一个很重要的问题。

### 4.4.2 神经网络的梯度
> 神经网络的学习也要求梯度。这里所说的梯度是指损失函数关于权重参数的梯度。

比如，有一个只有一个形状为 2 × 3 的权重 W 的神经网络，损失函数用 L 表示。此时，梯度可以用![](https://cdn.nlark.com/yuque/__latex/36e5597d13be0b891d1dbb4a7b2f5f0d.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D&id=RhceS)表示。数学式如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682077691375-f35f6560-958b-4f4f-b60a-c611e1107d44.png#averageHue=%23414141&clientId=u4287f989-f779-4&from=paste&height=149&id=ua5c3a116&name=image.png&originHeight=186&originWidth=392&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=21322&status=done&style=none&taskId=u27eb31d3-56e9-4e52-a495-27b6975e0ea&title=&width=313.6)<br />![](https://cdn.nlark.com/yuque/__latex/36e5597d13be0b891d1dbb4a7b2f5f0d.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D&id=AgjU9)的元素由各个元素关于 W 的偏导数构成。下面以一个简单的神经网络为例，来实现求梯度的代码。
```python
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
```
	这里使用了 common/functions.py 中的`softmax`和`cross_entropy_error`方法，以及common/gradient.py中的`numerical_gradient`方法。simpleNet类只有一个实例变量，即形状为2×3的权重参数。它有两个方法，一个是用于预测的`predict(x)`，另一个是用于求损失函数值的`loss(x,t)`。这里参数`x`接收输入数据，`t`接收正确解标签。

下面来使用`simpleNet`。
```python
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1]) # 正确解标签

net = simpleNet()
print(net.W) # 权重参数

p = net.predict(x)
print(p)
print(np.argmax(p)) # 最大值的索引
print(net.loss(x,t))
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682078741991-d3dd415d-a24e-42dd-8da1-0d3f4ac96744.png#averageHue=%2328333a&clientId=u4287f989-f779-4&from=paste&height=98&id=u4ac4ff76&name=image.png&originHeight=123&originWidth=442&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17176&status=done&style=none&taskId=u2657d7cb-891c-4e8c-b377-c1b45671d49&title=&width=353.6)

接下来求梯度。使用`numerical_gradient(f, x)`求梯度（这里定义的函数f(W)的参数W是一个伪参数。因为`numerical_gradient(f, x)`会在内部执行f(x),为了与之兼容而定义了f(W)）。
```python
f = lambda w: net.loss(x,t)
dW = numerical_gradient(f, net.W)
print(dW)
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682078877356-d8b13634-9af2-42f6-ab0a-b5b29de9ccce.png#averageHue=%232c3d45&clientId=u4287f989-f779-4&from=paste&height=50&id=ud3d91515&name=image.png&originHeight=62&originWidth=406&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10340&status=done&style=none&taskId=uf8755009-d9a1-4fcc-b28f-6b72b55065c&title=&width=324.8)<br />`numerical_gradient(f, x)`的参数`f`是函数，`x`是传给函数`f`的参数。因此，这里参数`x`取`net.W`，并定义一个计算损失函数的新函数`f`，然后把这个新定义的函数传递给`numerical_gradient(f, x)`。`numerical_gradient(f, net.W)`的结果是`dW`，一个形状为2 × 3的二维数组。

## 4.5 学习算法的实现
神经网络的学习步骤：

1. mini-batch：从训练数据中随机选出一部分数据，这部分数据称为mini-batch。我们的目标是减小mini-batch的损失函数的值。
2. 计算梯度：为了减小mini-batch的损失函数的值，需要求出各个权重参数的梯度。梯度表示损失函数的值减小最多的方向。
3. 更新参数：将权重参数沿梯度方向进行微小更新。
4. 重复上述三步

因为这里使用的数据是随机选择的mini batch数据，所以又称为随机梯度下降法（stochastic gradient descent）

### 4.5.1 2层神经网络的类
首先将这个2层神经网络实现为一个名为`TwoLayerNet`的类。
```python
import sys, os
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction")  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y
    
    # x: 输入数据，t: 监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y,t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y-t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
```

下面两表罗列了重要的变量与所有方法。

| 变量 | 说明 |
| --- | --- |
| params | 保存神经网络的参数的字典型变量（实例变量）。<br />params['W1']是第1层的权重，params['b1']是第1层的偏置。<br />params['W2']是第2层的权重，params['b2']是第2层的偏置 |
| grads | 保存梯度的字典型变量（numerical_gradient()方法的返回值）。<br />grads['W1']是第1层权重的梯度，grads['b1']是第1层偏置的梯度。<br />grads['W2']是第2层权重的梯度，grads['b2']是第2层偏置的梯度 |


| 方法 | 说明 |
| --- | --- |
| __init__(self, input_size, hidden_size, output_size) | 进行初始化。<br />参数从头开始依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数 |
| predict(self, x) | 进行识别（推理）。<br />参数x是图像数据 |
| loss(self, x, t) | 计算损失函数的值。<br />参数x是图像数据，t是正确解标签（后面3个方法的参数也一样） |
| accuracy(self, x, t) | 计算识别精度 |
| numerical_gradient(self, x, t) | 计算权重参数的梯度 |
| gradient(self, x, t) | 计算权重参数的梯度。<br />numerical_gradient()的高速版，将在下一章实现 |


### 4.5.2 mini-batch的实现
下面以TwoLayerNet类为对象，使用MNIST数据集进行学习。
```python
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
(x_train, t_train), (x_test, t_test) = \ load_mnist(normalize=True, one_hot_
laobel = True)
train_loss_list = []
# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 高速版!
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
```
随着学习的进行，损失函数的值在不断减小。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682083105031-124190af-3611-4b6b-8f7a-89ec18bbcee7.png#averageHue=%23434343&clientId=u4287f989-f779-4&from=paste&height=346&id=u6160cfeb&name=image.png&originHeight=433&originWidth=879&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=57597&status=done&style=none&taskId=ub4571a2a-1753-4f8e-925c-d3b056d4788&title=&width=703.2)

### 4.5.3 基于测试数据的评价
神经网络的学习中，必须确认是否能够正确识别训练数据以外的其他数据，即确认是否会发生过拟合。
> 过拟合：虽然训练数据中的数字图像能被正确辨别，但是不在训练数据中的数字图像却无法被识别的现象。


epoch：一个单位。一个epoch表示学习中所有训练数据均被使用过一次时的更新次数。比如，对于10000笔训练数据，用大小为100笔数据的mini-batch进行学习时，重复随机梯度下降法100次，所有的训练数据就都被“看过”了A。此时，100次就是一个epoch。

```python
import sys, os
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction")  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 超参数
iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形
markers = {'train':'o','test':'s'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```
每经过一个epoch，就对所有的训练数据和测试数据计算识别精度，并记录结果。之所以要计算每一个epoch的识别精度，是因为如果在for语句的循环中一直计算识别精度，会花费太多时间。因此，我们才会每经过一个epoch就记录一次训练数据的识别精度。<br />结果用图像表示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682083553496-ba1b6605-325d-4819-89f9-1f36915481dc.png#averageHue=%23fcfbfb&clientId=u4287f989-f779-4&from=paste&height=480&id=u8b6e3a59&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=28136&status=done&style=none&taskId=u26f71e52-7496-4569-ae28-b6239baa652&title=&width=640)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682083612377-489a093e-5464-4e9f-8b45-caec4869fe5f.png#averageHue=%23272f35&clientId=u4287f989-f779-4&from=paste&height=317&id=uee5c5dc3&name=image.png&originHeight=396&originWidth=836&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=69859&status=done&style=none&taskId=u3389a6ac-a3b7-4b85-8456-f61fe8d9ba9&title=&width=669)<br />实线表示训练数据的识别精度，虚线表示测试数据的识别精度。如图所示，随着epoch的前进（学习的进行），我们发现使用训练数据和测试数据评价的识别精度都提高了，并且，这两个识别精度基本上没有差异（两条线基本重叠在一起）。因此，可以说这次的学习中没有发生过拟合的现象。

# 5 误差反向传播法
## 5.1 计算图
> 计算图将计算过程用图形表示出来。这里说的图形是数据结构图，通过多个节点和边表示（连接节点的直线称为“边”）。

### 5.1.1 用计算图求解

问题1：太郎在超市买了2个100日元一个的苹果，消费税是10%，请计算支付金额。

计算图通过节点和箭头表示计算过程。将计算的中间结果写在箭头的上方，表示各个节点的计算结果从左向右传递。用计算图解问题1，求解过程如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682217736396-47e7d7b7-f852-4fc1-8243-c30143bc671e.png#averageHue=%23434343&clientId=ucd66bb0b-d041-4&from=paste&height=101&id=uda6ad531&name=image.png&originHeight=126&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=23977&status=done&style=none&taskId=u6a65c6a1-8d35-47b9-9aa2-bebb3bca2ae&title=&width=691.2)<br />虽然上图中把“× 2”“× 1.1”等作为一个运算整体，不过只用⚪表示乘法运算“×”也是可行的。如下图所示，可以将“2”和“1.1”分别作为变量“苹果的个数”和“消费税”标在⚪外面。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682217892520-4d1bca79-4e10-41de-82f2-8a0029f06531.png#averageHue=%23424242&clientId=ucd66bb0b-d041-4&from=paste&height=230&id=ubc729f19&name=image.png&originHeight=288&originWidth=863&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=34603&status=done&style=none&taskId=ue9340eab-2dea-447b-9e17-0479e2e8072&title=&width=690.4)

问题2：太郎在超市买了2个苹果、3个橘子。其中，苹果每个100日元，橘子每个150日元。消费税是10%，请计算支付金额。

用计算图解问题2，求解过程如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682217940376-62a35c3c-0561-4cdc-9e93-fa81bfd88b67.png#averageHue=%23424242&clientId=ucd66bb0b-d041-4&from=paste&height=295&id=u4f051ecc&name=image.png&originHeight=369&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=50824&status=done&style=none&taskId=u3ea9ce03-802b-46e4-ae39-74a432086e5&title=&width=688)<br />这个问题新增了加法节点“+”，用来合计苹果和句子的金额。

综上，用计算图解题需要按如下流程进行。

1. 构建计算图
2. 在计算图上，从左向右进行计算

这里的“从左向右进行计算”是一种正方向上的传播，简称为正向传播（forward propagation）。正向传播是从计算图出发点到结束点的传播。既然有正向传播这个名称，当然也可以考虑反向（从图上看的话，就是从右向左）的传播。实际上，这种传播称为反向传播（backward propagation）。

### 5.1.2 局部计算
> 计算图的特征是可以通过传递“局部计算”获得最终结果。“局部”这个词的意思是“与自己相关的某个小范围”。

局部计算是指，无论全局发生了什么，都能只根据与自己相关的信息输出接下来的结果。比如，在超市买了2个苹果和其他很多东西。此时，可以画出如下图所示的计算图。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682219151864-fa12a912-3157-47bb-8ec2-cf896d32a8ea.png#averageHue=%23424242&clientId=ucd66bb0b-d041-4&from=paste&height=325&id=u95eaec7f&name=image.png&originHeight=406&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=86793&status=done&style=none&taskId=u7b7af940-ec90-429a-b2d4-80d2da3fc8a&title=&width=691.2)<br />如上图所示，假设（经过复杂的计算）购买的其他很多东西总共花费4000日元。这里的重点是，各个节点处的计算都是局部计算。这意味着，例如苹果和其他很多东西的求和运算（4000 + 200 → 4200）并不关心4000这个数字是如何计算而来的，只要把两个数字相加就可以了。<br />换言之，各个节点处只需进行与自己有关的计算（在这个例子中是对输入的两个数字进行加法运算），不用考虑全局。

综上，计算图可以集中精力于局部计算。无论全局的计算有多么复杂，各个步骤所要做的就是对象节点的局部计算。虽然局部计算非常简单，但是通过传递它的计算结果，可以获得全局的复杂计算的结果。

### 5.1.3 为何用计算图解题
计算图的优点

1. 局部计算。无论全局是多么复杂的计算，都可以通过局部计算使各个节点致力于简单的计算，从而简化问题
2. 利用计算图可以将中间的计算结果全部保存起来
3. 可以通过反向传播高效计算导数

对于上面的问题1，假设我们想知道苹果价格的上涨会在多大程度上影响最终的支付金额，即求“支付金额关于苹果的价格的导数”。设苹果价格为x，支付金额为L，则相当于求![](https://cdn.nlark.com/yuque/__latex/e9aacec8b15fc1e9462a4aaf873e6494.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20x%7D&id=Egv8b)。这个导数的值表示当苹果的价格稍微上涨时，支付金额会增加多少。<br />“支付金额关于苹果的价格的导数”的值可以通过计算图的反向传播求出来。可以通过计算图的反向传播求导数，具体过程如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682220985618-1ec231fb-cb9a-415b-9737-cb8276896356.png#averageHue=%23424242&clientId=ucd66bb0b-d041-4&from=paste&height=228&id=u451ef698&name=image.png&originHeight=285&originWidth=862&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=37376&status=done&style=none&taskId=ub78b151b-275c-4fe0-9ecc-82fba1f86fc&title=&width=689.6)<br />反向传播使用与正方向相反的箭头（粗线）表示。反向传播传递“局部导数”，将导数的值写在箭头的下方。从这个结果中可知，“支付金额关于苹果的价格的导数”的值是2.2。这意味着，如果苹果的价格上涨1日元，最终的支付金额会增加2.2日元

计算中途求得的导数的结果（中间传递的导数）可以被共享，从而可以高效地计算多个导数。综上，计算图的优点是，可以通过正向传播和反向传播高效地计算各个变量的导数值。


















