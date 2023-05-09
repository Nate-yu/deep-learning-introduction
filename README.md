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
    - [5.2 链式法则](#52-链式法则)
        - [5.2.1 计算图的反向传播](#521-计算图的反向传播)
        - [5.2.2 什么是链式法则](#522-什么是链式法则)
        - [5.2.3 链式法则和计算图](#523-链式法则和计算图)
    - [5.3 反向传播](#53-反向传播)
        - [5.3.1 加法节点的反向传播](#531-加法节点的反向传播)
        - [5.3.2 乘法节点的反向传播](#532-乘法节点的反向传播)
        - [5.3.3 苹果的例子](#533-苹果的例子)
    - [5.4 简单层的实现](#54-简单层的实现)
        - [5.4.1 乘法层的实现](#541-乘法层的实现)
        - [5.4.2 加法层的实现](#542-加法层的实现)
    - [5.5 激活函数层的实现](#55-激活函数层的实现)
        - [5.5.1 ReLU层](#551-relu层)
        - [5.5.2 Sigmoid层](#552-sigmoid层)
    - [5.6 Affine/Softmax层的实现](#56-affinesoftmax层的实现)
        - [5.6.1 Affine层](#561-affine层)
        - [5.6.2 批版本的Affine层](#562-批版本的affine层)
        - [5.6.3 Softmax-with-Loss 层](#563-softmax-with-loss-层)
    - [5.7 误差反向传播法的实现](#57-误差反向传播法的实现)
        - [5.7.1 神经网络学习的全貌图](#571-神经网络学习的全貌图)
        - [5.7.2 对应误差反向传播的神经网络的实现](#572-对应误差反向传播的神经网络的实现)
        - [5.7.3 误差反向传播法的梯度确认](#573-误差反向传播法的梯度确认)
        - [5.7.4 使用误差反向传播法的学习](#574-使用误差反向传播法的学习)
- [6 与学习相关的技巧](#6-与学习相关的技巧)
    - [6.1 参数的更新](#61-参数的更新)
        - [6.1.1 SGD](#611-sgd)
        - [6.1.2 Momentum](#612-momentum)
        - [6.1.3 AdaGrad](#613-adagrad)
        - [6.1.4 Adam](#614-adam)
        - [6.1.5 更新方法的选择](#615-更新方法的选择)
        - [6.1.6 基于MNIST数据集的更新方法的比较](#616-基于mnist数据集的更新方法的比较)
    - [6.2 权重的初始值](#62-权重的初始值)
        - [6.2.1 可以将权重初始值设为0吗](#621-可以将权重初始值设为0吗)
        - [6.2.2 隐藏层的激活值的分布](#622-隐藏层的激活值的分布)
        - [6.2.3 ReLU的权重初始值](#623-relu的权重初始值)
        - [6.2.4 基于MNIST数据集的权重初始值的比较](#624-基于mnist数据集的权重初始值的比较)
    - [6.3 Batch Normalization](#63-batch-normalization)
        - [6.3.1 Batch Normalization 的算法](#631-batch-normalization-的算法)
        - [6.3.2 Batch Normalization的评估](#632-batch-normalization的评估)
    - [6.4 正则化](#64-正则化)
        - [6.4.1 过拟合](#641-过拟合)
        - [6.4.2 权值衰减](#642-权值衰减)
        - [6.4.3 Dropout](#643-dropout)
    - [6.5 超参数的验证](#65-超参数的验证)
        - [6.5.1 验证数据](#651-验证数据)
        - [6.5.2 超参数的最优化](#652-超参数的最优化)
        - [6.5.3 超参数最优化的实现](#653-超参数最优化的实现)
- [7 卷积神经网络](#7-卷积神经网络)
    - [7.1 整体结构](#71-整体结构)
    - [7.2 卷积层](#72-卷积层)
        - [7.2.1 全连接层存在的问题](#721-全连接层存在的问题)
        - [7.2.2 卷积运算](#722-卷积运算)
        - [7.2.3 填充](#723-填充)
        - [7.2.4 步幅](#724-步幅)
        - [7.2.5 3维数据的卷积运算](#725-3维数据的卷积运算)
        - [7.2.6 结合方块思考](#726-结合方块思考)
        - [7.2.7 批处理](#727-批处理)
    - [7.3 池化层](#73-池化层)
    - [7.4 卷积层和池化层的实现](#74-卷积层和池化层的实现)
        - [7.4.1 4维数组](#741-4维数组)
        - [7.4.2 基于im2col的展开](#742-基于im2col的展开)
        - [7.4.3 卷积层的实现](#743-卷积层的实现)
        - [7.4.4 池化层的实现](#744-池化层的实现)
    - [7.5 CNN的实现](#75-cnn的实现)
    - [7.6 CNN的可视化](#76-cnn的可视化)
        - [7.6.1 第1层权重的可视化](#761-第1层权重的可视化)
        - [7.6.2 基于分层结构的信息提取](#762-基于分层结构的信息提取)
    - [7.7 具有代表性的CNN](#77-具有代表性的cnn)
        - [7.7.1 LeNet](#771-lenet)
        - [7.7.2 AlexNet](#772-alexnet)

<!-- /TOC -->
# 1 Python知识预备
## 1.1 Python的安装
将使用的编程语言与库。

1. Python 3.x
2. NumPy（用于数值计算）
3. Matplotlib（将实验结果可视化）

Anaconda发行版。<br />[Anaconda 环境配置](https://www.yuque.com/abiny/wikclb/xp0imzu4wxouo4ag?view=doc_embed)

## 1.2 Python解释器
检查Python版本。打开终端，输入命令`python --version`，该命令会输出已安装的Python的版本信息<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681301360773-629be852-18e1-477b-844b-d42eba2dd227.png#averageHue=%23191919&clientId=u63205d68-842a-4&from=paste&height=132&id=u5e8dd190&originHeight=165&originWidth=535&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13408&status=done&style=none&taskId=uc9962723-718a-4fe5-861b-930e308b9de&title=&width=428)

输入命令`python`即可启动Python解释器<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681371465530-27d84266-ec22-4300-bf9e-d3c280031f56.png#averageHue=%23181818&clientId=u63116624-3e14-4&from=paste&height=168&id=ub0056b3e&originHeight=210&originWidth=1207&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20155&status=done&style=none&taskId=u7ec7fc0d-05d9-4c1e-ad31-ca573c7879d&title=&width=965.6)
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
从终端运行`man.py`<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681373379728-3ee6f331-4d85-4cb4-809c-f358c3cf9a5b.png#averageHue=%23191919&clientId=ubed99078-1978-4&from=paste&height=184&id=u0d72e8ff&originHeight=230&originWidth=1373&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=25821&status=done&style=none&taskId=uf6eccc05-0df9-4fb8-b313-d4a887f3a97&title=&width=1098.4)<br />这里我们定义了一个新类`Man`。上面的例子中，类Man生成了实例（对象）`m`。类Man的构造函数（初始化方法）会接收参数name，然后用这个参数初始化实例变量`self.name`。实例变量是存储在各个实例中的变量。Python 中可以像 self.name 这样，通过在 self 后面添加属性名来生成或访问实例变量。

## 1.4 NumPy
### 1.4.1 导入NumPy
`import numpy as np`<br />Python中使用`import`语句来导入库，这里的import numpy as np，直译的话就是“将numpy作为np导入”的意思。

### 1.4.2 NumPy数组
要生成NumPy数组，需要使用`np.array()`方法。`np.array()`接收Python列表作为参数，生成NumPy数组（numpy.ndarray）。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681374024600-e5872ac5-3042-4662-a2eb-36d5507395ce.png#averageHue=%23181818&clientId=ubed99078-1978-4&from=paste&height=134&id=u657cb9b2&originHeight=168&originWidth=412&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12039&status=done&style=none&taskId=uc6aa6201-c093-4347-bfa2-02f965d5969&title=&width=329.6)

下面是NumPy数组的算术运算的例子<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681374620471-07ec64ba-5417-4554-80fc-c0a47b282375.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=307&id=ua6b08fab&originHeight=384&originWidth=491&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27767&status=done&style=none&taskId=u5d3e7b50-cfa2-4a05-a2a7-5a01f09c7dc&title=&width=392.8)<br />这里需要注意的是，数组x和数组y的元素个数是相同的（两者均是元素个数为3的一维数组）。当x和y的元素个数相同时，可以对各个元素进行算术运算。如果元素个数不同，程序就会报错，所以元素个数保持一致非常重要。<br />NumPy数组与单一数值组合起来进行运算，需要在NumPy数组的各个元素和标量之间进行运算。这个功能也被称为“广播”。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681374771976-5fea14fa-504d-433f-9340-179615eb98e7.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=65&id=u0927b63c&originHeight=81&originWidth=306&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3835&status=done&style=none&taskId=u1066fda5-4862-491d-9dba-576cad53257&title=&width=244.8)

### 1.4.3 NumPy的N维数组
NumPy不仅可以生成一维数组（排成一列的数组），也可以生成多维数组。比如，可以生成如下的二维数组（矩阵）。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375015403-36f28ea5-0736-499d-b375-fb912683cb47.png#averageHue=%23151515&clientId=ubed99078-1978-4&from=paste&height=174&id=u6c5f5567&originHeight=217&originWidth=417&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12102&status=done&style=none&taskId=ueba25889-9dde-466b-88b4-a6f62ea1ffe&title=&width=333.6)<br />这里生成了一个2 × 2的矩阵A。另外，矩阵A的形状可以通过shape查看，矩阵元素的数据类型可以通过dtype查看。下面则是矩阵的算术运算<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375166295-b60edf62-f418-4c62-b0dc-a52163afba0f.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=158&id=ue8e8b997&originHeight=198&originWidth=390&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10618&status=done&style=none&taskId=u890432bc-1633-4231-bc54-c0c6c793f74&title=&width=312)<br />和数组的算术运算一样，矩阵的算术运算也可以在相同形状的矩阵间以对应元素的方式进行。并且，也可以通过标量（单一数值）对矩阵进行算术运算。这也是基于广播的功能。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375310039-18d8e7dc-2729-418c-90fa-fb832448714c.png#averageHue=%23181818&clientId=ubed99078-1978-4&from=paste&height=139&id=u3c252100&originHeight=174&originWidth=250&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=6559&status=done&style=none&taskId=ua283a115-47df-43c5-9da9-866681103ee&title=&width=200)

### 1.4.4 广播
> NumPy中，形状不同的数组之间也可以进行运算。之前的例子中，在2×2的矩阵A和标量10之间进行了乘法运算。

广播的实例<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375493160-ed6e6cf1-d24e-4a64-b9c5-ae0745edb32f.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=159&id=ud39f1815&originHeight=199&originWidth=350&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=9843&status=done&style=none&taskId=ub78ed560-4634-428b-bcbb-c74712c1851&title=&width=280)<br />在此运算中，一维数组B被“巧妙地”变成了和二位数组A相同的形状，然后再以对应元素的方式进行运算。综上，因为NumPy有广播功能，所以不同形状的数组之间也可以顺利地进行运算。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375562896-42e8b3fb-e43f-4ae8-b1af-0cd701b4c249.png#averageHue=%23414141&clientId=ubed99078-1978-4&from=paste&height=165&id=u21a9d845&originHeight=206&originWidth=1127&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24647&status=done&style=none&taskId=u7c400a42-15b9-47c6-89a9-f6d60d83ef9&title=&width=901.6)

### 1.4.5 访问元素
元素的索引从0开始。对各个元素的访问可按如下方式进行。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375748313-095cc8b4-682e-4b29-abc5-3826d00246e8.png#averageHue=%23131313&clientId=ubed99078-1978-4&from=paste&height=197&id=u8400083c&originHeight=246&originWidth=583&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13215&status=done&style=none&taskId=u4a0178db-cdbb-4a61-abe3-fcc0050f749&title=&width=466.4)<br />也可以使用`for`语句访问各个元素<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375823856-f20f51d2-a5b5-429d-afda-386525c7d60d.png#averageHue=%23131313&clientId=ubed99078-1978-4&from=paste&height=141&id=udc255b20&originHeight=176&originWidth=314&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=5267&status=done&style=none&taskId=u6fef30ca-7a7f-4600-835c-872dbf6d085&title=&width=251.2)

NumPy还可以使用数组访问各个元素。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375945066-25657b72-77f9-403d-ba25-29ae844905d8.png#averageHue=%23181818&clientId=ubed99078-1978-4&from=paste&height=120&id=u29252f3f&originHeight=150&originWidth=636&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14919&status=done&style=none&taskId=u7e197bb1-173c-4575-918f-934b2e76d62&title=&width=508.8)<br />运用这个标记法，可以获取满足一定条件的元素。例如，要从X中抽出大于15的元素，可以写成如下形式。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375993391-ae902c46-2e9d-49e7-b9a7-fc401218be99.png#averageHue=%23151515&clientId=ubed99078-1978-4&from=paste&height=100&id=uc256161c&originHeight=125&originWidth=609&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=7257&status=done&style=none&taskId=u14637be8-95c0-4ed4-b8c8-5f0b0bdd076&title=&width=487.2)<br />对NumPy数组使用不等号运算符等（上例中是X > 15）,结果会得到一个布尔型的数组。

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
这里使用NumPy的arange方法生成了[0, 0.1, 0.2, ..., 5.8, 5.9]的数据，将其设为x。对x的各个元素，应用NumPy的sin函数`np.sin()`，将x、y的数据传给`plt.plot()`方法，然后绘制图形。最后，通过`plt.show()`显示图形。运行上述代码后，就会显示如图所示的图形。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681376893281-47d41de9-437f-46e0-930a-130f5731a341.png#averageHue=%23fcfcfc&clientId=ubed99078-1978-4&from=paste&height=480&id=u9352eb6c&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=30810&status=done&style=none&taskId=u5d403bbc-1237-4703-acbb-19293443517&title=&width=640)

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
结果如图所示，图的标题、轴的标签名都被标出来了。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681376866898-0bf5e266-bbee-4cbe-bcea-9b5e57e75f4d.png#averageHue=%23fcfbfb&clientId=ubed99078-1978-4&from=paste&height=450&id=u8466cb98&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=46740&status=done&style=none&taskId=u07da9fda-3049-440e-b6ff-348a1cf2d2c&title=&width=600)

### 1.5.3 显示图像
pyplot中还提供了用于显示图像的方法imshow()。另外，可以使用matplotlib.image模块的imread()方法读入图像。代码实例如下。
```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('dataset/dog.jpg') # 读入图像
plt.imshow(img)

plt.show()
```
运行上述代码后，会显示如下图像<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681377913536-0e6c03eb-4d8f-4ce1-a4af-b1bd0c92fd1d.png#averageHue=%23dbc4b1&clientId=ubed99078-1978-4&from=paste&height=480&id=u0518a560&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=335581&status=done&style=none&taskId=u0de96ec0-6444-48b4-966c-e3c23bc4472&title=&width=640)<br />因为我的Python解释器运行在根目录下，且图片在dataset下，故图片路径为'dataset/dog.jpg'。

# 2 感知机
## 2.1 感知机的定义
> 感知机接收多个输入信号，输出一个信号。

和实际的电流不同的是，感知机的信号只有“流/不流”（1/0）两种取值。0对应“不传递信号”，1对应“传递信号”。如下图所示是一个接收两个输入信号的感知机的例子。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681440819573-d2627755-1569-47c2-9fe4-df8fea50fd57.png#averageHue=%23414141&clientId=u3972c055-bc56-4&from=paste&height=419&id=u6bf328c9&originHeight=524&originWidth=1493&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=35135&status=done&style=none&taskId=u9214393e-87a7-4342-ba07-825d3de6f98&title=&width=1194.4)<br />其中，x1、x2是输入信号，y 是输出信号，w1、w2是权重。图中的圆称为“神经元”或者“节点”。输入信号被送往神经元时，会被分别乘以固定的权重（w1x1、w2x2）。神经元会计算传送过来的信号的总和，只有当这个总和超过了某个界限值时，才会输出1。这也称为“神经元被激活” 。这里将这个界限值称为阈值，用符号θ表示。<br />上述即为感知机的运行原理，用数学公式表示即为如下：<br />![](https://cdn.nlark.com/yuque/__latex/b5414eef4b7284e5f0a28b65fa4257db.svg#card=math&code=y%3D%20%5Cbegin%7Bcases%7D0%20%5Cquad%20%28w_1x_1%20%2B%20w_2x_2%5Cle%5Ctheta%29%5C%5C%201%5Cquad%20%28w_1x_1%2Bw_2x_2%3E%5Ctheta%29%5Cend%7Bcases%7D&id=UYdgq)<br />感知机的多个输入信号都有各自固有的权重，这些权重发挥着控制各个信号的重要性的作用。即权重越大，对应该权重的信号的重要性就越高。

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

将 θ 换成 -b ，改写数学公式：<br />![](https://cdn.nlark.com/yuque/__latex/801b525f01e66b7bdc6abb731efed4d1.svg#card=math&code=y%3D%20%5Cbegin%7Bcases%7D0%20%5Cquad%20%28b%2Bw_1x_1%20%2B%20w_2x_2%5Cle0%29%5C%5C%201%5Cquad%20%28b%2Bw_1x_1%2Bw_2x_2%3E0%29%5Cend%7Bcases%7D&id=WUYmS),此处，b称为**偏置**，w1和w2称为**权重**。<br />感知机会计算输入信号和权重的乘积，然后加上偏置，如果这个值大于0则输出1，否则输出0。使用NumPy逐一确认结果。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681458245530-4e527aac-d5b0-4ef4-815b-04498c6f7ac9.png#averageHue=%231b1b1b&clientId=u21cf038b-070e-4&from=paste&height=230&id=u32b8a377&originHeight=287&originWidth=323&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14235&status=done&style=none&taskId=u4efd5878-b72b-4457-9aca-bfc566e3d57&title=&width=258.4)<br />在NumPy数组的乘法运算中，当两个数组的元素个数相同时，各个元素分别相乘，因此`w*x`的结果就是它们的各个元素分别相乘（[0, 1] * [0.5, 0.5] => [0, 0.5]）。之后，`np.sum(w*x)`再计算相乘后的各个元素的总和。最后再把偏置加到这个加权总和上。

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
输出：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681462775874-b8b92d7b-64d8-4d15-af91-679f8b225cde.png#averageHue=%23222c32&clientId=u21cf038b-070e-4&from=paste&height=81&id=u1cfbea84&originHeight=101&originWidth=240&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3928&status=done&style=none&taskId=u662b8860-04e1-438e-bad7-b0646028804&title=&width=192)

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
输出：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681462830103-0a9733a1-1836-4b63-b831-9a647f5df3ca.png#averageHue=%23212b30&clientId=u21cf038b-070e-4&from=paste&height=84&id=u835799bf&originHeight=105&originWidth=279&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=4034&status=done&style=none&taskId=u255bca7d-21ee-4bbb-9a26-ba2a1a2515a&title=&width=223.2)

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
输出：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681462714163-a8d450fa-dcb7-4379-8e6b-585ba74bc547.png#averageHue=%23222c31&clientId=u21cf038b-070e-4&from=paste&height=80&id=u597a2d12&originHeight=100&originWidth=239&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3962&status=done&style=none&taskId=ud777c716-6348-4356-a22f-f94c4e60754&title=&width=191.2)<br />与门、与非门、或门区别只在于权重参数的值，因此，在与非门和或门的实现中，仅设置权重和偏置的值这一点和与门的实现不同。

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
通过已有门电路的组合：<br />异或门的制作方法有很多，其中之一就是组合我们前面做好的与门、与非门、或门进行配置。与门，与非门，或门用如下图的符号表示<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681472867117-0a56e39e-d4f8-4f59-8e22-04e15c093b6a.png#averageHue=%23414141&clientId=u21cf038b-070e-4&from=paste&height=167&id=ue80250b0&originHeight=209&originWidth=906&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16198&status=done&style=none&taskId=ud6112002-d08f-405b-ae78-95dc944dd28&title=&width=724.8)<br />通过组合感知机（叠加层）就可以实现异或门。异或门可以通过如下所示配置来实现，这里，x1和x2表示输入信号，y表示输出信号，x1和x2是与非门和或门的输入，而与非门和或门的输出则是与门的输入。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681473004376-abd7ade6-d849-4a66-81fb-5bc2fc6adc8c.png#averageHue=%23414141&clientId=u21cf038b-070e-4&from=paste&height=193&id=u01e5cafb&originHeight=241&originWidth=701&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20588&status=done&style=none&taskId=u605abcec-265e-4d61-982f-8d0f5264084&title=&width=560.8)<br />验证正确性，把s1作为与非门的输出，把s2作为或门的输出，填入真值表

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
输出：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681473328701-19216ab6-3511-4f32-bc11-73fc03b34e10.png#averageHue=%23212a30&clientId=u21cf038b-070e-4&from=paste&height=81&id=ubd76c574&originHeight=101&originWidth=303&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=4237&status=done&style=none&taskId=uf5797a2c-dd42-4300-b131-327259eb8f6&title=&width=242.4)

用感知机的方表示方法（明确显示神经元）来表示这个异或门<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681473381307-ae7c0818-dcb4-4350-929e-db0bd17e57fe.png#averageHue=%23414141&clientId=u21cf038b-070e-4&from=paste&height=404&id=ueb31e869&originHeight=505&originWidth=883&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=49078&status=done&style=none&taskId=u29fe963c-dba1-463d-a9a2-3282706f889&title=&width=706.4)<br />异或门是一种多层结构的神经网络。这里，将最左边的一列称为第0层，中间的一列称为第1层，最右边的一列称为第2层。实际上，与门、或门是单层感知机，而异或门是2层感知机。叠加了多层的感知机也称为多层感知机（multi-layered perceptron）。<br />在如上图所示的2层感知机中，先在第0层和第1层的神经元之间进行信号的传送和接收，然后在第1层和第2层之间进行信号的传送和接收，具体如下所示。

1. 第0层的两个神经元接收输入信号，并将信号发送至第1层的神经元。
2. 第1层的神经元将信号发送至第2层的神经元，第2层的神经元输出y。

# 3 神经网络
> 神经网络的一个重要性质是它可以自动地从数据中学习到合适的权重参数。

## 3.1 从感知机到神经网络
如下图所示，把最左边的一列称为**输入层**，最右边的一列称为**输出层**，中间的一列称为**中间层**。中间层有时候也被称为隐藏层。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681561346694-117eab84-9dbd-4db3-8220-3d159886a29d.png#averageHue=%23424242&clientId=u664bfb51-2e30-4&from=paste&height=594&id=ue568955d&originHeight=742&originWidth=921&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=89516&status=done&style=none&taskId=u60bc8de4-886e-4cb5-8ac4-108055ab60e&title=&width=736.8)

简化感知机数学式：![](https://cdn.nlark.com/yuque/__latex/8fc00128696670950893256c60f16a21.svg#card=math&code=y%3Dh%28b%2Bw_1x_1%2Bw_2x_2%29&id=G7WZT)，我们用一个函数来表示这种分情况的动作（超过0则输出1，否则输出0）。<br />![](https://cdn.nlark.com/yuque/__latex/ce5f91ff43c8c12fcb79e91caf39a7ab.svg#card=math&code=h%28x%29%20%3D%20%5Cbegin%7Bcases%7D%200%20%5Cquad%20%28x%20%5Cle%200%29%20%5C%5C%201%20%5Cquad%20%28x%3E0%29%5Cend%7Bcases%7D&id=GoWvF)<br />输入信号的总和会被函数h(x)转换，转换后的值就是输出y。h（x）函数会将输入信号的总和转换为输出信号，这种函数一般称为激活函数（activation function）。其作用在于决定如何来激活输入信号的总和。<br />进一步来改进上式：<br />![](https://cdn.nlark.com/yuque/__latex/c0f2fc874d8df61947b105ef97705458.svg#card=math&code=%281%29%20%5Cquad%20a%20%3D%20b%20%2B%20w_1x_1%20%2B%20w_2x_2%20%0A&id=OLjcb)<br />![](https://cdn.nlark.com/yuque/__latex/ef7e54eae38bcb740b79e4a7f316c7f2.svg#card=math&code=%282%29%20%5Cquad%20y%20%3D%20h%28x%29%0A&id=zPld6)<br />首先，式（1）计算加权输入信号和偏置的总和，记为a。然后，式（2）用h()函数将a转换为输出y。下图为明确显示激活函数的计算过程。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681562541410-54919415-68b7-42d2-9f18-249e7b918385.png#averageHue=%23424242&clientId=u664bfb51-2e30-4&from=paste&height=307&id=u6f8e6ecf&originHeight=613&originWidth=689&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=43209&status=done&style=none&taskId=u07b64979-c9bc-4e91-bfb8-0c8817a31cd&title=&width=345)信号的加权总和为节点a，然后节点a被激活函数h()转换成节点y。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681562849199-b81a45ea-bca0-42c8-b842-b840280fb2f5.png#averageHue=%23414141&clientId=u664bfb51-2e30-4&from=paste&height=238&id=u0c9793c0&originHeight=298&originWidth=1316&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=38865&status=done&style=none&taskId=uc31b44f3-42b7-4582-8c50-cf0603cb52f&title=&width=1052.8)<br />左图是一般的神经元的图，右图是在神经元内部明确显示激活函数的计算过程的图（a表示输入信号的总和，h()表示激活函数，y表示输出）

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

对上面的代码进行解读：对NumPy数组进行不等号运算后，数组的各个元素都会进行不等号运算，生成一个布尔型数组，大于0的被转换为True，小于等于0的被转换为False，从而形成一个新的布尔型数组y。但我们想要的跃阶函数是会输出int型的0或1的函数，因此需要把数组y的元素类型从布尔型转换为int型。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681611582389-7bf9f2e8-bd77-4c28-8d3f-81fed8672778.png#averageHue=%2330343c&clientId=ufb91a362-4377-4&from=paste&height=217&id=uc36410ff&originHeight=271&originWidth=530&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=19501&status=done&style=none&taskId=u4af4b33c-87af-4ad6-a74d-03e5707ee3b&title=&width=424)<br />如上所示，可以用astype()方法转换NumPy数组的类型。astype()方法通过参数指定期望的类型，这个例子中是np.int64型。Python中将布尔型转换为int型后，True会转换为1，False会转换为0。

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
	step_function()以该NumPy数组为参数，对数组的各个元素执行阶跃函数运算，并以数组形式返回运算结果。对数组x、y进行绘图，结果如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681612000678-69066074-31e3-4ba1-a84b-e509b46c4726.png#averageHue=%23fcfcfc&clientId=ufb91a362-4377-4&from=paste&height=480&id=ud36ec270&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12443&status=done&style=none&taskId=u0479c1a1-3b0e-4544-a25a-5b38578db28&title=&width=640)其值呈阶梯式变化，所以称为阶跃函数。

### 3.2.3 sigmoid函数的实现与图像
用Python表示sigmoid函数如下
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
如果在这个sigmoid函数中输入一个NumPy数组，则结果如下所示<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681635280332-bd2a8352-4cd4-4a04-b509-ce2c717358c1.png#averageHue=%2331353d&clientId=ufb91a362-4377-4&from=paste&height=170&id=u332bef14&originHeight=212&originWidth=593&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17342&status=done&style=none&taskId=u04174489-7b40-407b-ae1e-91de5761b90&title=&width=474.4)<br />根据NumPy 的广播功能，如果在标量和NumPy数组之间进行运算，则标量会和NumPy数组的各个元素进行运算。

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
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681635664998-f0b2f0c8-f623-4c5c-96cb-3f1e3e99cd2e.png#averageHue=%23fcfcfc&clientId=ufb91a362-4377-4&from=paste&height=480&id=u4ceb15e4&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20800&status=done&style=none&taskId=ud6d74a35-2e41-48fe-9a16-74a08588d17&title=&width=640)

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
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681636235482-b59fc910-2e0c-4f38-a6a6-dacf95aaf11e.png#averageHue=%23fbfbfb&clientId=ufb91a362-4377-4&from=paste&height=480&id=u8ff07ac1&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24907&status=done&style=none&taskId=u41657456-2262-416d-a15d-34d59d8d628&title=&width=640)虚线为阶跃函数。

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

maximum函数会从输入的数值中选择较大的那个值进行输出。ReLU函数图像如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681636976693-e2e19e10-84ec-4dbd-aa3c-41873ae5d997.png#averageHue=%23fdfdfd&clientId=ufb91a362-4377-4&from=paste&height=480&id=u8a5fe94b&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14548&status=done&style=none&taskId=ucabee685-461f-466e-893d-95cacf93588&title=&width=640)

## 3.3 多维数组的运算
### 3.3.1 多维数组
用NumPy来生成多维数组。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681643307918-46f3a580-8bbc-437e-87d2-5ab3e4fc4ad1.png#averageHue=%232e323a&clientId=u59ebbd16-e1e7-4&from=paste&height=216&id=u54284b0f&originHeight=270&originWidth=470&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15105&status=done&style=none&taskId=u1f38e056-e47c-4e0b-a5cc-bef5bd3670b&title=&width=376)<br />数组的维数可以通过`np.dim()`函数获得，数组的形状可以通过实例变量`shape`获得。在上面的例子中，A是一维数组，由4个元素构成，`A.shape`的结果是一个元组。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681643517943-81f59bae-02fb-44c2-ad37-27ccbe406e3e.png#averageHue=%232d3139&clientId=u59ebbd16-e1e7-4&from=paste&height=202&id=uc7ffd61c&originHeight=253&originWidth=504&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12638&status=done&style=none&taskId=ud02b3a9b-e85d-4652-ac68-064ca238cb2&title=&width=403.2)<br />这里生成了一个 3 X 2 的数组B。其第一个维度有3个元素，第二个维度有2个元素。另外，第一个维度对应第0维，第二个维度对应第1维。

### 3.3.2 矩阵乘法
利用NumPy实现矩阵乘法。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681643820556-218862d9-e6a5-4447-8350-3920dc81d6f9.png#averageHue=%2331353d&clientId=u59ebbd16-e1e7-4&from=paste&height=196&id=ucb8029d9&originHeight=245&originWidth=440&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16972&status=done&style=none&taskId=u2a8c35e1-e137-408f-ac05-bf03705ed5a&title=&width=352)<br />这里的A、B均为2 X 2的矩阵，其乘积可以通过NumPy的`np.dot()`函数计算。`np.dot()`接收两个NumPy数组作为参数，并返回数组的乘积。<br />注意：在两个矩阵相乘时，矩阵A的第1维和矩阵B的第0维元素个数必须一致。另外，当A是二维矩阵，B是一维数组时，对应维度的元素个数要保持一致的原则依然成立。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681645792389-f62ce452-e9da-462b-829a-2ec930281e98.png#averageHue=%232f333b&clientId=u59ebbd16-e1e7-4&from=paste&height=176&id=udd9d4a41&originHeight=220&originWidth=558&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16623&status=done&style=none&taskId=u9995ea01-5894-4381-af63-f0cf4796f7c&title=&width=446.4)

### 3.3.3 神经网络的内积
以如下图所示的简单神经网络为对象（省略了偏置和激活函数，只有权重），使用NumPy矩阵来实现神经网络。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681645943756-99d872ff-d923-4331-bc0a-4403f74712f9.png#averageHue=%23414141&clientId=u59ebbd16-e1e7-4&from=paste&height=348&id=u74601007&originHeight=435&originWidth=1034&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=52363&status=done&style=none&taskId=u35db73ca-5808-4e4d-ab21-f505b6ffc8b&title=&width=827.2)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681646153811-f01fbcb5-51d7-423f-b9dd-88040812fcc5.png#averageHue=%232e323a&clientId=u59ebbd16-e1e7-4&from=paste&height=255&id=uabae394d&originHeight=319&originWidth=507&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20911&status=done&style=none&taskId=uc09c5e89-8704-4e5e-9a6e-ad504b6c0be&title=&width=405.6)<br />如上所示，使用np.dot可以一次性计算出Y的结果。

## 3.4 3层神经网络的实现
3层神经网络示意图<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681718849833-835a8ff6-5e78-4e74-b979-244ac9ec3a9e.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=349&id=ub68b167d&originHeight=436&originWidth=843&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=58696&status=done&style=none&taskId=ue9602726-24da-4a04-a051-0607c21680c&title=&width=674.4)<br />输入层有2个神经元，第1个隐藏层有3个神经元，第2个隐藏层有2个神经元，输出层有2个神经元

### 3.4.1 符号确认
权重符号如下<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681719977951-16438890-0340-4ff1-808d-51617a665b8b.png#averageHue=%23414141&clientId=u05e7a439-d16b-4&from=paste&height=307&id=ufa409226&originHeight=384&originWidth=830&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=50531&status=done&style=none&taskId=u1428a267-6e8d-48c9-bb10-9e9ae8a31a6&title=&width=664)<br />权重和隐藏层的神经元的右上角有一个“(1)”，表示权重和神经元的层号（即第1层的权重、第1层的神经元）。此外，权重的右下角有两个数字，它们是后一层的神经元和前一层的神经元的索引号。![](https://cdn.nlark.com/yuque/__latex/08c3265a2052ba6be0de989d959b7eff.svg#card=math&code=w_%7B12%7D%5E%7B%281%29%7D&id=spxxh)表示前一层的第2个神经元![](https://cdn.nlark.com/yuque/__latex/9929b4550bf80849e3bbd9bdace8be77.svg#card=math&code=x_2%0A&id=FIRSh)到后一层的第1个神经元![](https://cdn.nlark.com/yuque/__latex/8e3351610d813c64e18b3c901bafc333.svg#card=math&code=a_1%5E%7B%281%29%7D&id=nxfxC)的权重。权重右下角按照“后一层的索引号、前一层的索引号”的顺序排列。

### 3.4.2 各层间信号传递的实现
下面是输入层到第1层的第一个神经元的信号传递过程<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681720475570-d9853296-24ca-46af-bc82-72792b2d10f2.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=419&id=ue9118699&originHeight=524&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=90058&status=done&style=none&taskId=u30f68141-080c-407c-8406-fa898e751bb&title=&width=688)<br />上图中增加了表示偏置的神经元“1”。偏置的右下角的索引号只有一个，因为前一层的偏置神经元只有一个。<br />下面用数学式表示![](https://cdn.nlark.com/yuque/__latex/8e3351610d813c64e18b3c901bafc333.svg#card=math&code=a_1%5E%7B%281%29%7D&id=yagQ5)通过加权信号和偏置的和按如下方式进行计算：![](https://cdn.nlark.com/yuque/__latex/c77460825d40735a9a69fb0a277be610.svg#card=math&code=a_1%5E%7B%281%29%7D%20%3D%20w_%7B11%7D%5E%7B1%7Dx_1%20%2B%20w_%7B12%7D%5E%7B%281%29%7Dx_2%20%2B%20b_1%5E%7B%281%29%7D%0A&id=YXvL6)。此外，如果使用矩阵的乘法运算，则可以将第1层的加权表示成下面的式子：![](https://cdn.nlark.com/yuque/__latex/25e7c3556ab706b9587766023a7ee384.svg#card=math&code=%5Cbm%20%7BA%7D%5E%7B%281%29%7D%20%3D%20%5Cbm%7BXW%7D%5E%7B%281%29%7D%20%2B%20%5Cbm%7BB%7D%5E%7B%281%29%7D&id=Ir6RH)，其中各元素如下所示<br />![](https://cdn.nlark.com/yuque/__latex/3f100ca5bf0cb8a5bae7dadc67adcd1d.svg#card=math&code=%5Cbm%7BA%7D%5E%7B%281%29%7D%20%3D%20%5Cbegin%7Bpmatrix%7Da_1%5E%7B%281%29%7D%20%26%20a_2%5E%7B%281%29%7D%20%26%20a_3%5E%7B%281%29%7D%5Cend%7Bpmatrix%7D%EF%BC%8C%5Cbm%7BX%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%20x_1%20%26%20x_2%20%5Cend%7Bpmatrix%7D%EF%BC%8C%5Cbm%7BB%7D%5E%7B%281%29%7D%3D%5Cbegin%7Bpmatrix%7D%20b_1%5E%7B%281%29%7D%20%26%20b_2%5E%7B%281%29%7D%20%26%20b_3%5E%7B%281%29%7D%20%5Cend%7Bpmatrix%7D%EF%BC%8CW%5E%7B%281%29%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%20w_%7B11%7D%5E%7B%281%29%7D%20%26w_%7B21%7D%5E%7B%281%29%7D%20%26%20w_%7B31%7D%5E%7B%281%29%7D%20%5C%5C%20w_%7B12%7D%5E%7B%281%29%7D%20%26w_%7B22%7D%5E%7B%281%29%7D%20%26%20w_%7B32%7D%5E%7B%281%29%7D%5Cend%7Bpmatrix%7D&id=ogM9q)

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
W1为2 X 3的数组，X是元素个数为2的一维数组。第1层中激活函数的计算过程如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681722975139-248aaf98-4c04-4ddd-9ce1-5662001cbc99.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=496&id=u46daa2a3&originHeight=620&originWidth=856&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=128100&status=done&style=none&taskId=u19e7490d-cc79-4934-a5c8-4afb1c35bc3&title=&width=684.8)<br />如上图所示，隐藏层的加权和（加权信号和偏置的总和）用a表示，被激活函数转换后的信号用z表示。此外，图纸h()表示激活函数。这里使用sigmoid函数，用Python实现如下所示。
```python
Z1 = sigmoid(A1)

print(A1) # [0.3, 0.7, 1.1]
print(Z1) # [0.57444252, 0.66818777, 0.75026011]
```

下面来实现第1层到第2层的信号传递。图示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681723225327-ee1fd5bc-c0cc-450b-be54-39ee5527611c.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=464&id=udff0d667&originHeight=580&originWidth=851&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=132079&status=done&style=none&taskId=u20182c7a-da76-4713-b685-db4f5ba8666&title=&width=680.8)<br />除了第1层的输出（Z1）变成了第2层的输入这一点以外，这个实现和刚才的代码完全相同。
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

最后是第2层到输出层的信号传递。输出层的实现也和之前的实现基本相同。不过，最后的激活函数和之前的隐藏层有所不同。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681723450946-8a274b97-b915-41c9-9e43-ea328454b13c.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=433&id=uf7d73495&originHeight=541&originWidth=846&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=123080&status=done&style=none&taskId=ubff74a00-47a9-4d8a-8854-de57a5d174a&title=&width=676.8)
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

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681735934458-b2cd7956-824d-4541-9c3a-a88ee9095782.png#averageHue=%23414141&clientId=u05e7a439-d16b-4&from=paste&height=224&id=ud3f43a09&originHeight=280&originWidth=359&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18699&status=done&style=none&taskId=u3ff15381-92bf-44bb-a205-44776367db3&title=&width=287.2)

分类问题中使用的softmax函数可以用下面的数学式表示。<br />![](https://cdn.nlark.com/yuque/__latex/0134ce5b3fbc81c86adc88f5c68176b4.svg#card=math&code=y_k%20%3D%20%5Cfrac%7Be%5E%7Ba_k%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%7D%7D&id=rAlb0)<br />这个式子表示假设输出层共有n个神经元，计算第k个神经元的输出![](https://cdn.nlark.com/yuque/__latex/48e6989aee378b0671dcbc11187f8dd6.svg#card=math&code=y_k&id=LP6cI)。分子是输入信号![](https://cdn.nlark.com/yuque/__latex/eb1130b86c0023f2fc1466c5f4664eb9.svg#card=math&code=a_k&id=t4PXG)的指数函数，分母是所有输入信号的指数函数的和。softmax函数的图示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681736253756-5d6e7f44-0786-4110-8295-74d023641600.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=225&id=u1bb2d82b&originHeight=281&originWidth=295&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24835&status=done&style=none&taskId=u8af574ca-7cec-4c2f-8a56-8e5df98d5fb&title=&width=236)<br />softmax函数的输出通过箭头与所有的输入信号相连。从上面的数学式可以看出，输出层的各个神经元都受到所有输入信号的影响。

用Python解释器实现softmax函数如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681736653152-203f8efc-7a75-4824-9c75-2b02a03a99e0.png#averageHue=%23373b43&clientId=u05e7a439-d16b-4&from=paste&height=236&id=u841780ba&originHeight=295&originWidth=464&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27633&status=done&style=none&taskId=ub3278667-81bf-4c75-a14f-15eacb9af39&title=&width=371.2)<br />将其封装为`softmax()`函数。
```python
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
	return y
```

### 3.5.2 实现softmax函数时的注意事项
为了防止指数计算时的溢出，softmax函数的实现可以如下改进。<br />![](https://cdn.nlark.com/yuque/__latex/d2ef4ae5f3b96de651302b78138bfba3.svg#card=math&code=y_k%20%3D%20%5Cfrac%7Be%5E%7Ba_k%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%7D%7D%20%3D%20%5Cfrac%7BCe%5E%7Ba_k%7D%7D%7BC%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%7D%7D%20%3D%20%5Cfrac%7Be%5E%7Ba_k%2BlnC%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%2BlnC%7D%7D%20%3D%20%5Cfrac%7Be%5E%7Ba_k%2BC%27%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ene%5E%7Ba_i%2BC%27%7D%7D%28%E5%85%B6%E4%B8%AD%EF%BC%8CC%27%3DlnC%29&id=qNFH8)

这里的`C'`可以使用任何值，但是为了防止溢出，一般会使用输入信号中的最大值。具体实例如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681737161062-116991db-71ec-475a-a3dc-28c978f761db.png#averageHue=%23343840&clientId=u05e7a439-d16b-4&from=paste&height=214&id=u50e56a02&originHeight=267&originWidth=831&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=30728&status=done&style=none&taskId=u61af6bc7-eb73-434e-aab0-04cae6a74f1&title=&width=664.8)<br />如该例所示，通过减去输入信号中的最大值（上例中的c），我们发现原本为nan（not a number，不确定）的地方，现在被正确计算了。综上，softmax函数可以优化如下。
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
输出结果：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681737713902-efca57fb-a496-4530-bdd2-d62233921ba6.png#averageHue=%23253036&clientId=u05e7a439-d16b-4&from=paste&height=70&id=u06743c5d&originHeight=87&originWidth=862&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16499&status=done&style=none&taskId=uce62d9c4-c8d4-4f65-8139-02889311301&title=&width=689.6)<br />如上所示，softmax函数的输出是0.0到1.0之间的实数。并且，softmax函数的输出值的总和是1。输出总和为1是softmax函数的一个重要性质。正因为有了这个性质，我们才可以把softmax函数的输出解释为“概率”。

一般而言，神经网络只把输出值最大的神经元所对应的类别作为识别结果。并且，即便使用softmax函数，输出值最大的神经元的位置也不会变。因此，神经网络在进行分类时，输出层的softmax函数可以省略。

### 3.5.4 输出层的神经元数量
> 输出层的神经元数量需要根据待解决的问题来决定。对于分类问题，输出层的神经元数量一般设定为类别的数量。

比如，对于某个输入图像，预测是图中的数字0到9中的哪一个的问题（10类别分类问题），可以像下图这样，将输出层的神经元设定为10个。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681737983461-d7d91e85-9f59-4e2f-8e87-c93efc5d1208.png#averageHue=%23414141&clientId=u05e7a439-d16b-4&from=paste&height=339&id=u0c7f65eb&originHeight=424&originWidth=777&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=39871&status=done&style=none&taskId=u21ab90e8-8975-462a-a855-eea2fcb9bac&title=&width=621.6)<br />如上图所示，输出层的神经元从上往下依次对应数字0, 1, . . ., 9。此外，图中输出层的神经元的值用不同的灰度表示。这个例子中神经元y2颜色最深，输出的值最大。这表明这个神经网络预测的是y2对应的类别，也就是“2”。

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
执行上述代码之后，训练图像的第一张就会显示出来，如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681783390429-ef5bff05-c964-4200-812e-8b907f0b5b8a.png#averageHue=%23efefef&clientId=ude4bc88e-ee5e-4&from=paste&height=340&id=u4870c339&originHeight=425&originWidth=744&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=22445&status=done&style=none&taskId=u84f709ae-bf4b-4de8-889a-2af235f1105&title=&width=595.2)<br />需要注意的是，flatten=True时读入的图像是以一列（一维）NumPy数组的形式保存的。因此，显示图像时，需要把它变为原来的28像素 × 28像素的形状。可以通过reshape()方法的参数指定期望的形状，更改NumPy数组的形状。此外，还需要把保存为NumPy数组的图像数据转换为PIL用的数据对象，这个转换处理由`Image.fromarray()`来完成。而`np.uint8(img)`是将图像数据类型转换为8位无符号整数类型的函数，可以减少内存占用并提高计算速度。

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
考虑打包输入多张图像的情形。比如，我们想用`predict()`函数一次性打包处理100张图像。为此，可以把x的形状改为100 × 784，将100张图像打包作为输入数据。如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681786186999-c5cea2fe-954d-4b11-bdc4-7eab687cdf5d.png#averageHue=%23414141&clientId=ude4bc88e-ee5e-4&from=paste&height=128&id=u2e989359&originHeight=160&originWidth=848&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15741&status=done&style=none&taskId=uea038880-8f44-4242-922d-1565b36e08d&title=&width=678.4)<br />输入数据的形状为 100 × 784，输出数据的形状为100 × 10。这表示输入的100张图像的结果被一次性输出了。这种打包式的输入数据称为批（batch）。进行基于批处理的代码实现如下。
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

举例如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681786802220-92325511-2ac0-49d3-a200-da05ffce1848.png#averageHue=%232e323a&clientId=ude4bc88e-ee5e-4&from=paste&height=131&id=u9e849c30&originHeight=164&originWidth=1052&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15812&status=done&style=none&taskId=u88c3d31d-1485-4dc2-b55d-4bf934873ba&title=&width=841.6)

4. 最后，我们比较一下以批为单位进行分类的结果和实际的答案。为此，需要在NumPy数组之间使用比较运算符（==）生成由True/False构成的布尔型数组，并计算True的个数。

# 4 神经网络的学习
> 这里所说的“学习”是指从训练数据中自动获取最优权重参数的过程


## 4.1 从数据中学习
> 所谓“从数据中学习”，是指可以由数据自动决定权重参数的值。


下面是两种针对机器学习任务的方法，以及神经网络（深度学习）的方法。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682038464552-6d6b5be2-8cbb-4ba5-97b7-a4286e375cd2.png#averageHue=%23444444&clientId=u3bfb9bac-dc34-4&from=paste&height=315&id=u0600d5f5&originHeight=394&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=35557&status=done&style=none&taskId=uad14a795-3822-4859-83d8-116ed585ebf&title=&width=691.2)<br />如图所示，神经网络直接学习图像本身。在第二个方法，即利用特征量和机器学习的方法中，特征量仍是由人工设计的，而在神经网络中，连图像中包含的重要特征量也都是由机器来学习的。

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

因此，交叉熵误差实际上只计算正确解标签的输出的自然对数。也就是说，交叉熵误差的值是由正确解标签所对应的输出结果决定的。<br />自然对数图像如下如所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682041946272-e35499c8-1b16-49ba-8e80-f69d8ee6dee4.png#averageHue=%23fcfcfc&clientId=u3bfb9bac-dc34-4&from=paste&height=480&id=u0668b88f&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18983&status=done&style=none&taskId=u902c5ef3-bd76-4a6a-9a67-8eab603bbdf&title=&width=640)<br />x等于1时，y为0；随着x向0靠近，y逐渐变小。因此正确解标签对应的输出越大，CEE的值越接近0；当输出为1时，CEE为0。下面用Python实现交叉熵误差。
```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```
这里，参数 y 和 t 是NumPy数组。函数内部在计算`np.log`时，加上了一个微小值delta，这是因为当出现`np.log(0)`时，会变为负无限大的`-inf`，这样就会导致后续计算无法进行。作为保护性对策，添加一个微小值可以防止负无限大的发生。

使用`cross_entropy_error(y, t)`进行一些简单的计算<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682044355295-44486e4f-b0b6-484a-ab7d-67a86408e6f7.png#averageHue=%2332363e&clientId=u3bfb9bac-dc34-4&from=paste&height=344&id=udfe6465d&originHeight=430&originWidth=841&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=42271&status=done&style=none&taskId=u189e2927-b556-4d1d-8654-ad2ef4d17ff&title=&width=672.8)<br />第一个例子中，正确解标签对应的输出为0.6，此时交叉熵误差大约为0.51。第二个例子中，正确解标签对应的输出为0.1的低值，此时的交叉熵误差大约为2.3。

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

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682063452294-da2838cd-7404-4d42-a435-537d670a78e5.png#averageHue=%23fcfcfc&clientId=uc6be4a61-1f40-4&from=paste&height=480&id=ue1f2c96f&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=21810&status=done&style=none&taskId=u51282eea-4a98-48e9-b3ff-ac7fa9d3851&title=&width=640)

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

x=5处的切线如下<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682065463885-63e55d0e-89df-498a-a85d-eab0b74e63d2.png#averageHue=%23fcfcfb&clientId=uc6be4a61-1f40-4&from=paste&height=480&id=uaf7b73c5&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27844&status=done&style=none&taskId=u9a56b707-5ac6-4de7-a399-a0e4e9141c4&title=&width=640)

### 4.3.3 偏导数
如下一个函数，有两个变量。<br />![](https://cdn.nlark.com/yuque/__latex/d40c8e25200796ee3e5d872374b72528.svg#card=math&code=f%28x_0%2Cx_1%29%20%3D%20x_0%5E2%2Bx_1%5E2&id=WWued)<br />用Python实现如下。
```python
def function_2(x):
    return x[0] ** 2 + x[1] ** 2
    # 或者 return np.sum(x**2)
```

函数图像如下图所示<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682065831378-bc7e84ed-0f91-4893-ab6b-ee4098fed154.png#averageHue=%23fafafa&clientId=uc6be4a61-1f40-4&from=paste&height=480&id=u37a8a44e&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=153055&status=done&style=none&taskId=ub0b458ab-4322-4eaf-a150-ecc43b5ff09&title=&width=640)

求偏导数<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682066162084-9c39604d-c0cf-4a2c-a2dc-8993901259ef.png#averageHue=%23414141&clientId=uc6be4a61-1f40-4&from=paste&height=366&id=ub8b83a22&originHeight=457&originWidth=612&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=66679&status=done&style=none&taskId=u82c2e35a-3609-4c31-bdd0-ca11a0accc6&title=&width=489.6)<br />偏导数和单变量的导数一样，都是求某个地方的斜率。偏导数需要将多个变量中的某一个变量定为目标变量，并将其他变量固定为某个值。

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

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682068570155-649f67d5-8ec3-42ac-b59b-9b7476e022b3.png#averageHue=%23f6f6f6&clientId=uc6be4a61-1f40-4&from=paste&height=480&id=uce95b296&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=88822&status=done&style=none&taskId=u42af958e-b390-48c7-ba5a-b8f2d7ea542&title=&width=640)

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

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682077455478-f1190231-f757-4e5f-9e3a-eb6e8e8ffa7d.png#averageHue=%23fcfcfc&clientId=u4287f989-f779-4&from=paste&height=480&id=u62cd216a&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16645&status=done&style=none&taskId=uabc156d8-43a3-4c54-8ee0-d7c6800a936&title=&width=640)

注意：实验结果表明，学习率过大的话，会发散成一个很大的值；反过来，学习率过小的话，基本上没怎么更新就结束了。也就是说，设定合适的学习率是一个很重要的问题。

### 4.4.2 神经网络的梯度
> 神经网络的学习也要求梯度。这里所说的梯度是指损失函数关于权重参数的梯度。

比如，有一个只有一个形状为 2 × 3 的权重 W 的神经网络，损失函数用 L 表示。此时，梯度可以用![](https://cdn.nlark.com/yuque/__latex/36e5597d13be0b891d1dbb4a7b2f5f0d.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D&id=RhceS)表示。数学式如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682077691375-f35f6560-958b-4f4f-b60a-c611e1107d44.png#averageHue=%23414141&clientId=u4287f989-f779-4&from=paste&height=149&id=ua5c3a116&originHeight=186&originWidth=392&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=21322&status=done&style=none&taskId=u27eb31d3-56e9-4e52-a495-27b6975e0ea&title=&width=313.6)<br />![](https://cdn.nlark.com/yuque/__latex/36e5597d13be0b891d1dbb4a7b2f5f0d.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D&id=AgjU9)的元素由各个元素关于 W 的偏导数构成。下面以一个简单的神经网络为例，来实现求梯度的代码。
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
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682078741991-d3dd415d-a24e-42dd-8da1-0d3f4ac96744.png#averageHue=%2328333a&clientId=u4287f989-f779-4&from=paste&height=98&id=u4ac4ff76&originHeight=123&originWidth=442&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17176&status=done&style=none&taskId=u2657d7cb-891c-4e8c-b377-c1b45671d49&title=&width=353.6)

接下来求梯度。使用`numerical_gradient(f, x)`求梯度（这里定义的函数f(W)的参数W是一个伪参数。因为`numerical_gradient(f, x)`会在内部执行f(x),为了与之兼容而定义了f(W)）。
```python
f = lambda w: net.loss(x,t)
dW = numerical_gradient(f, net.W)
print(dW)
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682078877356-d8b13634-9af2-42f6-ab0a-b5b29de9ccce.png#averageHue=%232c3d45&clientId=u4287f989-f779-4&from=paste&height=50&id=ud3d91515&originHeight=62&originWidth=406&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10340&status=done&style=none&taskId=uf8755009-d9a1-4fcc-b28f-6b72b55065c&title=&width=324.8)<br />`numerical_gradient(f, x)`的参数`f`是函数，`x`是传给函数`f`的参数。因此，这里参数`x`取`net.W`，并定义一个计算损失函数的新函数`f`，然后把这个新定义的函数传递给`numerical_gradient(f, x)`。`numerical_gradient(f, net.W)`的结果是`dW`，一个形状为2 × 3的二维数组。

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
随着学习的进行，损失函数的值在不断减小。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682083105031-124190af-3611-4b6b-8f7a-89ec18bbcee7.png#averageHue=%23434343&clientId=u4287f989-f779-4&from=paste&height=346&id=u6160cfeb&originHeight=433&originWidth=879&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=57597&status=done&style=none&taskId=ub4571a2a-1753-4f8e-925c-d3b056d4788&title=&width=703.2)

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
每经过一个epoch，就对所有的训练数据和测试数据计算识别精度，并记录结果。之所以要计算每一个epoch的识别精度，是因为如果在for语句的循环中一直计算识别精度，会花费太多时间。因此，我们才会每经过一个epoch就记录一次训练数据的识别精度。<br />结果用图像表示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682083553496-ba1b6605-325d-4819-89f9-1f36915481dc.png#averageHue=%23fcfbfb&clientId=u4287f989-f779-4&from=paste&height=480&id=u8b6e3a59&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=28136&status=done&style=none&taskId=u26f71e52-7496-4569-ae28-b6239baa652&title=&width=640)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682083612377-489a093e-5464-4e9f-8b45-caec4869fe5f.png#averageHue=%23272f35&clientId=u4287f989-f779-4&from=paste&height=317&id=uee5c5dc3&originHeight=396&originWidth=836&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=69859&status=done&style=none&taskId=u3389a6ac-a3b7-4b85-8456-f61fe8d9ba9&title=&width=669)<br />实线表示训练数据的识别精度，虚线表示测试数据的识别精度。如图所示，随着epoch的前进（学习的进行），我们发现使用训练数据和测试数据评价的识别精度都提高了，并且，这两个识别精度基本上没有差异（两条线基本重叠在一起）。因此，可以说这次的学习中没有发生过拟合的现象。

# 5 误差反向传播法
## 5.1 计算图
> 计算图将计算过程用图形表示出来。这里说的图形是数据结构图，通过多个节点和边表示（连接节点的直线称为“边”）。

### 5.1.1 用计算图求解

问题1：太郎在超市买了2个100日元一个的苹果，消费税是10%，请计算支付金额。

计算图通过节点和箭头表示计算过程。将计算的中间结果写在箭头的上方，表示各个节点的计算结果从左向右传递。用计算图解问题1，求解过程如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682217736396-47e7d7b7-f852-4fc1-8243-c30143bc671e.png#averageHue=%23434343&clientId=ucd66bb0b-d041-4&from=paste&height=101&id=uda6ad531&originHeight=126&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=23977&status=done&style=none&taskId=u6a65c6a1-8d35-47b9-9aa2-bebb3bca2ae&title=&width=691.2)<br />虽然上图中把“× 2”“× 1.1”等作为一个运算整体，不过只用⚪表示乘法运算“×”也是可行的。如下图所示，可以将“2”和“1.1”分别作为变量“苹果的个数”和“消费税”标在⚪外面。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682217892520-4d1bca79-4e10-41de-82f2-8a0029f06531.png#averageHue=%23424242&clientId=ucd66bb0b-d041-4&from=paste&height=230&id=ubc729f19&originHeight=288&originWidth=863&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=34603&status=done&style=none&taskId=ue9340eab-2dea-447b-9e17-0479e2e8072&title=&width=690.4)

问题2：太郎在超市买了2个苹果、3个橘子。其中，苹果每个100日元，橘子每个150日元。消费税是10%，请计算支付金额。

用计算图解问题2，求解过程如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682217940376-62a35c3c-0561-4cdc-9e93-fa81bfd88b67.png#averageHue=%23424242&clientId=ucd66bb0b-d041-4&from=paste&height=295&id=u4f051ecc&originHeight=369&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=50824&status=done&style=none&taskId=u3ea9ce03-802b-46e4-ae39-74a432086e5&title=&width=688)<br />这个问题新增了加法节点“+”，用来合计苹果和句子的金额。

综上，用计算图解题需要按如下流程进行。

1. 构建计算图
2. 在计算图上，从左向右进行计算

这里的“从左向右进行计算”是一种正方向上的传播，简称为正向传播（forward propagation）。正向传播是从计算图出发点到结束点的传播。既然有正向传播这个名称，当然也可以考虑反向（从图上看的话，就是从右向左）的传播。实际上，这种传播称为反向传播（backward propagation）。

### 5.1.2 局部计算
> 计算图的特征是可以通过传递“局部计算”获得最终结果。“局部”这个词的意思是“与自己相关的某个小范围”。

局部计算是指，无论全局发生了什么，都能只根据与自己相关的信息输出接下来的结果。比如，在超市买了2个苹果和其他很多东西。此时，可以画出如下图所示的计算图。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682219151864-fa12a912-3157-47bb-8ec2-cf896d32a8ea.png#averageHue=%23424242&clientId=ucd66bb0b-d041-4&from=paste&height=325&id=u95eaec7f&originHeight=406&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=86793&status=done&style=none&taskId=u7b7af940-ec90-429a-b2d4-80d2da3fc8a&title=&width=691.2)<br />如上图所示，假设（经过复杂的计算）购买的其他很多东西总共花费4000日元。这里的重点是，各个节点处的计算都是局部计算。这意味着，例如苹果和其他很多东西的求和运算（4000 + 200 → 4200）并不关心4000这个数字是如何计算而来的，只要把两个数字相加就可以了。<br />换言之，各个节点处只需进行与自己有关的计算（在这个例子中是对输入的两个数字进行加法运算），不用考虑全局。

综上，计算图可以集中精力于局部计算。无论全局的计算有多么复杂，各个步骤所要做的就是对象节点的局部计算。虽然局部计算非常简单，但是通过传递它的计算结果，可以获得全局的复杂计算的结果。

### 5.1.3 为何用计算图解题
计算图的优点

1. 局部计算。无论全局是多么复杂的计算，都可以通过局部计算使各个节点致力于简单的计算，从而简化问题
2. 利用计算图可以将中间的计算结果全部保存起来
3. 可以通过反向传播高效计算导数

对于上面的问题1，假设我们想知道苹果价格的上涨会在多大程度上影响最终的支付金额，即求“支付金额关于苹果的价格的导数”。设苹果价格为x，支付金额为L，则相当于求![](https://cdn.nlark.com/yuque/__latex/e9aacec8b15fc1e9462a4aaf873e6494.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20x%7D&id=Egv8b)。这个导数的值表示当苹果的价格稍微上涨时，支付金额会增加多少。<br />“支付金额关于苹果的价格的导数”的值可以通过计算图的反向传播求出来。可以通过计算图的反向传播求导数，具体过程如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682220985618-1ec231fb-cb9a-415b-9737-cb8276896356.png#averageHue=%23424242&clientId=ucd66bb0b-d041-4&from=paste&height=228&id=u451ef698&originHeight=285&originWidth=862&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=37376&status=done&style=none&taskId=ub78b151b-275c-4fe0-9ecc-82fba1f86fc&title=&width=689.6)<br />反向传播使用与正方向相反的箭头（粗线）表示。反向传播传递“局部导数”，将导数的值写在箭头的下方。从这个结果中可知，“支付金额关于苹果的价格的导数”的值是2.2。这意味着，如果苹果的价格上涨1日元，最终的支付金额会增加2.2日元

计算中途求得的导数的结果（中间传递的导数）可以被共享，从而可以高效地计算多个导数。综上，计算图的优点是，可以通过正向传播和反向传播高效地计算各个变量的导数值。

## 5.2 链式法则
### 5.2.1 计算图的反向传播
下面是一个使用计算图的反向传播的例子：假设存在 y = f(x) 的计算，这个计算的反向传播如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682236797803-9fc3a4a6-3dc2-456e-bf1a-d63e9b791a68.png#averageHue=%23414141&clientId=ue9c1380f-c053-4&from=paste&height=133&id=u1fd01d03&originHeight=166&originWidth=863&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=8526&status=done&style=none&taskId=u0831afa3-3ec6-4e47-af32-3f1a1483601&title=&width=690.4)<br />如上图所示，反向传播的顺序是，将信号 E 乘以结点的局部导数（![](https://cdn.nlark.com/yuque/__latex/c98403508ed4f41213e396ea80b4523e.svg#card=math&code=%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D&id=SChaU)），然后将结果传递给下一个节点。比如，假设![](https://cdn.nlark.com/yuque/__latex/59dc7bf9020e8e091817820c213cf062.svg#card=math&code=y%20%3D%20f%28x%29%3Dx%5E2&id=axdRv)，则局部导数为![](https://cdn.nlark.com/yuque/__latex/204c8e7879fad098a56a9c248d896df9.svg#card=math&code=%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D%3D2x&id=foJ7t)。把这个局部导数乘以上游传过来的值（E），然后传递给前面的节点。

### 5.2.2 什么是链式法则
> 链式法则是关于复合函数的导数的性质，定义如下：如果某个函数由复合函数表示，则该复合函数的导数可以用构成复合函数的各个函数的导数的乘积表示。

以上就是链式法则的原理。假设有：![](https://cdn.nlark.com/yuque/__latex/e7a382bdb9fbb2165730bffc56c67c3e.svg#card=math&code=z%3Dt%5E2%2Ct%3Dx%2By&id=AoPNR)。链式法则用数学式表示如下。<br />![](https://cdn.nlark.com/yuque/__latex/9430b70c194e9b9d9be4fd80be88e7a9.svg#card=math&code=%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20x%7D%3D%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20t%7D%5Cfrac%7B%5Cpartial%20t%7D%7B%5Cpartial%20x%7D&id=aOfjn)<br />因此，局部导数![](https://cdn.nlark.com/yuque/__latex/2f3ad9cfed6335b25cd3f803aba6977a.svg#card=math&code=%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20x%7D%3D%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20t%7D%5Cfrac%7B%5Cpartial%20t%7D%7B%5Cpartial%20x%7D%3D2t%C2%B71%3D2%28x%2By%29&id=oZbiU)

### 5.2.3 链式法则和计算图
用“**2”节点表示平方运算，上面链式法则的计算用计算图表示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682237598895-926bbf13-ee9b-468c-8208-ff52c5f2c4ad.png#averageHue=%23414141&clientId=ue9c1380f-c053-4&from=paste&height=305&id=u16bf74e4&originHeight=381&originWidth=865&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=32081&status=done&style=none&taskId=u1e466180-071e-4737-994a-9e99664f6f4&title=&width=692)<br />如图所示，计算图的反向传播从右到左传播信号。反向传播的计算顺序是先将节点的输入信号乘以节点的局部导数，然后再传递给下一个节点。比如，反向传播时，“**2”节点的输入是![](https://cdn.nlark.com/yuque/__latex/867edb8efe0a9857e57f8c8bde159c02.svg#card=math&code=%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20z%7D&id=vSCLK)，将其乘以局部导数![](https://cdn.nlark.com/yuque/__latex/0cbfc06e476908382ec2b78ef775dbb8.svg#card=math&code=%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20t%7D&id=dECF1)，然后传递给下一个节点。<br />把![](https://cdn.nlark.com/yuque/__latex/e7a382bdb9fbb2165730bffc56c67c3e.svg#card=math&code=z%3Dt%5E2%2Ct%3Dx%2By&id=TJ1q2)的结果带入上图，如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682238130593-1c16d6c0-0960-4bca-a699-dbff308b6058.png#averageHue=%23414141&clientId=ue9c1380f-c053-4&from=paste&height=302&id=ub5567d67&originHeight=378&originWidth=861&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=26666&status=done&style=none&taskId=uaeddbdf6-6580-47a1-a842-67437229c0e&title=&width=688.8)

## 5.3 反向传播
### 5.3.1 加法节点的反向传播
以![](https://cdn.nlark.com/yuque/__latex/2b6fedd11fb5c16ef005b49edf1796c4.svg#card=math&code=z%3Dx%2By&id=nzLG2)为对象，观察其反向传播。且导数为：![](https://cdn.nlark.com/yuque/__latex/dcb5bf359ebb515a566e9ea13e29a384.svg#card=math&code=%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20x%7D%3D%201%2C%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y%7D%3D%201&id=OeX4k)。因此，用计算图表示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682298781855-1720ebaf-dccf-4da1-ac05-df0a7c719e0a.png#averageHue=%23414141&clientId=uc472159d-9ad5-4&from=paste&height=297&id=ufc7c8b33&originHeight=371&originWidth=863&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27623&status=done&style=none&taskId=ucc5a61fc-5869-4b96-b4e1-36dd6a13590&title=&width=690.4)<br />上图中，反向传播从上游传过来的导数乘以1，然后传向下游。也就是说，因为加法节点的反向传播只乘以1，所以输入的值回原封不动地流向下一个节点。

下面是一个加法的反向传播的具体例子。假设有“10+5=15”这一计算，反向传播时，从上游传来值1.3，则计算图表示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682298995603-c3b790cf-d0f6-4968-b15c-3af02bf87f74.png#averageHue=%23414141&clientId=uc472159d-9ad5-4&from=paste&height=296&id=uf4ca1eeb&originHeight=370&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24355&status=done&style=none&taskId=u1691fcec-648c-4457-a4da-39df5397429&title=&width=691.2)<br />因为加法节点的反向传播只是将输入信号输出到下一个节点，所以上图反向传播将 1.3 向下一个节点传递。

### 5.3.2 乘法节点的反向传播
以![](https://cdn.nlark.com/yuque/__latex/6db7773c03e177fa2ea87edb558d64a2.svg#card=math&code=z%3Dxy&id=ycALH)为对象，观察其反向传播。且导数为：![](https://cdn.nlark.com/yuque/__latex/ac7637282c1a6559b16b7002002ef6b4.svg#card=math&code=%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20x%7D%3D%20y%2C%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y%7D%3D%20x&id=LfbyP)。因此，用计算图表示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682299244081-3c1217e7-51da-42b6-acbf-81f8c2332fab.png#averageHue=%23414141&clientId=uc472159d-9ad5-4&from=paste&height=296&id=u6bfeb42b&originHeight=370&originWidth=861&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=28943&status=done&style=none&taskId=u4d366c6c-de11-4a32-9054-5c56b4747bc&title=&width=688.8)<br />乘法的反向传播会将上游的值乘以正向传播时输入信号的“翻转值”后传递给下游。

下面是一个乘法的反向传播的具体例子。假设有“10×5=15”这一计算，反向传播时，从上游传来值1.3，则计算图表示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682299380995-c296305b-1bf4-4c00-80e3-bb6b28b6536c.png#averageHue=%23414141&clientId=uc472159d-9ad5-4&from=paste&height=298&id=u662fa36f&originHeight=372&originWidth=865&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=26739&status=done&style=none&taskId=u53095dbb-a1e6-4410-9d23-1bdbd65c9e9&title=&width=692)<br />因为乘法的反向传播会乘以输入信号的翻转值，所以各自可按1.3 × 5 = 6.5、1.3 × 10 = 13计算。另外，加法的反向传播只是将上游的值传给下游，并不需要正向传播的输入信号。但是，乘法的反向传播需要正向传播时的输入信号值。因此，实现乘法节点的反向传播时，要保存正向传播的输入信号。

### 5.3.3 苹果的例子

1. 支付金额关于苹果价格的导数
2. 支付金额关于苹果个数的导数
3. 支付金额关于消费税的导数

计算图的反向传播如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682299726103-0feb8e5c-667b-4259-b635-511a574564e8.png#averageHue=%23424242&clientId=uc472159d-9ad5-4&from=paste&height=250&id=u120e1948&originHeight=313&originWidth=866&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=40451&status=done&style=none&taskId=ude1aa208-96c2-4cca-a75a-114fa1f465f&title=&width=692.8)

正如前面所说，乘法节点的反向传播会将输入信号翻转后传给下游。从上图结果可知，苹果的价格的导数是2.2，苹果个数的导数是110，消费税的导数是200。这可以可以解释为，如果消费税和苹果的价格增加相同的值，则消费税将对最终价格产生200倍大小的影响，苹果的价格将产生2.2倍大小的影响。

## 5.4 简单层的实现
### 5.4.1 乘法层的实现
> 层的实现中有两个共通的方法（接口）forward()和backward()。forward()对应正向传播，backward()对应反向传播。

实现乘法层如下。
```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None


    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy
```
`__init__()`中会初始化实例变量x和y，它们用于保存正向传播时的输入值。`forward()`接收x和y两个参数，将它们相乘后输出。`backward()`将从上游传来的导数（dout）乘以正向传播的翻转值，然后传给下游。

下面使用`MulLayer`实现前面购买苹果的例子。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682301489234-c52e11cf-45fa-4469-a8d4-9854fc7b3590.png#averageHue=%23424242&clientId=uc472159d-9ad5-4&from=paste&height=268&id=u70026d66&originHeight=335&originWidth=865&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=45766&status=done&style=none&taskId=udd2494ad-9aa4-4a17-be68-438291942e6&title=&width=692)<br />使用此乘法层，实现上图的正向传播。
```python
from layer_naive import *

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple,apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price) # 220
```

此外，关于各个变量的导数，可由`backward()`求出。
```python
# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax) # 2.2 110 200
```
这里调用`backward()`的顺序与调用`forward()`的顺序相反。此外，要注意`backward()`的参数中需要输入“关于正向传播时的输出变量的导数”。比如，mul_apple_layer乘法层在正向传播时会输出apple_price，在反向传播时，则会将apple_price的导数dapple_price设为参数。

### 5.4.2 加法层的实现
实现加法节点的加法层。
```python
class AddLayer:
    def __init__(slef):
        pass

    def forward(self,x,y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
```
加法层不需要特意进行初始化，所以`__init__()`中什么也不运行。加法层的`forward()`接收x和y两个参数，将它们相加后输出。`backward()`将上游传来的导数（dout）原封不动地传递给下游。

使用加法层和乘法层，实现下图购买两个苹果和三个橘子的例子。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682304720493-8e58ff95-021b-4d58-976d-22b1a796661a.png#averageHue=%23424242&clientId=uc472159d-9ad5-4&from=paste&height=321&id=u0f12ed06&originHeight=401&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=66016&status=done&style=none&taskId=ua832a0dc-c909-49a9-9097-3463fbe8bae&title=&width=691.2)
```python
from layer_naive import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price) # 715
print(dapple_num,dapple, dorange,dorange_num,dtax) # 110 2.2 3.3 165 650
```

1. 首先，生成必要的层，以合适的顺序调用正向传播的forward()方法。
2. 然后，用与正向传播相反的顺序调用反向传播的backward()方法，就可以求出想要的导数。

## 5.5 激活函数层的实现
把构成神经网络的层实现为一个类。先来实现激活函数的ReLU层和Sigmoid层。

### 5.5.1 ReLU层
激活函数ReLU（Rectified Linear Unit）如下所示。<br />![](https://cdn.nlark.com/yuque/__latex/9d0f1a216c00bc2a32b1be48afb0ac37.svg#card=math&code=y%20%3D%20%5Cbegin%7Bcases%7D%20x%20%5Cquad%20%28x%20%3E%200%29%20%5C%5C%200%20%5Cquad%20%28x%20%5Cle%200%29%20%5Cend%7Bcases%7D&id=S6fvp)<br />通过上式，可以求出y关于x的导数如下所示。<br />![](https://cdn.nlark.com/yuque/__latex/0d840b23d3cdbe13fe01cfa3abd4db61.svg#card=math&code=%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D%20%3D%20%5Cbegin%7Bcases%7D%201%20%5Cquad%20%28x%20%3E%200%29%20%5C%5C%200%20%5Cquad%20%28x%20%5Cle%200%29%20%5Cend%7Bcases%7D&id=QHj9S)<br />在上式中，如果正向传播时的输入x大于0，则反向传播会将上游的值原封不动地传给下游。反过来，如果正向传播时的x小于等于0，则反向传播中传给下游的信号将停在此处。用计算图表示如下所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682337722393-fa7ed26c-c10e-4b61-afe3-b82f150e50fb.png#averageHue=%23414141&clientId=ud5e17f85-f041-4&from=paste&height=186&id=ua487415d&originHeight=232&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=21209&status=done&style=none&taskId=u4bb7abd5-ee6e-49ff-b623-0a7f88a6351&title=&width=691.2)<br />下面用Python来实现ReLU层。在神经网络的层的实现中，一般假定forward()和backward()的参数是NumPy数组。
```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
```
Relu类有实例变量mask。这个变量mask是由True/False构成的NumPy数组，它会把正向传播时的输入x的元素中小于等于0的地方保存为True，其他地方（大于0的元素）保存为False。<br />如下所示，mask变量保存了由True/False构成的NumPy数组。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682338558080-9d0d8733-9ff9-404f-97eb-89df8c8c6179.png#averageHue=%232f333b&clientId=ud5e17f85-f041-4&from=paste&height=202&id=uf8ea3e88&originHeight=253&originWidth=590&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17696&status=done&style=none&taskId=u0a4c32ee-f6ad-411b-9126-7f6fad20871&title=&width=472)<br />如果正向传播时的输入值小于等于0，则反向传播的值为0。因此，反向传播中会使用正向传播时保存的mask，将从上游传来的dout的mask中的元素为True的地方设为0。
> ReLU层的作用就像电路中的开关一样。正向传播时，有电流通过的话，就将开关设为ON；没有电流通过的话，就将开关设为OFF。反向传播时，开关为ON的话，电流会直接通过；开关为OFF的话，则不会有电流通过。


### 5.5.2 Sigmoid层
接下来实现sigmoid函数，函数式如下。<br />![](https://cdn.nlark.com/yuque/__latex/1cf6880d0b81950256848e4a1e6598cb.svg#card=math&code=y%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D&id=h868C)<br />用计算图表示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682338766094-e2d0fa39-ccbb-4482-8c6f-db3a79b4863f.png#averageHue=%23414141&clientId=ud5e17f85-f041-4&from=paste&height=186&id=u5c92dc47&originHeight=233&originWidth=837&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=19375&status=done&style=none&taskId=u4c4cba0f-4000-43dd-985e-dffc8d7a4f4&title=&width=669.6)<br />上图中，除了“×”和“+”节点外，还出现了新的“exp”和“/”节点。“exp”节点会进行![](https://cdn.nlark.com/yuque/__latex/27ffe57a1279a99ac0a6e3f972209ac7.svg#card=math&code=y%3De%5Ex&id=gjg0E)的计算，“/”节点会进行的![](https://cdn.nlark.com/yuque/__latex/6c314e80ccfb22fe6ba873bc757298b4.svg#card=math&code=y%3D%20%5Cfrac1x&id=hdfAD)计算。

如计算图所示，sigmoid函数式的计算由局部计算的传播构成。下面来进行计算图的反向传播。

1. “/”节点表示![](https://cdn.nlark.com/yuque/__latex/6c314e80ccfb22fe6ba873bc757298b4.svg#card=math&code=y%3D%20%5Cfrac1x&id=ur5pw)，其导数可以解析性地表示为：![](https://cdn.nlark.com/yuque/__latex/be208dc62c18c03976c5a22d7ea58f25.svg#card=math&code=%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D%20%3D%20-%5Cfrac%7B1%7D%7Bx%5E2%7D%20%3D%20-y%5E2&id=rJ5Bq)。反向传播时，会将上游的值乘以![](https://cdn.nlark.com/yuque/__latex/141163ecd267ecd58bc39155dee3314b.svg#card=math&code=-y%5E2%0A&id=rOokX)（正向传播的输出的平方乘以−1后的值）后，再传给下游。计算图如下所示。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682339461211-0fa7e039-c3b8-492b-a310-b9b08b5d32c7.png#averageHue=%23424242&clientId=ud5e17f85-f041-4&from=paste&height=151&id=u9477a15e&originHeight=189&originWidth=805&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=23286&status=done&style=none&taskId=u95dfa3ce-fc3a-4a34-8ddf-42b754edb42&title=&width=644)

2. “+”节点将上游的值原封不动地传给下游，计算图如下所示。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682339504598-377fe734-6d43-48ab-bcba-690c95ca0ff0.png#averageHue=%23424242&clientId=ud5e17f85-f041-4&from=paste&height=158&id=ucbbe688f&originHeight=198&originWidth=804&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=26215&status=done&style=none&taskId=u6675f10b-0d43-4b48-9d23-a771a2b45d1&title=&width=643)

3. “exp”节点表示![](https://cdn.nlark.com/yuque/__latex/27ffe57a1279a99ac0a6e3f972209ac7.svg#card=math&code=y%3De%5Ex&id=Bh11m)，其导数为![](https://cdn.nlark.com/yuque/__latex/20615c046540f52e46778e46245037f9.svg#card=math&code=%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D%20%3D%20e%5Ex%0A&id=NiA2Y)，计算图中，上游的值乘以正向传播时的输出（这个例子中是exp(−x)）后，再传给下游。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682339731256-01a7dc74-7c28-4f31-b63e-7219dddc7ae0.png#averageHue=%23424242&clientId=ud5e17f85-f041-4&from=paste&height=146&id=u94f41dfc&originHeight=183&originWidth=815&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27924&status=done&style=none&taskId=u4f6d2ff2-dc9e-4ffe-82a5-9b893a003c5&title=&width=652)

4. “×”节点将正向传播时的值翻转后做乘法运算。因此，这里要乘以−1。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682339766106-b4856185-bb1a-49f0-925b-67fd1479f807.png#averageHue=%23414141&clientId=ud5e17f85-f041-4&from=paste&height=158&id=ua4b72cff&originHeight=198&originWidth=839&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=31337&status=done&style=none&taskId=ufcfbdfb6-4fe0-472d-9772-5531be83ce5&title=&width=671.2)

由上图结果及上述内容可知，Sigmoid层的反向传播的输出为![](https://cdn.nlark.com/yuque/__latex/96dc19e52fb6a51ecbd8135607f6cb45.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20y%7Dy%5E2e%5E%7B-x%7D&id=hX142)，这个值会传播给下游的节点。这里需要注意的是，![](https://cdn.nlark.com/yuque/__latex/96dc19e52fb6a51ecbd8135607f6cb45.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20y%7Dy%5E2e%5E%7B-x%7D&id=BUfU1)这个值只根据正向传播时的输入x和输出y就可以算出来。故上面的计算图可以画成如下图所示的集约化“sigmoid”节点。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682339960237-caf33d95-bdda-47fb-bc59-a1b266c306f7.png#averageHue=%23414141&clientId=ud5e17f85-f041-4&from=paste&height=138&id=u1f1adf52&originHeight=173&originWidth=838&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14014&status=done&style=none&taskId=u16316542-11b4-4a68-954f-cff25196a9f&title=&width=670.4)

另外，![](https://cdn.nlark.com/yuque/__latex/96dc19e52fb6a51ecbd8135607f6cb45.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20y%7Dy%5E2e%5E%7B-x%7D&id=tHy2p)可以进一步整理如下。<br />![](https://cdn.nlark.com/yuque/__latex/b050f44b85dfc82326f7ad700b949048.svg#card=math&code=%5Cbegin%7Bequation%2A%7D%20%25%E5%8A%A0%2A%E8%A1%A8%E7%A4%BA%E4%B8%8D%E5%AF%B9%E5%85%AC%E5%BC%8F%E7%BC%96%E5%8F%B7%0A%09%5Cbegin%7Bsplit%7D%0A%09%09%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20y%7Dy%5E2e%5E%7B-x%7D%0A%09%09%26%20%3D%20%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20y%7D%20%5Cfrac%7B1%7D%7B%281%2Be%5E%7B-x%7D%29%5E2%7De%5E%7B-x%7D%5C%5C%0A%09%09%26%20%3D%20%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20y%7D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D%5Cfrac%7Be%5E%7B-x%7D%7D%7B1%2Be%5E%7B-x%7D%7D%5C%5C%0A%09%09%26%20%3D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20y%7Dy%281-y%29%0A%09%5Cend%7Bsplit%7D%0A%5Cend%7Bequation%2A%7D%20&id=vWWue)<br />故下图为Sigmoid层的反向传播，只根据正向传播的输出就能计算出来。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682340392354-fc9b76d0-ea94-4cf5-ae0a-4a0cd18727f6.png#averageHue=%23414141&clientId=ud5e17f85-f041-4&from=paste&height=138&id=u33c631ab&originHeight=173&originWidth=837&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12963&status=done&style=none&taskId=u890094ff-7fd6-41e5-9c85-6679b073dd5&title=&width=669.6)

现在用Python实现Sigmoid层。
```python
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self,x):
        out = sigmoid(x)
        self.out = out
        return out
    
    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
```
	这个实现中，正向传播时将输出保存在了实例变量out中，然后，反向传播时，使用该变量out进行计算。

## 5.6 Affine/Softmax层的实现
### 5.6.1 Affine层
> 神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为“仿射变换”。因此，这里将进行仿射变换的处理实现为“Affine层”。

现在将这里进行的求矩阵的乘积与偏置的和的运算用计算图表示出来。将乘积运算用“dot”节点表示的话，则`np.dot(X, W) + B`的运算可用下图所示的计算图表示出来。另外，在各个变量的上方标记了它们的形状（比如，计算图上显示了X的形状为(2,)，X·W的形状为(3,)等）。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682388119453-1f5c061c-f326-48bb-b8c6-8b00e057af67.png#averageHue=%23414141&clientId=ueefafb51-da6a-4&from=paste&height=302&id=u2159ed74&originHeight=378&originWidth=865&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=26007&status=done&style=none&taskId=u03024b04-48c5-48e0-9efa-c77a3ef5a6a&title=&width=692)<br />上图是比较简单的计算图，不过要注意X、W、B是矩阵（多维数组）。之前我们见到的计算图中各个节点间流动的是标量，而这个例子中各个节点间传播的是矩阵。

下面来考虑上面这个计算图的反向传播。以矩阵为对象的反向传播，按矩阵的各个元素进行计算时，步骤和以标量为对象的计算图相同。实际写一下的话，可以得到下式。<br />![](https://cdn.nlark.com/yuque/__latex/d10b21fb572bdff2a07c348c102f2ea6.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20X%7D%20%3D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20Y%7D%C2%B7W%5ET%20%5C%5C%0A%20%5C%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%20%3D%20X%5ET%C2%B7%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20Y%7D&id=AUJNy)<br />根据上式写出计算图的反向传播。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682388616114-0167e161-f262-4efc-a54d-4284600715bc.png#averageHue=%23414141&clientId=ueefafb51-da6a-4&from=paste&height=386&id=u98233bc4&originHeight=482&originWidth=862&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=62187&status=done&style=none&taskId=u11758d87-231e-4163-86c8-771291a6c16&title=&width=689.6)<br />注意计算图中各个变量的形状。![](https://cdn.nlark.com/yuque/__latex/94e79ad0c1aabeafef9e2fc4af6adf66.svg#card=math&code=X&id=c6UHm)和![](https://cdn.nlark.com/yuque/__latex/9f0e97e32d9617b725589f3aada0fa54.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20X%7D&id=BqMaD)形状相同，![](https://cdn.nlark.com/yuque/__latex/a36915ecf0b5605493f5aeaf1480a9ac.svg#card=math&code=W&id=K76rh)和![](https://cdn.nlark.com/yuque/__latex/36e5597d13be0b891d1dbb4a7b2f5f0d.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D&id=GAXNO)形状相同。

### 5.6.2 批版本的Affine层
前面介绍的Affine层的输入X是以单个数据为对象的。现在我们考虑N个数据一起进行正向传播的情况，也就是批版本的Affine层。下面是批版本的Affine层的计算图。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682389275505-be09b6cf-65da-4a77-9c26-5b468bbfd67c.png#averageHue=%23424242&clientId=ueefafb51-da6a-4&from=paste&height=414&id=uc98b5281&originHeight=517&originWidth=865&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=76503&status=done&style=none&taskId=uf9060104-e48d-4b8b-9145-1f9f877b49f&title=&width=692)<br />与刚刚不同的是，现在输入X的形状是(N, 2)。之后就和前面一样，在计算图上进行单纯的矩阵计算。反向传播时，如果注意矩阵的形状，就可以和前面一样推导出![](https://cdn.nlark.com/yuque/__latex/9f0e97e32d9617b725589f3aada0fa54.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20X%7D&id=mZU8I)和![](https://cdn.nlark.com/yuque/__latex/36e5597d13be0b891d1dbb4a7b2f5f0d.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D&id=CNTy0)。

正向传播时，偏置会被加到每一个数据（第1个、第2个……）上。因此，反向传播时，各个数据的反向传播的值需要汇总为偏置的元素。代码表示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682389506230-c7bc3999-4eb8-4184-8a9e-61348c006450.png#averageHue=%2330343c&clientId=ueefafb51-da6a-4&from=paste&height=178&id=u20c2ae7e&originHeight=222&originWidth=526&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16556&status=done&style=none&taskId=u282cb4ed-7472-4ad3-ae85-1e325325b6a&title=&width=420.8)<br />这个例子中，假定数据有2个（N = 2）。偏置的反向传播会对这2个数据的导数按元素进行求和。因此，这里使用了np.sum()对第0轴（以数据为单位的轴，axis=0）方向上的元素进行求和。

综上所述，Affine的实现如下所示。
```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None

        # 权重和偏执参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0],-1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape) # 还原输入数据的形状（对应张量）
        return dx
```

### 5.6.3 Softmax-with-Loss 层
下面来实现Softmax层。考虑到这里也包含作为损失函数的交叉熵误差（cross entropy error），所以称为“Softmax-with-Loss层”。Softmax-with-Loss层（Softmax函数和交叉熵误差）的计算图简化版如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682390663211-c9b02d65-332c-4a9e-8056-75bbb5c6dc16.png#averageHue=%23424242&clientId=ueefafb51-da6a-4&from=paste&height=446&id=u2a9907d2&originHeight=558&originWidth=865&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27640&status=done&style=none&taskId=ud0b17473-ccf5-4c9b-8d6b-68278407df4&title=&width=692)<br />上图中，softmax 函数记为 Softmax 层，交叉熵误差记为Cross Entropy Error层。这里假设要进行3类分类，从前面的层接收3个输入（得分）。Softmax 层将输入（a1, a2, a3）正规化，输出（y1, y2, y3）。Cross Entropy Error层接收Softmax的输出（y1, y2, y3）和教师标签（t1, t2, t3），从这些数据中输出损失L。

注意反向传播的结果，（y1− t1, y2− t2, y3− t3）是Softmax层的输出和教师标签的差分。神经网络的反向传播会把这个差分表示的误差传递给前面的层，这是神经网络学习中的重要性质。
> 神经网络学习的目的就是通过调整权重参数，使神经网络的输出（Softmax的输出）接近教师标签。

这里考虑一个具体的例子，比如思考教师标签是（0, 1, 0），Softmax层的输出是(0.3, 0.2, 0.5)的情形。因为正确解标签处的概率是0.2（20%），这个时候的神经网络未能进行正确的识别。此时，Softmax层的反向传播传递的是(0.3, −0.8, 0.5)这样一个大的误差。<br />下面来进行Softmax-with-Loss层的实现。
```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None


    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size),self.t] -= 1
            dx = dx / batch_size


        return dx
```
请注意反向传播时，将要传播的值除以批的大小（batch_size）后，传递给前面的层的是单个数据的误差。

## 5.7 误差反向传播法的实现
### 5.7.1 神经网络学习的全貌图
神经网络学习的步骤如下所示。

1. mini-batch：从训练数据中随机选择一部分数据
2. 计算梯度：计算损失函数关于各个权重参数的梯度
3. 更新参数：将权重参数沿梯度方向进行微小的更新
4. 重复上述3步

### 5.7.2 对应误差反向传播的神经网络的实现
把2层神经网络实现为`TwoLayerNet`。下面是这个类的实例变量和方法的表。

| 实例变量 | 说明 |
| --- | --- |
| params | 保存神经网络的参数的字典型变量。<br />params['W1']是第1层的权重，params['b1']是第1层的偏置。<br />params['W2']是第2层的权重，params['b2']是第2层的偏置 |
| layers | 保存神经网络的层的有序字典型变量。<br />以layers['Affine1']、layers['ReLu1']、layers['Affine2']的形式，通过有序字典保存各个层 |
| lastLayer | 神经网络的最后一层。<br />本例中为SoftmaxWithLoss层 |


| 方法 | 说明 |
| --- | --- |
| __init__(self, input_size, hidden_size, output_size, weight_init_std) | 进行初始化。<br />参数从头开始依次是输入层的神经元数、隐藏层的神经元数、输出层的神经元数、初始化权重时的高斯分布的规模 |
| predict(self, x) | 进行识别（推理）。<br />参数x是图像数据 |
| loss(self, x, t) | 计算损失函数的值。<br />参数X是图像数据、t是正确解标签 |
| accuracy(self, x, t) | 计算识别精度 |
| numerical_gradient(self, x, t) | 通过数值微分计算关于权重参数的梯度 |
| gradient(self, x, t) | 通过误差反向传播法计算关于权重参数的梯度 |


下面则是`TwoLayerNet`的代码实现
```python
import sys
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction") # 为了导入父目录的文件而进行的设定
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:


    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size) # 生成指定形状的随机数数组
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)


        return x
    
    # x: 输入数据，t: 监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # x: 输入数据, t: 监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)


        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])


        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)


        # backward
        dout = 1
        dout = self.lastLayer.back(dout)


        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)


        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db


        return grads
```
	OrderedDict是有序字典，“有序”是指它可以记住向字典里添加元素的顺序。因此，神经网络的正向传播只需按照添加元素的顺序调用各层的forward()方法就可以完成处理，而反向传播只需要按照相反的顺序调用各层即可。因为Affine层和ReLU层的内部会正确处理正向传播和反向传播，所以这里要做的事情仅仅是以正确的顺序连接各层，再按顺序（或者逆序）调用各层。

### 5.7.3 误差反向传播法的梯度确认
数值微分通常情况下的作用：经常会比较数值微分的结果和误差反向传播法的结果，以确认误差反向传播法的实现是否正确。
> 确认数值微分求出的梯度结果和误差反向传播法求出的结果是否一致（严格地讲，是非常相近）的操作称为梯度确认（gradient check）


梯度确认的代码实现如下所示。
```python
import sys
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction") # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


# 读入数据
(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


x_batch = x_train[:3]
t_batch = t_train[:3]


grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)


# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ": " + str(diff))
```
和以前一样，读入MNIST数据集。然后，使用训练数据的一部分，确认数值微分求出的梯度和误差反向传播法求出的梯度的误差。这里误差的计算方法是求各个权重参数中对应元素的差的绝对值，并计算其平均值。运行上面的代码后，会输出如下结果。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682431825977-0d2765c2-5be3-4503-a3b8-a7b0ae68e940.png#averageHue=%23263137&clientId=ub03a5fe0-7642-4&from=paste&height=84&id=u8ccb31fb&originHeight=105&originWidth=619&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18543&status=done&style=none&taskId=uf5c52a3e-cb5d-40a1-9520-b4609a8391d&title=&width=495.2)<br />从这个结果可以看出，通过数值微分和误差反向传播法求出的梯度的差非常小。比如，第1层的偏置的误差是3.7e-10。这样一来，我们就知道了通过误差反向传播法求出的梯度是正确的，误差反向传播法的实现没有错误。

### 5.7.4 使用误差反向传播法的学习
我们来看一下使用了误差反向传播法的神经网络的学习的实现。和之前的实现相比，不同之处仅在于通过误差反向传播法求梯度这一点。实现代码如下所示。
```python
import sys
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction") # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


# 读入数据
(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []


iter_per_epoch = max(train_size / batch_size, 1)


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]


    # 通过误差反向传播法求梯度
    grad = network.gradient(x_batch, t_batch)


    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]


    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)


    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
```
输出如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682432540364-f16857ef-de82-4ec0-bd57-9a46384cdfa2.png#averageHue=%23262e34&clientId=ub03a5fe0-7642-4&from=paste&height=290&id=u20964b01&originHeight=362&originWidth=622&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=37989&status=done&style=none&taskId=u492bb29a-7a45-4525-b4db-35e8e8b4ace&title=&width=497.6)

# 6 与学习相关的技巧
## 6.1 参数的更新

1. **最优化**（optimization）：神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题，解决这个问题的过程称为
2. **随机梯度下降法**（stochastic gradient descent）：使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数，简称SGD。

### 6.1.1 SGD
用数学式可以将SGD写成如下式子。<br />![](https://cdn.nlark.com/yuque/__latex/2e6aec2a64dd04664477fcc19c235c33.svg#card=math&code=W%20%5Cleftarrow%20W-%5Ceta%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D&id=KigMI)<br />上面把需要更新的权重参数记为W，把损失函数关于W的梯度记为![](https://cdn.nlark.com/yuque/__latex/671705d77aa781580f4be5015e46735a.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%0A&id=p716x)。![](https://cdn.nlark.com/yuque/__latex/96b0657b0bd8ddf567b3f27d8ad467e6.svg#card=math&code=%5Ceta%0A&id=Q6Kit)表示学习率，实际上会取 0.01 或 0.001 这些事先决定好的值。式子中的![](https://cdn.nlark.com/yuque/__latex/1c4b6f6d50a08c763be1abeca063a01f.svg#card=math&code=%5Cleftarrow&id=YwUvw)表示用右边的值更新左边的值。SGD是朝着梯度方向只前进一定距离的简单方法。<br />下面将SGD实现为一个Python类。
```python
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr


    def update(self, params, grads):
        for key in params.key():
            params[key] -= self.lr * grads[key]
```
这里，进行初始化时的参数lr表示learning rate（学习率）。这个学习率会保存为实例变量。此外，代码段中还定义了`update(params, grads)`方法，这个方法在SGD中会被反复调用。参数params和grads（与之前的神经网络的实现一样）是字典型变量，按params['W1']、grads['W1']的形式，分别保存了权重参数和它们的梯度。

考虑这个函数的最小值问题：![](https://cdn.nlark.com/yuque/__latex/e35c8aabe2007e96db6bf421d268465b.svg#card=math&code=f%28x%2Cy%29%3D%5Cfrac%7B1%7D%7B20%7Dx%5E2%2By%5E2&id=AV2ub)<br />如下图所示，上式表示的函数是向 x 轴方向延伸的“碗”状函数。实际上，其的等高线呈向x轴方向延伸的椭圆状。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682473469104-b0eec39e-3b51-4997-9b73-a50d0efd6362.png#averageHue=%234a4a4a&clientId=u7e2c328a-3440-4&from=paste&height=308&id=u6acbab83&originHeight=385&originWidth=878&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=210900&status=done&style=none&taskId=u31d37ffc-849d-4cb8-8a0e-80eb5205dd5&title=&width=702.4)

下面是函数的梯度。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682473564028-85a9e5f2-7f48-4eef-9305-73817d18ab7a.png#averageHue=%23414141&clientId=u7e2c328a-3440-4&from=paste&height=457&id=u29474502&originHeight=571&originWidth=858&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=60674&status=done&style=none&taskId=ud5bafb0a-884c-498e-8f3e-c68b47dfef4&title=&width=686.4)<br />这个梯度的特征是，y轴方向上大，x轴方向上小。换句话说，就是y轴方向的坡度大，而x轴方向的坡度小。

SGD的缺点是，如果函数的形状非均向（anisotropic），比如呈延伸状，搜索的路径就会非常低效。SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。

### 6.1.2 Momentum
用数学式表示Momentum方法，如下所示。<br />![](https://cdn.nlark.com/yuque/__latex/18134f44d8fa40a783b3b517876d839b.svg#card=math&code=v%5Cleftarrow%5Calpha%20v-%5Ceta%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%5C%5C%0A%5C%20%5C%5C%0AW%5Cleftarrow%20W%2Bv&id=PCiro)<br />和前面的SGD一样，W表示要更新的权重参数，![](https://cdn.nlark.com/yuque/__latex/36e5597d13be0b891d1dbb4a7b2f5f0d.svg#card=math&code=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D&id=BrAnO)表示损失函数关于W的梯度，![](https://cdn.nlark.com/yuque/__latex/7483c6745bb07f292eba02b3a9b55c26.svg#card=math&code=%5Ceta&id=BnZpR)表示学习率。这里出现了一个变量![](https://cdn.nlark.com/yuque/__latex/a770a282bbfa0ae1ec474b7ed311656d.svg#card=math&code=v&id=ar3G0)，对应物理上的速度。<br />上面第一个式子表示类物体在梯度方向上的受力，在这个力的作用下，物体的速度增加这一物理法则。如下图所示，Momentum方法给人的感觉就像是小球在地面上滚动。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682475517203-0ca2a041-1024-4c4c-9845-452688d3b9cb.png#averageHue=%23424242&clientId=u7e2c328a-3440-4&from=paste&height=109&id=u46ad7634&originHeight=136&originWidth=862&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16077&status=done&style=none&taskId=u71f73332-0e0b-4321-8c23-a142cb17701&title=&width=689.6)<br />第一个式子中有![](https://cdn.nlark.com/yuque/__latex/03730898a71a86a75da1612602691f66.svg#card=math&code=%5Calpha%20v&id=s5lem)这一项。在在物体不受任何力时，该项承担使物体逐渐减速的任务（α设定为0.9之类的值），对应物理上的地面摩擦或空气阻力。下面是Momentum的代码实现。
```python
class Momentum:
    """Momentum SGD"""
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
```
实例变量v会保存物体的速度。初始化时，v中什么都不保存，但当第一次调用`update()`时，v会以字典型变量的形式保存与参数结构相同的数据。剩余部分的代码就是将上面两个数学式写出来。

现在尝试用Momentum解决![](https://cdn.nlark.com/yuque/__latex/e35c8aabe2007e96db6bf421d268465b.svg#card=math&code=f%28x%2Cy%29%3D%5Cfrac%7B1%7D%7B20%7Dx%5E2%2By%5E2&id=aOLA8)的最优化问题。如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682476525886-612623d6-a933-45b6-96bf-ed9c3cbfbdea.png#averageHue=%23414141&clientId=u7e2c328a-3440-4&from=paste&height=463&id=uc7793ada&originHeight=579&originWidth=862&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=60134&status=done&style=none&taskId=u527e3da3-9e7d-488a-a06b-d672f285a96&title=&width=689.6)<br />上图中，更新路径就像小球在碗中滚动一样。和SGD相比，我们发现“之”字形的“程度”减轻了。这是因为虽然x轴方向上受到的力非常小，但是一直在同一方向上受力，所以朝同一个方向会有一定的加速。反过来，虽然y轴方向上受到的力很大，但是因为交互地受到正方向和反方向的力，它们会互相抵消，所以y轴方向上的速度不稳定。因此，和SGD时的情形相比，可以更快地朝x轴方向靠近，减弱“之”字形的变动程度。

### 6.1.3 AdaGrad
在神经网络的学习中，学习率（数学式中记为η）的值很重要。学习率过小，会导致学习花费过多时间；反过来，学习率过大，则会导致学习发散而不能正确进行。<br />在关于学习率的有效技巧中，有一种被称为**学习率衰减**（learning rate decay）的方法，即随着学习的进行，使学习率逐渐减小。实际上，一开始“多”学，然后逐渐“少”学的方法，在神经网络的学习中经常被使用。<br />逐渐减小学习率的想法，相当于将“全体”参数的学习率值一起降低。而AdaGrad进一步发展了这个想法，针对“一个一个”的参数，赋予其“定制”的值。<br />AdaGrad会为参数的每个元素适当地调整学习率，与此同时进行学习。下面，用数学式表示AdaGrad的更新方法。<br />![](https://cdn.nlark.com/yuque/__latex/421cd4bc4772fed6d3be300893336da9.svg#card=math&code=h%5Cleftarrow%20h%2B%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%20%5Codot%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%20%5C%5C%20%5C%20%5C%5C%0AW%5Cleftarrow%20W-%5Ceta%5Cfrac%7B1%7D%7B%5Csqrt%7Bh%7D%7D%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D&id=DSKe0)<br />这里和前面的SGD一样，新出现的变量h，保存了以前所有梯度值的平方和（第一个式子中的![](https://cdn.nlark.com/yuque/__latex/91ce821a8ca33e5aa3c3361109e6a911.svg#card=math&code=%5Codot%0A&id=zSnU5)表示对应矩阵元素的乘法）。然后在更新参数时，通过乘以![](https://cdn.nlark.com/yuque/__latex/611e428d10b70be15264eda0cee04849.svg#card=math&code=%5Cfrac%7B1%7D%7B%5Csqrt%7Bh%7D%7D&id=n3qi6)，就可以调整学习的尺度。<br />这意味着，参数的元素中变动较大（被大幅更新）的元素的学习率将变小。也就是说，可以按参数的元素进行学习率衰减，使变动大的参数的学习率逐渐减小。

下面用Python来实现AdaGrad。
```python
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None


    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in  params.items():
                self.h[key] = np.zeros_like(val)


        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```
这里需要注意的是，最后一行加上了微小值1e-7。这是为了防止当`self.h[key]`中有0时，将0用作除数的情况。

试着使用AdaGrad解决![](https://cdn.nlark.com/yuque/__latex/e35c8aabe2007e96db6bf421d268465b.svg#card=math&code=f%28x%2Cy%29%3D%5Cfrac%7B1%7D%7B20%7Dx%5E2%2By%5E2&id=O0gAI)的最优化问题。如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682478475723-14b9bba4-0e0c-4882-9d37-d77db20e8f4e.png#averageHue=%23414141&clientId=u7e2c328a-3440-4&from=paste&height=486&id=u89f7d043&originHeight=607&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=60314&status=done&style=none&taskId=u0cb6f1b8-135f-4f52-975d-c39aa464243&title=&width=688)<br />由上图的结果可知，函数的取值高效地向着最小值移动。由于y轴方向上的梯度较大，因此刚开始变动较大，但是后面会根据这个较大的变动按比例进行调整，减小更新的步伐。因此，y轴方向上的更新程度被减弱，“之”字形的变动程度有所衰减。

### 6.1.4 Adam
Adam理论方法：融合了Momentum和AdaGrad的方法。通过组合前面两个方法的优点，有望实现参数空间的高效搜索。<br />试着使用Adam解决式![](https://cdn.nlark.com/yuque/__latex/e35c8aabe2007e96db6bf421d268465b.svg#card=math&code=f%28x%2Cy%29%3D%5Cfrac%7B1%7D%7B20%7Dx%5E2%2By%5E2&id=LcAw9)的最优化问题。如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682513133337-de38f409-422a-4570-b92b-7ac49a69e8cf.png#averageHue=%23414141&clientId=u25675a7c-ab54-4&from=paste&height=484&id=u1423dab6&originHeight=605&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=60189&status=done&style=none&taskId=u48115f98-1c27-4f5c-8e32-42203d37e46&title=&width=688)<br />在上图中，基于 Adam 的更新过程就像小球在碗中滚动一样。虽然Momentun也有类似的移动，但是相比之下，Adam的小球左右摇晃的程度有所减轻。这得益于学习的更新程度被适当地调整了。

### 6.1.5 更新方法的选择
下面来比较一下4种方法。如下图所示，根据使用的方法不同，参数更新的路径也不同。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682514246011-1bfa2757-912e-455d-8a21-30452fa9425e.png#averageHue=%23faf9f9&clientId=u25675a7c-ab54-4&from=paste&height=762&id=u9702b908&originHeight=953&originWidth=1305&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=182186&status=done&style=none&taskId=u6baa7f98-7ace-4511-aa77-b585db268c9&title=&width=1044)<br />只看这个图的话，AdaGrad似乎是最好的，不过也要注意，结果会根据要解决的问题而变。并且，很显然，超参数（学习率等）的设定值不同，结果也会发生变化。

### 6.1.6 基于MNIST数据集的更新方法的比较
以手写数字识别为例，比较SGD、Momentum、AdaGrad、Adam这4种方法，并确认不同的方法在学习进展上有多大程度的差异。比较结果如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682515939024-842bfbe9-5845-4b85-b6c6-9da186b96757.png#averageHue=%23faf6f4&clientId=u25675a7c-ab54-4&from=paste&height=480&id=u79d502e7&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=82010&status=done&style=none&taskId=u1dda623f-e15e-4467-a508-fc8e57ef941&title=&width=640)<br />横轴表示学习的迭代次数（iteration），纵轴表示损失函数的值（loss）

由上图可知，与SGD相比，其他3种方法学习得更快，而且速度基本相同，仔细看的话，AdaGrad的学习进行得稍微快一点。这个实验需要注意的地方是，实验结果会随学习率等超参数、神经网络的结构（几层深等）的不同而发生变化。不过，一般而言，与SGD相比，其他3种方法可以学习得更快，有时最终的识别精度也更高。

## 6.2 权重的初始值
### 6.2.1 可以将权重初始值设为0吗
> 将权重初始值设为0的话，将无法正确进行学习。

这是因为在误差反向传播法中，所有的权重值都会进行相同的更新。比如，在2层神经网络中，假设第1层和第2层的权重为0。这样一来，正向传播时，因为输入层的权重为0，所以第2层的神经元全部会被传递相同的值。第2层的神经元中全部输入相同的值，这意味着反向传播时第2层的权重全部都会进行相同的更新。因此，权重被更新为相同的值，并拥有了对称的值（重复的值），这使得神经网络拥有许多不同的权重的意义丧失了。为了防止“权重均一化”（严格地讲，是为了瓦解权重的对称结构），必须随机生成初始值。

### 6.2.2 隐藏层的激活值的分布
下面做一个实验，观察权重初始值是如何影响隐藏层的激活值的分布的。向一个5层神经网络（激活函数使用sigmoid函数）传入随机生成的输入数据，用直方图绘制各层激活值的数据分<br />布。实验代码如下。
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

# 接受一个参数x并返回其双曲正切值
def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000, 100) # 1000个数据
node_num = 100 # 各隐藏层的节点（神经元）数
hidden_layer_size = 5 # 隐藏层有5层
activations = {} # 激活值的结果保存在这里

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    
    w = np.random.randn(node_num, node_num) * 1

    a = np.dot(x, w)
    z = sigmoid(a)

    activations[i] = z

# 绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([],[]) # 将y轴的刻度标签设置为空
    plt.hist(a.flatten(), 30, range=(0,1)) # 绘制直方图

plt.show()
```
这里假设神经网络有5层，每层有100个神经元。然后，用高斯分布随机生成1000个数据作为输入数据，并把它们传给5层神经网络。激活函数使用sigmoid函数，各层的激活值的结果保存在activations变量中。

代码绘制的直方图如下所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682649909027-73d1027a-03d7-4561-98c0-c2647f4ae957.png#averageHue=%23f8f8f8&clientId=u1e7ead7a-7a9f-4&from=paste&height=480&id=u92e7b9ca&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14784&status=done&style=none&taskId=u0649bade-571e-4268-a8d2-a8ff12ad233&title=&width=640)<br />从上图可知，各层的激活值呈偏向0和1的分布。这里使用的sigmoid函数是S型函数，随着输出不断地靠近0（或者靠近1），它的导数的值逐渐接近0。因此，偏向0和1的数据分布会造成反向传播中梯度的值不断变小，最后消失。这个问题称为梯度消失（gradient vanishing）。层次加深的深度学习中，梯度消失的问题可能会更加严重。

下面，将权重的标准差设为0.01，进行相同的实验。
```python
# w = np.random.randn(node_num, node_num) * 1
w = np.random.randn(node_num, node_num) * 0.01
```
使用标准差为0.01的高斯分布时，各层的激活值的分布如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682650505915-8efc2109-a60a-41c0-ab66-5a9f0f230bf4.png#averageHue=%23f8f8f8&clientId=u1e7ead7a-7a9f-4&from=paste&height=480&id=u9203b624&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=15126&status=done&style=none&taskId=u535231fc-854e-4c99-afd4-4b9b00c08a7&title=&width=640)<br />这次呈集中在0.5附近的分布。因为不像刚才的例子那样偏向0和1，所以不会发生梯度消失的问题。但是，激活值的分布有所偏向，说明在表现力上会有很大问题。因为如果有多个神经元都输出几乎相同的值，那它们就没有存在的意义了。

为了使各层的激活值呈现出具有相同广度的分布，需要合适的权重尺度。结论：如果前一层的节点数为n，则初始值使用标准差为![](https://cdn.nlark.com/yuque/__latex/a9d77d3d6322e1d940f78a792f066b02.svg#card=math&code=%5Cfrac%7B1%7D%7B%5Csqrt%7Bn%7D%7D&id=m9dzi)的分布。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682650688567-6b711280-00c8-45b4-a93a-8606ed6cea95.png#averageHue=%23424242&clientId=u1e7ead7a-7a9f-4&from=paste&height=375&id=uc19e6e48&originHeight=750&originWidth=1329&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=100912&status=done&style=none&taskId=u268921ef-8915-4a67-a692-746f9cd98da&title=&width=665)

使用Xavier初始值后，前一层的节点数越多，要设定为目标节点的初始值的权重尺度就越小。使用Xavier初始值进行如下实验。
```python
node_num = 100 # 前一层的节点数
w = np.random.randn(node_num, node_num) / np.sqrt(1.0/node_num)
```

使用Xavier初始值后的结果如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682650854613-10dfb797-8236-42a2-82a4-9b9c15a25d32.png#averageHue=%23f8f8f8&clientId=u1e7ead7a-7a9f-4&from=paste&height=480&id=u6da83faf&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12793&status=done&style=none&taskId=u2c409c10-03f7-4ab3-8761-d548200e81a&title=&width=640)<br />从这个结果可知，越是后面的层，图像变得越歪斜，但是呈现了比之前更有广度的分布。因为各层间传递的数据有适当的广度，所以sigmoid函数的表现力不受限制，有望进行高效的学习。

### 6.2.3 ReLU的权重初始值
当激活函数使用ReLU时，一般推荐使用ReLU专用的初始值，也就是Kaiming He等人推荐的初始值，也称为“He初始值”。当前一层的节点数为n 时，He 初始值使用标准差为![](https://cdn.nlark.com/yuque/__latex/f8a22f43ff54ac581b5e856d03f0636a.svg#card=math&code=%5Csqrt%7B%5Cfrac%7B2%7D%7Bn%7D%7D&id=K66s6)的高斯分布。<br />下面来看一下激活函数使用ReLU时激活值的分布。给出了3个实验的结果，依次是权重初始值为标准差是0.01的高斯分布（下文简写为“std = 0.01”）时、初始值为Xavier初始值时、初始值为ReLU专用的“He初始值”时的结果<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682733429237-d0604fd2-44a7-4fea-a4d0-3464cc7f2752.png#averageHue=%23f8f8f8&clientId=u8524795f-50e2-4&from=paste&height=300&id=u7b9b5b8d&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13020&status=done&style=none&taskId=u7d322b8c-9977-4a9c-91aa-c92d9ddd3a0&title=&width=400)![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682733471102-45635c13-d3f3-479f-a186-dbc9a995935f.png#averageHue=%23f8f8f8&clientId=u8524795f-50e2-4&from=paste&height=300&id=ud2b6f227&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12411&status=done&style=none&taskId=ue24497e4-93c0-4ccd-9311-e77b5adc198&title=&width=400)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682733511153-bb8c90a3-7f3b-4f0d-8097-4e85f798c449.png#averageHue=%23f8f8f8&clientId=u8524795f-50e2-4&from=paste&height=300&id=u3e1edfb6&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12253&status=done&style=none&taskId=u6183ab7d-a74b-40f6-bb8a-96139b4bbdd&title=&width=400)<br />观察实验结果可知

1. 当“std = 0.01”时，各层的激活值非常小
2. 接下来是初始值为Xavier初始值时的结果。在这种情况下，随着层的加深，偏向一点点变大。实际上，层加深后，激活值的偏向变大，学习时会出现梯度消失的问题。
3. 当初始值为He初始值时，各层中分布的广度相同。由于即便层加深，数据的广度也能保持不变，因此逆向传播时，也会传递合适的值。

总结一下，当激活函数使用ReLU时，权重初始值使用He初始值，当激活函数为sigmoid或tanh等S型曲线函数时，初始值使用Xavier初始值。这是目前的最佳实践。

### 6.2.4 基于MNIST数据集的权重初始值的比较
下面通过实际的数据，观察不同的权重初始值的赋值方法会在多大程度上影响神经网络的学习。这里，我们基于std = 0.01、Xavier初始值、He初始值进行实验。实验结果如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682768068884-2bcbf1db-613c-443a-afac-348ffbb0199f.png#averageHue=%23fbfaf8&clientId=ud03c7d79-b997-4&from=paste&height=480&id=u319d14d8&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=47513&status=done&style=none&taskId=u843f4256-a327-41a7-b680-221362a9070&title=&width=640)

这个实验中，神经网络有5层，每层有100个神经元，激活函数使用的是ReLU。从上图的结果可知，std = 0.01时完全无法进行学习。这和刚才观察到的激活值的分布一样，是因为正向传播中传递的值很小（集中在0附近的数据）。因此，逆向传播时求到的梯度也很小，权重几乎不进行更新。相反，当权重初始值为Xavier初始值和He初始值时，学习进行得很顺利。并且，我们发现He初始值时的学习进度更快一些。

## 6.3 Batch Normalization
### 6.3.1 Batch Normalization 的算法
优点：

- 可以使学习快速进行（可以增大学习率）。
- 不那么依赖初始值（对于初始值不用那么神经质）。
- 抑制过拟合（降低Dropout等的必要性）。

Batch Norm的思路是调整各层的激活值分布使其拥有适当的广度。为此，要向神经网络中插入对数据分布进行正规化的层，即Batch Normalization层（下文简称Batch Norm层），如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682768562624-21b1c53d-ea35-4495-88ed-f220195bfdaf.png#averageHue=%23444444&clientId=ud03c7d79-b997-4&from=paste&height=201&id=u0f6c4c20&originHeight=251&originWidth=858&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14812&status=done&style=none&taskId=u2533e81f-98ea-4ca1-8ff4-87d57a5e70a&title=&width=686.4)

Batch Norm，顾名思义，以进行学习时的mini-batch为单位，按mini-batch进行正规化。具体而言，就是进行使数据分布的均值为0、方差为1的正规化。用数学式表示的话，如下所示。<br />![](https://cdn.nlark.com/yuque/__latex/7624513dfb7efbd9451d99186e84ed34.svg#card=math&code=%5Cmu_B%20%5Cleftarrow%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Emx_i%20%5C%5C%0A%5C%20%5C%5C%0A%5Csigma%5E2_B%20%5Cleftarrow%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%28x_i-%5Cmu_B%29%5E2%20%5C%5C%0A%5C%20%5C%5C%0A%5Chat%7Bx%7D%20%5Cleftarrow%20%5Cfrac%7Bx_i-%5Cmu_B%7D%7B%5Csqrt%7B%5Csigma_B%5E2%2B%5Cepsilon%7D%7D%0A%0A&id=K09Jt)<br />这里对 mini-batch 的 m 个输入数据的集合 B = {x1, x2, . . . , xm} 求均值![](https://cdn.nlark.com/yuque/__latex/342ce27e84d7a807b2cc3bc48fa19785.svg#card=math&code=%5Cmu_B&id=oPG9x)和方差![](https://cdn.nlark.com/yuque/__latex/44bd7a146d2d3d5be1e887d1e976647d.svg#card=math&code=%5Csigma%5E2_B&id=MIKQM)。然后，对输入数据进行均值为0、方差为1（合适的分布）的正规化。第三个式子中的 ![](https://cdn.nlark.com/yuque/__latex/7c102e7a7d231bf935f9bc23417779a8.svg#card=math&code=%5Cepsilon&id=cKKh7)是一个微小值，这是为了防止出现除以0的情况。<br />上式所做的是将mini-batch的输入数据{x1, x2, . . . , xm}变换为均值为0、方差为1的数据 ![](https://cdn.nlark.com/yuque/__latex/c3250d8e3ddfbbb53294f4de57c62c6b.svg#card=math&code=%5Chat%7Bx%7D&id=L5W6k)。通过将这个处理插入到激活函数的前面（或者后面），可以减小数据分布的偏向。

接着，Batch Norm层会对正规化后的数据进行缩放和平移的变换，用数学式可以如下表示<br />![](https://cdn.nlark.com/yuque/__latex/eb59939534f83061388c4deef455a90b.svg#card=math&code=y_i%20%5Cleftarrow%20%5Cgamma%20%5Chat%7Bx_i%7D%20%2B%20%5Cbeta&id=yzknH)<br />这里，![](https://cdn.nlark.com/yuque/__latex/4aa418d6f0b6fbada90489b4374752e5.svg#card=math&code=%5Cgamma&id=TJ5RZ)和 ![](https://cdn.nlark.com/yuque/__latex/6100158802e722a88c15efc101fc275b.svg#card=math&code=%5Cbeta&id=LKUG5)是参数，一开始![](https://cdn.nlark.com/yuque/__latex/c60931bf99aee7b205a1bd42f4ceeaea.svg#card=math&code=%5Cgamma%20%3D%201%2C%20%5Cbeta%20%3D%200&id=Wgjkg)，然后再通过学习调整到合适的值。

以上就是Batch Norm的算法。这个算法是神经网络上的正向传播。用计算图可以表示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682770047200-fd01046e-a3fb-4e2e-a6e4-5ef3f6b6a019.png#averageHue=%23424242&clientId=ud03c7d79-b997-4&from=paste&height=214&id=ucaaab434&originHeight=268&originWidth=858&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=40979&status=done&style=none&taskId=u92ab5981-8616-4877-8bda-ead14a25600&title=&width=686.4)

### 6.3.2 Batch Normalization的评估
现在我们使用 Batch Norm 层进行实验。首先，使用 MNIST 数据集，观察使用Batch Norm层和不使用Batch Norm层时学习的过程会如何变化。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682817854310-680f5693-9d21-4470-938f-93e59a155990.png#averageHue=%23414141&clientId=ua97b6888-9e44-4&from=paste&height=451&id=u8cac3f66&originHeight=564&originWidth=702&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=43112&status=done&style=none&taskId=uea15a25b-62a1-4f59-b231-487b202b926&title=&width=561.6)<br />由上图结果可知，使用Batch Norm后，学习进行得更快了。

接着，给予不同的初始值尺度，观察学习的过程如何变化。下图是权重初始值的标准差为各种不同的值时的学习过程图。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682818095644-eccf54e1-7a46-4c32-9846-90c8ae867582.png#averageHue=%23fafaf9&clientId=ua97b6888-9e44-4&from=paste&height=1054&id=ue9fc8ed6&originHeight=1317&originWidth=2560&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=222360&status=done&style=none&taskId=ud48fd730-d5d3-4839-b91d-e25021ea67a&title=&width=2048)<br />我们发现，几乎所有的情况下都是使用Batch Norm时学习进行得更快。同时也可以发现，实际上，在不使用Batch Norm的情况下，如果不赋予一个尺度好的初始值，学习将完全无法进行。

综上，通过使用Batch Norm，可以推动学习的进行。并且，对权重初始值变得健壮（“对初始值健壮”表示不那么依赖初始值）。

## 6.4 正则化
> 过拟合指的是只能拟合训练数据，但不能很好地拟合不包含在训练数据中的其他数据的状态。


### 6.4.1 过拟合
发生过拟合的原因，主要有以下两个。

- 模型拥有大量参数、表现力强。
- 训练数据少。

这里，我们故意满足这两个条件，制造过拟合现象。为此，要从MNIST数据集原本的60000个训练数据中只选定300个，并且，为了增加网络的复杂度，使用7层网络（每层有100个神经元，激活函数为ReLU）。

1. 读入数据
```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
# 为了再现过拟合，减少学习数据
x_train = x_train[:300]
t_train = t_train[:300]
```

2. 训练数据（按epoch分别算出所有训练数据和所有测试数据的识别精度）
```python
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100,100, 100, 100], output_size=10)
optimizer = SGD(lr=0.01) # 用学习率为0.01的SGD更新参数

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break
```

train_acc_list和test_acc_list中以epoch为单位（看完了所有训练数据的单位）保存识别精度。现在，我们将这些列表（train_acc_list、test_acc_list）绘成图，结果如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682821248050-e42eb92b-fad5-4170-94ab-405a180a4c3b.png#averageHue=%23fcfaf9&clientId=ua97b6888-9e44-4&from=paste&height=480&id=u4b4fbace&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=32232&status=done&style=none&taskId=u516d73f0-feba-4550-adbf-5efc77a6f6d&title=&width=640)<br />过了 100 个 epoch 左右后，用训练数据测量到的识别精度几乎都为100%。但是，对于测试数据，离100%的识别精度还有较大的差距。如此大的识别精度差距，是只拟合了训练数据的结果。从图中可知，模型对训练时没有使用的一般数据（测试数据）拟合得不是很好。

### 6.4.2 权值衰减
> 权值衰减是一直以来经常被使用的一种抑制过拟合的方法。该方法通过在学习的过程中对大的权重进行惩罚，来抑制过拟合。很多过拟合原本就是因为权重参数取值过大才发生的。

如果将权重记为W，L2范数的权值衰减就是![](https://cdn.nlark.com/yuque/__latex/e4a2bd7b70cde95138f14c6fc1c332d5.svg#card=math&code=%5Cfrac%7B1%7D%7B2%7D%20%5Clambda%20W%5E2&id=oGv5s)，然后将这个![](https://cdn.nlark.com/yuque/__latex/e4a2bd7b70cde95138f14c6fc1c332d5.svg#card=math&code=%5Cfrac%7B1%7D%7B2%7D%20%5Clambda%20W%5E2&id=Y72TZ)加到损失函数上。这里，λ是控制正则化强度的超参数。λ设置得越大，对大的权重施加的惩罚就越重。此外，![](https://cdn.nlark.com/yuque/__latex/e4a2bd7b70cde95138f14c6fc1c332d5.svg#card=math&code=%5Cfrac%7B1%7D%7B2%7D%20%5Clambda%20W%5E2&id=d8xlx)开头的![](https://cdn.nlark.com/yuque/__latex/72067414d4f00caec2212e5c10479a88.svg#card=math&code=1%5Cover2&id=bobmd)是用于将![](https://cdn.nlark.com/yuque/__latex/e4a2bd7b70cde95138f14c6fc1c332d5.svg#card=math&code=%5Cfrac%7B1%7D%7B2%7D%20%5Clambda%20W%5E2&id=iVmmj)的求导结果变成λW的调整用常量。

对于刚刚进行的实验，应用λ = 0.1的权值衰减，结果如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682821534433-df23fbae-f570-41be-b341-466397f7b4ed.png#averageHue=%23fcfaf9&clientId=ua97b6888-9e44-4&from=paste&height=480&id=u75f903cc&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=39085&status=done&style=none&taskId=udf1d265f-5f85-40cd-bad3-bcabea7bcfd&title=&width=640)<br />虽然训练数据的识别精度和测试数据的识别精度之间有差距，但是与没有使用权值衰减的上图的结果相比，差距变小了。这说明过拟合受到了抑制。此外，还要注意，训练数据的识别精度没有达到100%（1.0）

### 6.4.3 Dropout
> Dropout是一种在学习的过程中随机删除神经元的方法。训练时，随机选出隐藏层的神经元，然后将其删除。

被删除的神经元不再进行信号的传递，如下图所示。训练时，每传递一次数据，就会随机选择要删除的神经元。然后，测试时，虽然会传递所有的神经元信号，但是对于各个神经元的输出，要乘上训练时的删除比例后再输出。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682821635957-8a97ad89-1f61-4600-b5be-397bbc89a0a6.png#averageHue=%23434343&clientId=ua97b6888-9e44-4&from=paste&height=305&id=u643b7bf4&originHeight=381&originWidth=859&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=158316&status=done&style=none&taskId=u6d9ae83d-4c48-4ed0-bc6f-c0947077f04&title=&width=687.2)

下面我们来实现Dropout。这里的实现重视易理解性。不过，因为训练时如果进行恰当的计算的话，正向传播时单纯地传递数据就可以了（不用乘以删除比例），所以深度学习的框架中进行了这样的实现。
```python
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask
```
这里的要点是，每次正向传播时，self.mask中都会以False的形式保存要删除的神经元。self.mask会随机生成和x形状相同的数组，并将值比dropout_ratio大的元素设为True。反向传播时的行为和ReLU相同。也就是说，正向传播时传递了信号的神经元，反向传播时按原样传递信号；正向传播时没有传递信号的神经元，反向传播时信号将停在那里。

现在，我们使用MNIST数据集进行验证，以确认Dropout的效果。另外，源代码中使用了Trainer类来简化实现。Dropout的实验和前面的实验一样，使用7层网络（每层有100个神经元，激活函数为ReLU），一个使用Dropout，另一个不使用Dropout，实验的结果如图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682822834369-b1b18c90-05c2-489a-bfbd-df64b9fc4549.png#averageHue=%23fcfaf9&clientId=ua97b6888-9e44-4&from=paste&height=300&id=ue9a2f6a4&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=29915&status=done&style=none&taskId=u44c32dd7-d7da-4c2d-8e2c-426d53a0cea&title=&width=400)![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682822620871-40090cc1-662f-4b8c-b325-54a0e1ae25ae.png#averageHue=%23fcfaf9&clientId=ua97b6888-9e44-4&from=paste&height=300&id=u179462e6&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=34184&status=done&style=none&taskId=uc451006a-5e0e-4f86-a355-82adb4caa91&title=&width=400)<br />上图中，通过使用Dropout，训练数据和测试数据的识别精度的差距变小了。并且，训练数据也没有到达100%的识别精度。像这样，通过使用Dropout，即便是表现力强的网络，也可以抑制过拟合。


## 6.5 超参数的验证
> 超参数：各层的神经元数量、batch大小、参数更新时的学习率或权值衰减等


### 6.5.1 验证数据
注意：不能使用测试数据评估超参数的性能。这是因为如果使用测试数据调整超参数，超参数的值会对测试数据发生过拟合。因此，调整超参数时，必须使用超参数专用的确认数据。
> 用于调整超参数的数据，一般称为验证数据（validation data）。我们使用这个验证数据来评估超参数的好坏。

训练数据用于参数（权重和偏置）的学习，验证数据用于超参数的性能评估。为了确认泛化能力，要在最后使用（比较理想的是只用一次）测试数据。

根据不同的数据集，有的会事先分成训练数据、验证数据、测试数据三部分，有的只分成训练数据和测试数据两部分，有的则不进行分割。在这种情况下，用户需要自行进行分割。如果是MNIST数据集，获得验证数据的最简单的方法就是从训练数据中事先分割20%作为验证数据，代码如下所示。
```python
(x_train, t_train), (x_test, t_test) = load_mnist()

# 打乱训练数据
x_train, t_train = shuffle_dataset(x_train, t_train)

# 分割验证数据
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]
```
这里，分割训练数据前，先打乱了输入数据和教师标签。这是因为数据集的数据可能存在偏向（比如，数据从“0”到“10”按顺序排列等）。

### 6.5.2 超参数的最优化
进行超参数的最优化时，需要逐渐缩小超参数的“好值”的存在范围。在超参数的搜索中，需要尽早放弃那些不符合逻辑的超参数。于是，在超参数的最优化中，减少学习的epoch，缩短一次评估所需的时间是一个不错的办法。归纳超参数最优化内容如下。

1. 设定超参数的范围。
2. 从设定的超参数范围中随机采样。
3. 使用步骤1中采样到的超参数的值进行学习，通过验证数据评估识别精度（但是要将epoch设置得很小）。
4. 重复步骤1和步骤2（100次等），根据它们的识别精度的结果，缩小超参数的范围。

反复进行上述操作，不断缩小超参数的范围，在缩小到一定程度时，从该范围中选出一个超参数的值。这就是进行超参数的最优化的一种方法。

### 6.5.3 超参数最优化的实现
现在，我们使用MNIST数据集进行超参数的最优化。这里我们将学习率和控制权值衰减强度的系数（下文称为“权值衰减系数”）这两个超参数的搜索问题作为对象。

通过从 0.001到 1000这样的对数尺度的范围中随机采样进行超参数的验证。这在Python中可以写成10 ** np.random.uniform(-3, 3)。在该实验中，权值衰减系数的初始范围为![](https://cdn.nlark.com/yuque/__latex/69806b82ffec24468b61d0dfa2945017.svg#card=math&code=10%5E%7B-8%7D&id=hk4vo)到![](https://cdn.nlark.com/yuque/__latex/0862c3e735b7923c475f6e3c30ff4961.svg#card=math&code=10%5E%7B-4%7D&id=lmS6V)，学习率的初始范围为![](https://cdn.nlark.com/yuque/__latex/1f35f0355bf21f1a16eea780823a6768.svg#card=math&code=10%5E%7B-6%7D&id=Urkht)到![](https://cdn.nlark.com/yuque/__latex/b6ff88cd45e46c1201e2a4dcdafd91e0.svg#card=math&code=10%5E%7B-2%7D&id=bpx5P)。此时，超参数的随机采样的代码如下所示。
```python
weight_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)
```
像这样进行随机采样后，再使用那些值进行学习。之后，多次使用各种超参数的值重复进行学习，观察合乎逻辑的超参数在哪里。实验结果如下所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682843311782-0d3ff2a7-14af-43bd-a41d-ad9a22b2fde4.png#averageHue=%23fbfbfa&clientId=ua97b6888-9e44-4&from=paste&height=1054&id=sR5q3&originHeight=1317&originWidth=2560&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=209147&status=done&style=none&taskId=u2ff7ce99-b341-49fb-bec8-37cff698b0f&title=&width=2048)<br />上图中，按识别精度从高到低的顺序排列了验证数据的学习的变化。从图中可知，直到“Best-5”左右，学习进行得都很顺利。因此，我们来观察一下“Best-5”之前的超参数的值（学习率和权值衰减系数），结果如下所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1682843377286-cb3295f9-032a-4aad-9568-a26a56bfb40e.png#averageHue=%23313a41&clientId=ua97b6888-9e44-4&from=paste&height=98&id=u90b4368b&originHeight=122&originWidth=859&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=35635&status=done&style=none&taskId=uc3088e59-23ec-4d62-9745-3cac4ab18aa&title=&width=687.2)<br />从这个结果可以看出，学习率在0.001到0.01、权值衰减系数在![](https://cdn.nlark.com/yuque/__latex/69806b82ffec24468b61d0dfa2945017.svg#card=math&code=10%5E%7B-8%7D&id=CWqYr)到![](https://cdn.nlark.com/yuque/__latex/1f35f0355bf21f1a16eea780823a6768.svg#card=math&code=10%5E%7B-6%7D&id=n0GhO)之间时，学习可以顺利进行。像这样，观察可以使学习顺利进行的超参数的范围，从而缩小值的范围。然后，在这个缩小的范围中重复相同的操作。这样就能缩小到合适的超参数的存在范围，然后在某个阶段，选择一个最终的超参数的值。

# 7 卷积神经网络
> 卷积神经网络（Convolutional Neural Network，CNN），CNN被用于图像识别、语音识别等各种场合


## 7.1 整体结构
> CNN中新出现了卷积层（Convolution层）和池化层（Pooling层）

之前介绍的神经网络中，相邻层的所有神经元之间都有连接，这称为全连接（fully-connected）。另外，我们用Affine层实现了全连接层。如果使用这个Affine层，一个5层的全连接的神经网络就可以通过下图所示的网络结构来实现。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683188143169-30350504-8646-4143-9309-8bdf56379782.png#averageHue=%23424242&clientId=u7b232bfd-962a-4&from=paste&height=168&id=uf14fcadc&originHeight=210&originWidth=863&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=11770&status=done&style=none&taskId=u4c76c176-d6da-4c46-b9f5-3c6ef01e4db&title=&width=690.4)<br />如上图所示，全连接的神经网络中，Affine层后面跟着激活函数ReLU层（或者 Sigmoid 层）。这里堆叠了 4 层“Affine-ReLU”组合，然后第 5 层是Affine层，最后由Softmax层输出最终结果（概率）。

而下面是CNN的一个例子。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683188209057-51b95893-8a23-48e4-bc28-b18d608d1ae0.png#averageHue=%23464646&clientId=u7b232bfd-962a-4&from=paste&height=151&id=ua7b542a6&originHeight=189&originWidth=861&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13883&status=done&style=none&taskId=u214c0d4e-36e5-4676-be08-f29f2cef93a&title=&width=688.8)<br />如上图所示，CNN中新增了 Convolution 层和 Pooling 层。CNN 的层的连接顺序是“Convolution - ReLU -（Pooling）”（Pooling 层有时会被省略）。这可以理解为之前的“Affine - ReLU”连接被替换成了“Convolution - ReLU -（Pooling）”连接。

还需要注意的是，在上图的 CNN 中，靠近输出的层中使用了之前的“Affine - ReLU”组合。此外，最后的输出层中使用了之前的“Affine - Softmax”组合。这些都是一般的CNN中比较常见的结构。

## 7.2 卷积层
### 7.2.1 全连接层存在的问题
> 在全连接层中，相邻层的神经元全部连接在一起，输出的数量可以任意决定。


全连接层存在的问题：数据的形状被“忽视”了。比如，输入数据是图像时，图像通常是高、长、通道方向上的3维形状。但是，向全连接层输入时，需要将3维数据拉平为1维数据。因为全连接层会忽视形状，将全部的输入数据作为相同的神经元（同一维度的神经元）处理，所以无法利用与形状相关的信息

而卷积层可以保持形状不变。当输入数据是图像时，卷积层会以 3 维数据的形式接收输入数据，并同样以3 维数据的形式输出至下一层。因此，在 CNN 中，可以（有可能）正确理解图像等具有形状的数据。

在 CNN 中，有时将卷积层的输入输出数据称为特征图（feature map）。其中，卷积层的输入数据称为输入特征图（input feature map），输出数据称为输出特征图（output feature map）。

### 7.2.2 卷积运算
卷积层进行的处理就是卷积运算。卷积运算相当于图像处理中的“滤波器运算”。下面是一个具体例子。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683190065731-75863618-0fa1-42b4-9288-e89559893006.png#averageHue=%23414141&clientId=u7b232bfd-962a-4&from=paste&height=246&id=u7d8dbf0c&originHeight=308&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24721&status=done&style=none&taskId=u36959c18-a093-44c8-b7cd-ac71fd2619d&title=&width=688)<br />中间的符号表示卷积运算。如上图所示，卷积运算对输入数据应用滤波器。在这个例子中，输入数据是有高长方向的形状的数据，滤波器也一样，有高长方向上的维度。假设用（height, width）表示数据和滤波器的形状，则在本例中，输入大小是(4, 4)，滤波器大小是(3, 3)，输出大小是(2, 2)。

下面来解释一下卷积运算的例子中都进行了什么样的计算。下图展示了卷积运算的计算顺序。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683190237122-0fda626e-b00b-4c89-b3d7-ac64a515b30b.png#averageHue=%23454545&clientId=u7b232bfd-962a-4&from=paste&height=877&id=u83d1b62f&originHeight=1096&originWidth=859&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=73215&status=done&style=none&taskId=uafe6ac84-0506-4a70-97c8-37aa4113aa8&title=&width=687.2)<br />对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用。这里所说的窗口是指图上图中灰色的3 × 3的部分。<br />如图所示，将**各个位置上滤波器的元素和输入的对应元素相乘**，然后再**求和**（有时将这个计算称为乘积累加运算）。然后，将这个结果保存到输出的对应位置。将这个过程在所有位置都进行一遍，就可以得到卷积运算的输出。

在全连接的神经网络中，除了权重参数，还存在偏置。CNN中，滤波器的参数就对应之前的权重。并且，CNN中也存在偏置。包含偏置的卷积运算的处理流如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683190545003-4bad755e-91ad-4b2a-95c0-9559055cb4f1.png#averageHue=%23424242&clientId=u7b232bfd-962a-4&from=paste&height=174&id=u007e3f21&originHeight=217&originWidth=858&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=29170&status=done&style=none&taskId=u3d42e35f-4971-447d-9b9a-4849bc3ec81&title=&width=686.4)<br />如上图所示，向应用了滤波器的数据加上了偏置。偏置通常只有1个（1 × 1）（本例中，相对于应用了滤波器的4个数据，偏置只有1个），这个值会被加到应用了滤波器的所有元素上。

### 7.2.3 填充
> 在进行卷积层的处理之前，有时要向输入数据的周围填入固定的数据（比如0等），这称为填充（padding）

比如在下图的例子中，对大小为(4, 4)的输入数据应用了幅度为1的填充。（“幅度为1的填充”是指用幅度为1像素的0填充周围。）<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683190719229-7d0e30da-716c-47c1-a372-30d896c56f43.png#averageHue=%23424242&clientId=u7b232bfd-962a-4&from=paste&height=289&id=uc382ff31&originHeight=361&originWidth=859&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=40511&status=done&style=none&taskId=ub3162c68-da55-4073-a03f-9992bd56b52&title=&width=687.2)<br />图中用虚线表示填充，并省略了填充的内容“0”。如上图所示，通过填充，大小为(4, 4)的输入数据变成了(6, 6)的形状。然后，应用大小为(3, 3)的滤波器，生成了大小为(4, 4)的输出数据。

### 7.2.4 步幅
> 应用滤波器的位置间隔称为步幅（stride）

之前的例子中步幅都是1，如果将步幅设为2，则如下图所示，应用滤波器的窗口的间隔变为2个元素。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683190982768-2eb781f6-a055-4cc6-80e5-fda90ecee2d3.png#averageHue=%23444444&clientId=u7b232bfd-962a-4&from=paste&height=490&id=u4dc61767&originHeight=612&originWidth=861&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=61754&status=done&style=none&taskId=ue4ac2b72-aabd-419d-95a3-92cdefd7e9b&title=&width=688.8)<br />像这样，步幅可以指定应用滤波器的间隔。综上，增大步幅后，输出大小会变小。而增大填充后，输出大小会变大。

将这样的关系写成算式。假设输入大小为(H, W)，滤波器大小为(FH, FW)，输出大小为(OH, OW)，填充为P，步幅为S。此时，输出大小可通过下面的式子进行计算。<br />![](https://cdn.nlark.com/yuque/__latex/1bbab030427e686025027005b59113cd.svg#card=math&code=OH%3D%5Cfrac%7BH%2B2P-FH%7D%7BS%7D%2B1%20%5C%5C%0A%5C%20%5C%5C%0AOW%3D%5Cfrac%7BW%2B2P-FW%7D%7BS%7D%2B1&id=scNRy)<br />使用此算式做几个计算。

1. 输入大小(H, W)：(4,4)；填充P：1；步幅S：1；滤波器大小(FH, FW)：(3,3)

![](https://cdn.nlark.com/yuque/__latex/17cc473a17c08b1bc9be940e9328155d.svg#card=math&code=OH%3D%5Cfrac%7B4%2B2%C2%B71-3%7D%7B1%7D%2B1%3D4%20%5C%5C%0A%5C%20%5C%5C%0AOW%3D%5Cfrac%7B4%2B2%C2%B71-3%7D%7B1%7D%2B1%3D4&id=ExYnS)

2. 输入大小(H, W)：(7, 7)；填充P：0；步幅S：2；滤波器大小(FH, FW)：(3, 3)

![](https://cdn.nlark.com/yuque/__latex/fbfe22a8c46aab816511ac6c1db3ce70.svg#card=math&code=OH%3D%5Cfrac%7B7%2B2%C2%B70-3%7D%7B2%7D%2B1%3D3%20%5C%5C%0A%5C%20%5C%5C%0AOW%3D%5Cfrac%7B7%2B2%C2%B70-3%7D%7B2%7D%2B1%3D3&id=fYC0n)

3. 输入大小(H, W)：(28, 31)；填充P：2；步幅S：3；滤波器大小(FH, FW)：(5, 5)

![](https://cdn.nlark.com/yuque/__latex/300aa959e752d8e02e9526fffba68016.svg#card=math&code=OH%3D%5Cfrac%7B28%2B2%C2%B72-5%7D%7B3%7D%2B1%3D10%20%5C%5C%0A%5C%20%5C%5C%0AOW%3D%5Cfrac%7B31%2B2%C2%B72-5%7D%7B3%7D%2B1%3D11&id=b8l7S)

如这些例子所示，通过在计算式中代入值，就可以计算输出大小。这里需要注意的是，虽然只要代入值就可以计算输出大小，但是所设定的值必须使式 ![](https://cdn.nlark.com/yuque/__latex/d8a947a8423980f6d0dfc4103451aee7.svg#card=math&code=%5Cfrac%7BH%2B2P-FH%7D%7BS%7D&id=Ozopp)和![](https://cdn.nlark.com/yuque/__latex/22b641e6606759b9d2ef4d8cf2758d0c.svg#card=math&code=%5Cfrac%7BW%2B2P-FW%7D%7BS%7D&id=QeiIY)分别可以除尽。当输出大小无法除尽时（结果是小数时），需要采取报错等对策。

### 7.2.5 3维数据的卷积运算
下图是卷积运算的例子与计算顺序。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683424319622-15fac21d-59d6-4c1b-ac4e-ec031756150b.png#averageHue=%23424242&clientId=u80612cfc-fbd1-4&from=paste&height=273&id=ua609052e&originHeight=341&originWidth=862&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=42002&status=done&style=none&taskId=u3efb41a0-e17d-43a7-8c2f-225b54c92a9&title=&width=689.6)<br />通道方向上有多个特征图时，会按通道进行输入数据和滤波器的卷积运算，并将结果相加，从而得到输出。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683424417755-b81a5f83-4162-447e-8d4d-2958f42e3bb0.png#averageHue=%23434343&clientId=u80612cfc-fbd1-4&from=paste&height=995&id=u7570ffbd&originHeight=1244&originWidth=863&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=137046&status=done&style=none&taskId=u7820db03-5dc3-4efc-ba80-5bfcdabbd60&title=&width=690.4)

需要注意的是，在3维数据的卷积运算中，输入数据和滤波器的通道数要设为相同的值。滤波器大小可以设定为任意值（不过，每个通道的滤波器大小要全部相同）。

### 7.2.6 结合方块思考
将数据和滤波器结合长方体的方块来考虑，3维数据的卷积运算会很容易理解。把3维数据表示为多维数组时，书写顺序为（channel, height, width）。比如，通道数为 C、高度为 H、长度为W的数据的形状可以写成（C, H, W）。滤波器也一样，要按（channel, height, width）的顺序书写。比如，通道数为 C、滤波器高度为 FH（Filter Height）、长度为FW（Filter Width）时，可以写成（C, FH, FW）。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683424675301-8b32edc6-369c-4b08-bf11-62e3c4a22a96.png#averageHue=%23424242&clientId=u80612cfc-fbd1-4&from=paste&height=353&id=ufc6d8490&originHeight=441&originWidth=863&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=32339&status=done&style=none&taskId=u1bea468c-dfd4-4009-847e-d9ccd0dead5&title=&width=690.4)

在通道方向上也拥有多个卷积运算的输出，就需要用到多个滤波器（权重）。如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683424880151-3b21147f-f220-4ee3-bc0e-897d709f72f6.png#averageHue=%23424242&clientId=u80612cfc-fbd1-4&from=paste&height=405&id=uc1a60729&originHeight=506&originWidth=866&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=44770&status=done&style=none&taskId=u0b2a1787-d9d8-405a-9be3-316773a8386&title=&width=692.8)

通过应用FN个滤波器，输出特征图也生成了FN个。如果将这FN个特征图汇集在一起，就得到了形状为(FN, OH, OW)的方块。将这个方块传给下一层，就是CNN的处理流。

作为4维数据，滤波器的权重数据要按(output_channel, input_channel, height, width) 的顺序书写。比如，通道数为 3、大小为 5 × 5 的滤波器有20个时，可以写成(20, 3, 5, 5)。

在上图的例子中，如果进一步追加偏置的加法运算处理，则结果如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683425092019-f5599111-8c03-4e00-b6c9-4d62c837c3e4.png#averageHue=%23424242&clientId=u80612cfc-fbd1-4&from=paste&height=230&id=ue08e09e4&originHeight=288&originWidth=863&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=37215&status=done&style=none&taskId=ua80b9d67-729b-4801-8ee8-e7ad2a6f0ee&title=&width=690.4)<br />上图中，每个通道只有一个偏置。这里，偏置的形状是 (FN, 1, 1)，滤波器的输出结果的形状是(FN, OH, OW)。这两个方块相加时，要对滤波器的输出结果(FN, OH, OW)按通道加上相同的偏置值。

### 7.2.7 批处理
对于卷积运算的批处理，需要将在各层间传递的数据保存为4维数据。具体地讲，就是按(batch_num, channel, height, width)的顺序保存数据。比如，将上一节中最后一张图的例子中的处理改成对N个数据进行批处理时，数据的形状如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683425403948-fd88f218-3c6e-484a-b77e-aa003647e5c6.png#averageHue=%23424242&clientId=u80612cfc-fbd1-4&from=paste&height=232&id=ued411b7a&originHeight=290&originWidth=867&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=41029&status=done&style=none&taskId=u2ec433e4-c6b0-41f1-bf5e-4d075781aa3&title=&width=693.6)<br />在上图的批处理版的数据流中，在各个数据的开头添加了批用的维度。这里需要注意的是，网络间传递的是 4 维数据，对这 N 个数据进行了卷积运算。也就是说，批处理将 N 次的处理汇总成了 1 次进行。

## 7.3 池化层
> 池化是缩小高、长方向上的空间的运算。

如下图所示，进行将2 × 2的区域集约成1个元素的处理，缩小空间大小。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683427022899-4df51b7e-e6c3-46a0-9bb1-d77897049115.png#averageHue=%23444444&clientId=u80612cfc-fbd1-4&from=paste&height=289&id=ucde02605&originHeight=361&originWidth=866&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=28959&status=done&style=none&taskId=ue0658aee-2a18-4f30-aa66-663789c941c&title=&width=692.8)<br />上图的例子是按步幅2进行 2 × 2 的Max池化时的处理顺序。“Max池化”是获取最大值的运算，“2 × 2”表示目标区域的大小。如图所示，从 2 × 2 的区域中取出最大的元素。此外，这个例子中将步幅设为了2，所以 2 × 2 的窗口的移动间隔为2个元素。一般来说，池化的窗口大小会和步幅设定成相同的值。

池化层的特征：

1. 没有要学习的参数：池化层和卷积层不同，没有要学习的参数。池化只是从目标区域中取最大值（或者平均值），所以不存在要学习的参数。
2. 通道数不发生变化：经过池化运算，输入数据和输出数据的通道数不会发生变化。如下图所示，计算是按通道独立进行的。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683427637521-464007b9-738d-4eea-b1eb-ece1483dec19.png#averageHue=%23424242&clientId=u80612cfc-fbd1-4&from=paste&height=346&id=u2450b782&originHeight=432&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=31907&status=done&style=none&taskId=ubd8f1bf8-f120-48d2-b5c8-5950c0c2348&title=&width=691.2)

3. 对微小的位置变化具有健壮性：输入数据发生微小偏差时，池化仍会返回相同的结果。因此，池化对输入数据的微小偏差具有健壮性。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683427746798-3d3c3bf2-f339-4089-a504-f0ade22fe61a.png#averageHue=%23444444&clientId=u80612cfc-fbd1-4&from=paste&height=197&id=u48356d44&originHeight=246&originWidth=866&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=31406&status=done&style=none&taskId=ua03e216e-e9e3-4db3-a137-c8952d0e8b7&title=&width=692.8)

## 7.4 卷积层和池化层的实现
### 7.4.1 4维数组
> 所谓4维数据，比如数据的形状是(10, 1, 28, 28)，则它对应10个高为28、长为28、通道为1的数据。

访问第1个和第2个数据：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683428888630-375f0ccd-eda3-447f-a9c6-8ecf69b719e0.png#averageHue=%23414141&clientId=u80612cfc-fbd1-4&from=paste&height=56&id=u6c828c07&originHeight=70&originWidth=327&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=11215&status=done&style=none&taskId=u88e4550c-1a47-43b6-9267-47066ea758a&title=&width=261.6)<br />访问第1个数据的第1个通道的空间数据：![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683428911281-02b031bf-286d-4ed5-af9e-10f690aebc70.png#averageHue=%23414141&clientId=u80612cfc-fbd1-4&from=paste&height=32&id=u9e265461&originHeight=40&originWidth=291&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=5083&status=done&style=none&taskId=uc3599567-402c-4f27-8109-2eb75e89042&title=&width=232.8)

### 7.4.2 基于im2col的展开
im2col是一个函数，将输入数据展开以适合滤波器（权重）。如下图所示，对3维的输入数据应用im2col后，数据转换为2维矩阵（正确地讲，是把包含批数量的4维数据转换成了2维数据）。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683429037171-c6016ce5-1d31-40b8-8ee0-5b4816f84412.png#averageHue=%23494949&clientId=u80612cfc-fbd1-4&from=paste&height=254&id=u36323be1&originHeight=318&originWidth=861&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=11584&status=done&style=none&taskId=u3409c8fe-c0c8-47a8-8154-1755ff926f9&title=&width=688.8)

im2col会把输入数据展开以适合滤波器（权重）。具体地说，如下图所示，对于输入数据，将应用滤波器的区域（3维方块）横向展开为1列。im2col会在所有应用滤波器的地方进行这个展开处理。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683429124083-8d0f331d-c5dd-4e0d-aa37-0c1c276b50cd.png#averageHue=%234a4a4a&clientId=u80612cfc-fbd1-4&from=paste&height=276&id=uf180f385&originHeight=345&originWidth=861&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=19335&status=done&style=none&taskId=u83e52a2c-2af1-419a-af3a-634a6a22fa3&title=&width=688.8)

使用im2col展开输入数据后，之后就只需将卷积层的滤波器（权重）纵向展开为1列，并计算2个矩阵的乘积即可。这和全连接层的Affine层进行的处理基本相同。如下图所示，基于im2col方式的输出结果是2维矩阵。因为CNN中数据会保存为4维数组，所以要将2维输出数据转换为合适的形状。以上就是卷积层的实现流程。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683429276884-1aed5249-380c-4676-a278-915c001f61bd.png#averageHue=%23434343&clientId=u80612cfc-fbd1-4&from=paste&height=412&id=u037e8b20&originHeight=515&originWidth=861&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=39876&status=done&style=none&taskId=u653f5f8d-80cb-4ded-97b0-9a5b67f3d93&title=&width=688.8)

### 7.4.3 卷积层的实现
im2col这一便捷函数具有以下接口。<br />`im2col (input_data, filter_h, filter_w, stride=1, pad=0)`

- input_data：由（数据量，通道，高，长）的4维数组构成的输入数据
- filter_h：滤波器的高
- filter_w：滤波器的长
- stride：步幅
- pad：填充

im2col会考虑滤波器大小、步幅、填充，将输入数据展开为2维数组。现在，我们来实际使用一下这个im2col。
```python
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
```
上面的两个例子，第一个的批大小为1，通道为3的 7 × 7 的数据， 第二个的批大小为10，数据形状和第一个相同。分别对其应用im2col函数，在这两种情形下，第2维的元素个数均为75。这是滤波器（通道为3、大小为5 × 5）的元素个数的总和。批大小为1时，im2col的结果是(9, 75)。而第2个例子中批大小为10，所以保存了10倍的数据，即(90, 75)。

现在使用im2col来实现卷积层。这里我们将卷积层实现为名为Convolution的类。
```python
import sys
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction") # 为了导入父目录的文件而进行的设定
from common.util import im2col
import numpy as np

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T # 滤波器的展开
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
```
卷积层的初始化方法将滤波器（权重）、偏置、步幅、填充作为参数接收。滤波器是(FN, C, FH, FW)的 4 维形状。另外，FN、C、FH、FW分别是 Filter Number（滤波器数量）、Channel、Filter Height、Filter Width的缩写。<br />展开滤波器的部分将各个滤波器的方块纵向展开为 1 列。这里通过`reshape(FN,-1)`将参数指定为-1，这是reshape的一个便利的功能。通过在reshape时指定为-1，reshape函数会自动计算-1维度上的元素个数，以使多维数组的元素个数前后一致。比如，(10, 3, 5, 5)形状的数组的元素个数共有 750个，指定reshape(10,-1)后，就会转换成(10, 75)形状的数组。

forward的实现中，最后会将输出大小转换为合适的形状。转换时使用了NumPy的transpose函数。transpose会更改多维数组的轴的顺序。如下图所示，通过指定从0开始的索引（编号）序列，就可以更改轴的顺序。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683447270181-9a013492-e5b7-4100-bef8-6e745a43789f.png#averageHue=%23414141&clientId=u80612cfc-fbd1-4&from=paste&height=203&id=ud6548757&originHeight=254&originWidth=865&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24133&status=done&style=none&taskId=udadcdde7-bcdc-4343-a62c-90e187f4135&title=&width=692)

### 7.4.4 池化层的实现
池化层的实现和卷积层相同，都使用im2col展开输入数据。不过，池化的情况下，在通道方向上是独立的，这一点和卷积层不同。具体地讲，如下图所示，池化的应用区域按通道单独展开。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683531887254-164cd502-f2cf-4f30-a587-fbbd533a29fd.png#averageHue=%23434242&clientId=u3b1ff29d-14ec-4&from=paste&height=510&id=ue69c9337&originHeight=638&originWidth=866&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=59490&status=done&style=none&taskId=u33ae99c7-2bd8-4f77-b9f2-47c44870ffa&title=&width=692.8)<br />像这样展开之后，只需对展开的矩阵求各行的最大值，并转换为合适的形状即可<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683531981252-3a9b85b0-4158-4ee4-8618-4e72a9783c74.png#averageHue=%23434343&clientId=u3b1ff29d-14ec-4&from=paste&height=343&id=u82e5062c&originHeight=429&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=55210&status=done&style=none&taskId=ubfac7361-6ab1-43a2-aa68-95c756c9ae7&title=&width=691.2)

池化层的实现按下面3个阶段进行。

1. 展开输入数据
2. 求各行的最大值
3. 转换为合适的输出大小

下面用Python来实现一下。
```python
import sys
sys.path.append("D:\Download\ProgrammingTools\VSCode\CodeWorkSpace\deep-learning-introduction") # 为了导入父目录的文件而进行的设定
from common.util import im2col
import numpy as np


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad


    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 最大值
        out = np.max(col, axis=1)

        # 转换
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)


        return out
```

## 7.5 CNN的实现
将要实现的CNN如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683534474402-d2df9af8-4906-499c-b387-ba691d96aeeb.png#averageHue=%23424242&clientId=u3b1ff29d-14ec-4&from=paste&height=225&id=u412826a4&originHeight=281&originWidth=863&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12810&status=done&style=none&taskId=ue1a69b57-eca0-45de-bc13-fa8dbd02990&title=&width=690.4)<br />如上所示，网络的构成是“Convolution - ReLU - Pooling -Affine - ReLU - Affine - Softmax”，我们将它实现为名为SimpleConvNet的类。首先来看一下SimpleConvNet的初始化（__init__），取下面这些参数。

- input_dim: 输入数据的的维度（通道、高、长）
- conv_param: 卷积层的超参数（字典）。字典关键字如下：
   - filter_num: 滤波器数量
   - filter_size: 滤波器大小
   - stride: 步幅
   - pad: 填充
- hidden_size: 隐藏层（全连接）的神经元数量
- output_size: 输出层（全连接）的神经元数量
- weightt_int_std: 初始化时权重的标准差

SimpleConvNet初始化的实现分为下面三部分说明。

1. 初始化的最开始部分
```python
class SimpleConvNet:
    def __init__(self,input_dim=(1,28,28),
                 conv_param={'filter_num':30, 'filter_size': 5,
                             'pad': 0, 'stride': 1},
                hidden_size=100,output_size=10,weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
```
这里将由初始化参数传入的卷积层的超参数从字典中取了出来（以方便后面使用），然后，计算卷积层的输出大小。

2. 权重参数的初始化部分
```python
self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0],filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
```
学习所需的参数是第1层的卷积层和剩余两个全连接层的权重和偏置。将这些参数保存在实例变量的params字典中。将第1层的卷积层的权重设为关键字W1，偏置设为关键字b1。同样，分别用关键字W2、b2和关键字W3、b3来保存第2个和第3个全连接层的权重和偏置。

3. 生成必要的层
```python
self.layers = OrderedDict()
self.layers['Conv1'] = Convolution(self.params['W1'],
                           self.params['b1'],
                           conv_param['stride'],
                           conv_param['pad'])
self.layers['Relu1'] = Relu()
self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
self.layers['Affine1'] = Affine(self.params['W2'],
                            self.params['b2'])
self.layers['Relu2'] = Relu()
self.layers['Affine2'] = Affine(self.params['W3'],
                            self.params['b3'])
self.last_layer = SoftmaxWithLoss()
```
从最前面开始按顺序向有序字典（OrderedDict）的layers中添加层。只有最后的SoftmaxWithLoss层被添加到别的变量lastLayer中。

以上就是SimpleConvNet的初始化中进行的处理。像这样初始化后，进行推理的predict方法和求损失函数值的loss方法就可以像下面这样实现。
```python
def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """求损失函数
        参数x是输入数据、t是教师标签
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)
```
	这里，参数x是输入数据，t是教师标签。用于推理的predict方法从头开始依次调用已添加的层，并将结果传递给下一层。在求损失函数的loss方法中，除了使用predict方法进行的forward处理之外，还会继续进行forward处理，直到到达最后的SoftmaxWithLoss层。

接下来是基于误差反向传播法求梯度的代码实现。
```python
def gradient(self, x, t):
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
```
	参数的梯度通过误差反向传播法（反向传播）求出，通过把正向传播和反向传播组装在一起来完成。

## 7.6 CNN的可视化
### 7.6.1 第1层权重的可视化
现在，我们将卷积层（第 1层）的滤波器显示为图像。这里，我们来比较一下学习前和学习后的权重，结果如下图所示。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683600612630-9f6fc965-550e-474d-96dd-012d840230fa.png#averageHue=%23aeaeae&clientId=u08175942-5bc2-4&from=paste&height=300&id=u10179241&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12380&status=done&style=none&taskId=u8333e9b8-c48a-4a66-b0e6-c91eaba1086&title=&width=400)![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683600620071-65a0ceac-6f5d-4709-8028-fcd7d8077fef.png#averageHue=%23a0a0a0&clientId=u08175942-5bc2-4&from=paste&height=300&id=u7c263bfb&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=11888&status=done&style=none&taskId=u89c6e4d5-972d-4d93-975a-cb6d1928116&title=&width=400)

上图中，学习前的滤波器是随机进行初始化的，所以在黑白的浓淡上没有规律可循，但学习后的滤波器变成了有规律的图像。我们发现，通过学习，滤波器被更新成了有规律的滤波器，比如从白到黑渐变的滤波器、含有块状区域（称为blob）的滤波器等。<br />右边有规律的滤波器在观察边缘（颜色变化的分界线）和斑块（局部的块状区域）等。由此可知，卷积层的滤波器会提取边缘或斑块等原始信息。

### 7.6.2 基于分层结构的信息提取
下图展示了进行一般物体识别（车或狗等）的8层CNN。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683600837729-31d4addd-91d3-4d24-9c9f-6d0ed9ee2063.png#averageHue=%23515151&clientId=u08175942-5bc2-4&from=paste&height=323&id=u0ab33f7a&originHeight=404&originWidth=861&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=218783&status=done&style=none&taskId=u38e3bfd7-0305-4202-9f8c-fe87cc50fc0&title=&width=688.8)<br />AlexNet网络结构堆叠了多层卷积层和池化层，最后经过全连接层输出结果。上图的方块表示的是中间数据，对于这些中间数据，会连续应用卷积运算。<br />上图是CNN的卷积层中提取的信息。第1层的神经元对边缘或斑块有响应，第3层对纹理有响应，第5层对物体部件有响应，最后的全连接层对物体的类别（狗或车）有响应

如果堆叠了多层卷积层，则随着层次加深，提取的信息也愈加复杂、抽象。最开始的层对简单的边缘有响应，接下来的层对纹理有响应，再后面的层对更加复杂的物体部件有响应。也就是说，随着层次加深，神经元从简单的形状向“高级”信息变化。换句话说，就像我们理解东西的“含义”一样，响应的对象在逐渐变化。

## 7.7 具有代表性的CNN
### 7.7.1 LeNet
> LeNet进行手写数字识别的网络。它有连续的卷积层和池化层（正确地讲，是只“抽选元素”的子采样层），最后经全连接层输出结果。

与现在的CNN的不同之处

1. LeNet 中使用sigmoid 函数，而现在的 CNN中主要使用 ReLU 函数。
2. 原始的LeNet中使用子采样（subsampling）缩小中间数据的大小，而现在的CNN中Max池化是主流。

### 7.7.2 AlexNet
AlexNet网络结构如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1683601154839-1f9f440a-4d0e-4ec6-87da-bf1aa1f89022.png#averageHue=%23424242&clientId=u08175942-5bc2-4&from=paste&height=266&id=u6dd08628&originHeight=332&originWidth=864&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=53337&status=done&style=none&taskId=u970594c9-d574-4267-96ce-f6dc1a2e152&title=&width=691.2)<br />AlexNet叠有多个卷积层和池化层，最后经由全连接层输出结果。

AlexNet与LeNet的差异如下。

1. 激活函数使用ReLU。
2. 使用进行局部正规化的LRN（Local Response Normalization）层。
3. 使用Dropout。

