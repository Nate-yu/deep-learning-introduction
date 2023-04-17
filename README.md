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

和实际的电流不同的是，感知机的信号只有“流/不流”（1/0）两种取值。0对应“不传递信号”，1对应“传递信号”。如下图所示是一个接收两个输入信号的感知机的例子。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681440819573-d2627755-1569-47c2-9fe4-df8fea50fd57.png#averageHue=%23414141&clientId=u3972c055-bc56-4&from=paste&height=419&id=u6bf328c9&name=image.png&originHeight=524&originWidth=1493&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=35135&status=done&style=none&taskId=u9214393e-87a7-4342-ba07-825d3de6f98&title=&width=1194.4)<br />其中，x1、x2是输入信号，y 是输出信号，w1、w2是权重。图中的圆称为“神经元”或者“节点”。输入信号被送往神经元时，会被分别乘以固定的权重（w1x1、w2x2）。神经元会计算传送过来的信号的总和，只有当这个总和超过了某个界限值时，才会输出1。这也称为“神经元被激活” 。这里将这个界限值称为阈值，用符号θ表示。<br />上述即为感知机的运行原理，用数学公式表示即为如下：<br />$y= \begin{cases}0 \quad (w_1x_1 + w_2x_2\le\theta)\\ 1\quad (w_1x_1+w_2x_2>\theta)\end{cases}$<br />感知机的多个输入信号都有各自固有的权重，这些权重发挥着控制各个信号的重要性的作用。即权重越大，对应该权重的信号的重要性就越高。

## 2.2 简单逻辑电路
> 与门：与门是有两个输入和一个输出的门电路。

下表这种输入信号和输出信号的对应表称为“真值表”。与门仅在两个输入均为1输出1，其他时候则输出0。

| $x_1$ | $x_2$ | $y$ |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |


> 与非门：与非门就是颠倒了与门的输出。

用真值表表示的话，如下表所示，仅当x1和x2同时为1时输出0，其他时候则输出1。

| $x_1$ | $x_2$ | $y$ |
| --- | --- | --- |
| 0 | 0 | 1 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |


> 或门：或门是“只要有一个输入信号是1，输出就为1”的逻辑电路。

| $x_1$ | $x_2$ | $y$ |
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

将 θ 换成 -b ，改写数学公式：<br />$y= \begin{cases}0 \quad (b+w_1x_1 + w_2x_2\le0)\\ 1\quad (b+w_1x_1+w_2x_2>0)\end{cases}$,此处，b称为**偏置**，w1和w2称为**权重**。<br />感知机会计算输入信号和权重的乘积，然后加上偏置，如果这个值大于0则输出1，否则输出0。使用NumPy逐一确认结果。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681458245530-4e527aac-d5b0-4ef4-815b-04498c6f7ac9.png#averageHue=%231b1b1b&clientId=u21cf038b-070e-4&from=paste&height=230&id=u32b8a377&name=image.png&originHeight=287&originWidth=323&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14235&status=done&style=none&taskId=u4efd5878-b72b-4457-9aca-bfc566e3d57&title=&width=258.4)<br />在NumPy数组的乘法运算中，当两个数组的元素个数相同时，各个元素分别相乘，因此`w*x`的结果就是它们的各个元素分别相乘（[0, 1] * [0.5, 0.5] => [0, 0.5]）。之后，`np.sum(w*x)`再计算相乘后的各个元素的总和。最后再把偏置加到这个加权总和上。

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

| $x_1$ | $x_2$ | $y$ |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

用前面的感知机是无法实现异或门的。感知机的局限性就在于它只能表示一条直线分割的空间。

## 2.5 多层感知机
通过已有门电路的组合：<br />异或门的制作方法有很多，其中之一就是组合我们前面做好的与门、与非门、或门进行配置。与门，与非门，或门用如下图的符号表示<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681472867117-0a56e39e-d4f8-4f59-8e22-04e15c093b6a.png#averageHue=%23414141&clientId=u21cf038b-070e-4&from=paste&height=167&id=ue80250b0&name=image.png&originHeight=209&originWidth=906&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=16198&status=done&style=none&taskId=ud6112002-d08f-405b-ae78-95dc944dd28&title=&width=724.8)<br />通过组合感知机（叠加层）就可以实现异或门。异或门可以通过如下所示配置来实现，这里，x1和x2表示输入信号，y表示输出信号，x1和x2是与非门和或门的输入，而与非门和或门的输出则是与门的输入。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681473004376-abd7ade6-d849-4a66-81fb-5bc2fc6adc8c.png#averageHue=%23414141&clientId=u21cf038b-070e-4&from=paste&height=193&id=u01e5cafb&name=image.png&originHeight=241&originWidth=701&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20588&status=done&style=none&taskId=u605abcec-265e-4d61-982f-8d0f5264084&title=&width=560.8)<br />验证正确性，把s1作为与非门的输出，把s2作为或门的输出，填入真值表

| $x_1$ | $x_2$ | $s_1$ | $s_2$ | $y$ |
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

简化感知机数学式：$y=h(b+w_1x_1+w_2x_2)$，我们用一个函数来表示这种分情况的动作（超过0则输出1，否则输出0）。<br />$h(x) = \begin{cases} 0 \quad (x \le 0) \\ 1 \quad (x>0)\end{cases}$<br />输入信号的总和会被函数h(x)转换，转换后的值就是输出y。h（x）函数会将输入信号的总和转换为输出信号，这种函数一般称为激活函数（activation function）。其作用在于决定如何来激活输入信号的总和。<br />进一步来改进上式：<br />$(1) \quad a = b + w_1x_1 + w_2x_2$<br />$(2) \quad y = h(x)$<br />首先，式（1）计算加权输入信号和偏置的总和，记为a。然后，式（2）用h()函数将a转换为输出y。下图为明确显示激活函数的计算过程。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681562541410-54919415-68b7-42d2-9f18-249e7b918385.png#averageHue=%23424242&clientId=u664bfb51-2e30-4&from=paste&height=307&id=u6f8e6ecf&name=image.png&originHeight=613&originWidth=689&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=43209&status=done&style=none&taskId=u07b64979-c9bc-4e91-bfb8-0c8817a31cd&title=&width=345)信号的加权总和为节点a，然后节点a被激活函数h()转换成节点y。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681562849199-b81a45ea-bca0-42c8-b842-b840280fb2f5.png#averageHue=%23414141&clientId=u664bfb51-2e30-4&from=paste&height=238&id=u0c9793c0&name=image.png&originHeight=298&originWidth=1316&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=38865&status=done&style=none&taskId=uc31b44f3-42b7-4582-8c50-cf0603cb52f&title=&width=1052.8)<br />左图是一般的神经元的图，右图是在神经元内部明确显示激活函数的计算过程的图（a表示输入信号的总和，h()表示激活函数，y表示输出）

> “朴素感知机”是指单层网络，指的是激活函数使用了阶跃函数 A 的模型。“多层感知机”是指神经网络，即使用 sigmoid 函数等平滑的激活函数的多层网络。

## 3.2 激活函数
> 阶跃函数：以阈值为界，一旦输入超过阈值，就切换输出。这样的函数称为“阶跃函数”

因此，可以说感知机中使用了阶跃函数作为激活函数。也就是说，在激活函数的众多候选函数中，感知机使用了阶跃函数。
### 3.2.1 sigmoid函数
神经网络中经常使用的一个激活函数就是sigmoid函数：$h(x) = \frac{1}{1+e^{-x}}$。神经网络中用sigmoid函数作为激活函数，进行信号的转换，转换后的信号被传送给下一个神经元。神经元的多层<br />连接的构造、信号的传递方法等，基本上和感知机是一样的。

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

ReLU函数可以标识为下面的式子。<br />$h(x) = \begin{cases} x \quad (x > 0) \\ 0 \quad (x \le 0) \end{cases}$<br />ReLU函数可用如下代码实现。
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
权重符号如下<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681719977951-16438890-0340-4ff1-808d-51617a665b8b.png#averageHue=%23414141&clientId=u05e7a439-d16b-4&from=paste&height=307&id=ufa409226&name=image.png&originHeight=384&originWidth=830&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=50531&status=done&style=none&taskId=u1428a267-6e8d-48c9-bb10-9e9ae8a31a6&title=&width=664)<br />权重和隐藏层的神经元的右上角有一个“(1)”，表示权重和神经元的层号（即第1层的权重、第1层的神经元）。此外，权重的右下角有两个数字，它们是后一层的神经元和前一层的神经元的索引号。$w_{12}^{(1)}$表示前一层的第2个神经元$x_2$到后一层的第1个神经元$a_1^{(1)}$的权重。权重右下角按照“后一层的索引号、前一层的索引号”的顺序排列。

### 3.4.2 各层间信号传递的实现
下面是输入层到第1层的第一个神经元的信号传递过程<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681720475570-d9853296-24ca-46af-bc82-72792b2d10f2.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=419&id=ue9118699&name=image.png&originHeight=524&originWidth=860&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=90058&status=done&style=none&taskId=u30f68141-080c-407c-8406-fa898e751bb&title=&width=688)<br />上图中增加了表示偏置的神经元“1”。偏置的右下角的索引号只有一个，因为前一层的偏置神经元只有一个。<br />下面用数学式表示$a_1^{(1)}$通过加权信号和偏置的和按如下方式进行计算：$a_1^{(1)} = w_{11}^{1}x_1 + w_{12}^{(1)}x_2 + b_1^{(1)}$。此外，如果使用矩阵的乘法运算，则可以将第1层的加权表示成下面的式子：$\bm {A}^{(1)} = \bm{XW}^{(1)} + \bm{B}^{(1)}$，其中各元素如下所示<br />$\bm{A}^{(1)} = \begin{pmatrix}a_1^{(1)} & a_2^{(1)} & a_3^{(1)}\end{pmatrix}，\bm{X} = \begin{pmatrix} x_1 & x_2 \end{pmatrix}，\bm{B}^{(1)}=\begin{pmatrix} b_1^{(1)} & b_2^{(1)} & b_3^{(1)} \end{pmatrix}，W^{(1)} = \begin{pmatrix} w_{11}^{(1)} &w_{21}^{(1)} & w_{31}^{(1)} \\ w_{12}^{(1)} &w_{22}^{(1)} & w_{32}^{(1)}\end{pmatrix}$

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

分类问题中使用的softmax函数可以用下面的数学式表示。<br />$y_k = \frac{e^{a_k}}{\sum_{i=1}^ne^{a_i}}$<br />这个式子表示假设输出层共有n个神经元，计算第k个神经元的输出$y_k$。分子是输入信号$a_k$的指数函数，分母是所有输入信号的指数函数的和。softmax函数的图示如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681736253756-5d6e7f44-0786-4110-8295-74d023641600.png#averageHue=%23424242&clientId=u05e7a439-d16b-4&from=paste&height=225&id=u1bb2d82b&name=image.png&originHeight=281&originWidth=295&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24835&status=done&style=none&taskId=u8af574ca-7cec-4c2f-8a56-8e5df98d5fb&title=&width=236)<br />softmax函数的输出通过箭头与所有的输入信号相连。从上面的数学式可以看出，输出层的各个神经元都受到所有输入信号的影响。

用Python解释器实现softmax函数如下。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681736653152-203f8efc-7a75-4824-9c75-2b02a03a99e0.png#averageHue=%23373b43&clientId=u05e7a439-d16b-4&from=paste&height=236&id=u841780ba&name=image.png&originHeight=295&originWidth=464&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27633&status=done&style=none&taskId=ub3278667-81bf-4c75-a14f-15eacb9af39&title=&width=371.2)<br />将其封装为`softmax()`函数。
```python
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
	return y
```

### 3.5.2 实现softmax函数时的注意事项
为了防止指数计算时的溢出，softmax函数的实现可以如下改进。<br />$y_k = \frac{e^{a_k}}{\sum_{i=1}^ne^{a_i}} = \frac{Ce^{a_k}}{C\sum_{i=1}^ne^{a_i}} = \frac{e^{a_k+lnC}}{\sum_{i=1}^ne^{a_i+lnC}} = \frac{e^{a_k+C'}}{\sum_{i=1}^ne^{a_i+C'}}(其中，C'=lnC)$

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



