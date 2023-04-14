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
