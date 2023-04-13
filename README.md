[toc]
# 1 Python知识预备
## 1.1 Python的安装
将使用的编程语言与库。

1. Python 3.x
2. NumPy（用于数值计算）
3. Matplotlib（将实验结果可视化）

Anaconda发行版。
[Anaconda 环境配置](https://www.yuque.com/abiny/wikclb/xp0imzu4wxouo4ag?view=doc_embed)

## 1.2 Python解释器
检查Python版本。打开终端，输入命令`python --version`，该命令会输出已安装的Python的版本信息
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681301360773-629be852-18e1-477b-844b-d42eba2dd227.png#averageHue=%23191919&clientId=u63205d68-842a-4&from=paste&height=132&id=u5e8dd190&name=image.png&originHeight=165&originWidth=535&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13408&status=done&style=none&taskId=uc9962723-718a-4fe5-861b-930e308b9de&title=&width=428)

输入命令`python`即可启动Python解释器
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681371465530-27d84266-ec22-4300-bf9e-d3c280031f56.png#averageHue=%23181818&clientId=u63116624-3e14-4&from=paste&height=168&id=ub0056b3e&name=image.png&originHeight=210&originWidth=1207&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20155&status=done&style=none&taskId=u7ec7fc0d-05d9-4c1e-ad31-ca573c7879d&title=&width=965.6)
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
这里有一个特殊的`__init__`方法，这是进行初始化的方法，也称为构造函数（constructor），只在生成类的实例时被调用一次。此外，在方法的第一个参数中明确地写入表示自身（自身的实例）的`self`是Python的一个特点。
具体代码实例：
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
从终端运行`man.py`
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681373379728-3ee6f331-4d85-4cb4-809c-f358c3cf9a5b.png#averageHue=%23191919&clientId=ubed99078-1978-4&from=paste&height=184&id=u0d72e8ff&name=image.png&originHeight=230&originWidth=1373&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=25821&status=done&style=none&taskId=uf6eccc05-0df9-4fb8-b313-d4a887f3a97&title=&width=1098.4)
这里我们定义了一个新类`Man`。上面的例子中，类Man生成了实例（对象）`m`。类Man的构造函数（初始化方法）会接收参数name，然后用这个参数初始化实例变量`self.name`。实例变量是存储在各个实例中的变量。Python 中可以像 self.name 这样，通过在 self 后面添加属性名来生成或访问实例变量。

## 1.4 NumPy
### 1.4.1 导入NumPy
`import numpy as np`
Python中使用`import`语句来导入库，这里的import numpy as np，直译的话就是“将numpy作为np导入”的意思。

### 1.4.2 NumPy数组
要生成NumPy数组，需要使用`np.array()`方法。`np.array()`接收Python列表作为参数，生成NumPy数组（numpy.ndarray）。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681374024600-e5872ac5-3042-4662-a2eb-36d5507395ce.png#averageHue=%23181818&clientId=ubed99078-1978-4&from=paste&height=134&id=u657cb9b2&name=image.png&originHeight=168&originWidth=412&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12039&status=done&style=none&taskId=uc6aa6201-c093-4347-bfa2-02f965d5969&title=&width=329.6)

下面是NumPy数组的算术运算的例子

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681374620471-07ec64ba-5417-4554-80fc-c0a47b282375.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=307&id=ua6b08fab&name=image.png&originHeight=384&originWidth=491&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=27767&status=done&style=none&taskId=u5d3e7b50-cfa2-4a05-a2a7-5a01f09c7dc&title=&width=392.8)

这里需要注意的是，数组x和数组y的元素个数是相同的（两者均是元素个数为3的一维数组）。当x和y的元素个数相同时，可以对各个元素进行算术运算。如果元素个数不同，程序就会报错，所以元素个数保持一致非常重要。
NumPy数组与单一数值组合起来进行运算，需要在NumPy数组的各个元素和标量之间进行运算。这个功能也被称为“广播”。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681374771976-5fea14fa-504d-433f-9340-179615eb98e7.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=65&id=u0927b63c&name=image.png&originHeight=81&originWidth=306&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=3835&status=done&style=none&taskId=u1066fda5-4862-491d-9dba-576cad53257&title=&width=244.8)

### 1.4.3 NumPy的N维数组
NumPy不仅可以生成一维数组（排成一列的数组），也可以生成多维数组。比如，可以生成如下的二维数组（矩阵）。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375015403-36f28ea5-0736-499d-b375-fb912683cb47.png#averageHue=%23151515&clientId=ubed99078-1978-4&from=paste&height=174&id=u6c5f5567&name=image.png&originHeight=217&originWidth=417&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=12102&status=done&style=none&taskId=ueba25889-9dde-466b-88b4-a6f62ea1ffe&title=&width=333.6)

这里生成了一个2 × 2的矩阵A。另外，矩阵A的形状可以通过shape查看，矩阵元素的数据类型可以通过dtype查看。下面则是矩阵的算术运算

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375166295-b60edf62-f418-4c62-b0dc-a52163afba0f.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=158&id=ue8e8b997&name=image.png&originHeight=198&originWidth=390&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=10618&status=done&style=none&taskId=u890432bc-1633-4231-bc54-c0c6c793f74&title=&width=312)

和数组的算术运算一样，矩阵的算术运算也可以在相同形状的矩阵间以对应元素的方式进行。并且，也可以通过标量（单一数值）对矩阵进行算术运算。这也是基于广播的功能。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375310039-18d8e7dc-2729-418c-90fa-fb832448714c.png#averageHue=%23181818&clientId=ubed99078-1978-4&from=paste&height=139&id=u3c252100&name=image.png&originHeight=174&originWidth=250&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=6559&status=done&style=none&taskId=ua283a115-47df-43c5-9da9-866681103ee&title=&width=200)

### 1.4.4 广播
> NumPy中，形状不同的数组之间也可以进行运算。之前的例子中，在2×2的矩阵A和标量10之间进行了乘法运算。

广播的实例

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375493160-ed6e6cf1-d24e-4a64-b9c5-ae0745edb32f.png#averageHue=%23161616&clientId=ubed99078-1978-4&from=paste&height=159&id=ud39f1815&name=image.png&originHeight=199&originWidth=350&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=9843&status=done&style=none&taskId=ub78ed560-4634-428b-bcbb-c74712c1851&title=&width=280)

在此运算中，一维数组B被“巧妙地”变成了和二位数组A相同的形状，然后再以对应元素的方式进行运算。综上，因为NumPy有广播功能，所以不同形状的数组之间也可以顺利地进行运算。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375562896-42e8b3fb-e43f-4ae8-b1af-0cd701b4c249.png#averageHue=%23414141&clientId=ubed99078-1978-4&from=paste&height=165&id=u21a9d845&name=image.png&originHeight=206&originWidth=1127&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=24647&status=done&style=none&taskId=u7c400a42-15b9-47c6-89a9-f6d60d83ef9&title=&width=901.6)

### 1.4.5 访问元素
元素的索引从0开始。对各个元素的访问可按如下方式进行。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375748313-095cc8b4-682e-4b29-abc5-3826d00246e8.png#averageHue=%23131313&clientId=ubed99078-1978-4&from=paste&height=197&id=u8400083c&name=image.png&originHeight=246&originWidth=583&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=13215&status=done&style=none&taskId=u4a0178db-cdbb-4a61-abe3-fcc0050f749&title=&width=466.4)

也可以使用`for`语句访问各个元素

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375823856-f20f51d2-a5b5-429d-afda-386525c7d60d.png#averageHue=%23131313&clientId=ubed99078-1978-4&from=paste&height=141&id=udc255b20&name=image.png&originHeight=176&originWidth=314&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=5267&status=done&style=none&taskId=u6fef30ca-7a7f-4600-835c-872dbf6d085&title=&width=251.2)

NumPy还可以使用数组访问各个元素。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375945066-25657b72-77f9-403d-ba25-29ae844905d8.png#averageHue=%23181818&clientId=ubed99078-1978-4&from=paste&height=120&id=u29252f3f&name=image.png&originHeight=150&originWidth=636&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14919&status=done&style=none&taskId=u7e197bb1-173c-4575-918f-934b2e76d62&title=&width=508.8)

运用这个标记法，可以获取满足一定条件的元素。例如，要从X中抽出大于15的元素，可以写成如下形式。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681375993391-ae902c46-2e9d-49e7-b9a7-fc401218be99.png#averageHue=%23151515&clientId=ubed99078-1978-4&from=paste&height=100&id=uc256161c&name=image.png&originHeight=125&originWidth=609&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=7257&status=done&style=none&taskId=u14637be8-95c0-4ed4-b8c8-5f0b0bdd076&title=&width=487.2)

对NumPy数组使用不等号运算符等（上例中是X > 15）,结果会得到一个布尔型的数组。

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
这里使用NumPy的arange方法生成了[0, 0.1, 0.2, ..., 5.8, 5.9]的数据，将其设为x。对x的各个元素，应用NumPy的sin函数`np.sin()`，将x、y的数据传给`plt.plot()`方法，然后绘制图形。最后，通过`plt.show()`显示图形。运行上述代码后，就会显示如图所示的图形。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681376893281-47d41de9-437f-46e0-930a-130f5731a341.png#averageHue=%23fcfcfc&clientId=ubed99078-1978-4&from=paste&height=480&id=u9352eb6c&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=30810&status=done&style=none&taskId=u5d403bbc-1237-4703-acbb-19293443517&title=&width=640)

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
结果如图所示，图的标题、轴的标签名都被标出来了。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681376866898-0bf5e266-bbee-4cbe-bcea-9b5e57e75f4d.png#averageHue=%23fcfbfb&clientId=ubed99078-1978-4&from=paste&height=450&id=u8466cb98&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=46740&status=done&style=none&taskId=u07da9fda-3049-440e-b6ff-348a1cf2d2c&title=&width=600)

### 1.5.3 显示图像
pyplot中还提供了用于显示图像的方法imshow()。另外，可以使用matplotlib.image模块的imread()方法读入图像。代码实例如下。
```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('dataset/dog.jpg') # 读入图像
plt.imshow(img)

plt.show()
```
运行上述代码后，会显示如下图像
![image.png](https://cdn.nlark.com/yuque/0/2023/png/25941432/1681377913536-0e6c03eb-4d8f-4ce1-a4af-b1bd0c92fd1d.png#averageHue=%23dbc4b1&clientId=ubed99078-1978-4&from=paste&height=480&id=u0518a560&name=image.png&originHeight=600&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=335581&status=done&style=none&taskId=u0de96ec0-6444-48b4-966c-e3c23bc4472&title=&width=640)
因为我的Python解释器运行在根目录下，且图片在dataset下，故图片路径为'dataset/dog.jpg'。




























