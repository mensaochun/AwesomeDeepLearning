# 机器学习中常用的数据集归纳

## 1.通用数据集

### 1.1 UCI

#### 1.1.1 下载地址

​    http://archive.ics.uci.edu/ml/index.php

#### 1.1.2 简要介绍

​    UCI数据库是加州大学欧文分校(University of CaliforniaIrvine)提出的用于机器学习的数据库，这个数据库目前共有436个数据集，其数目还在不断增加，是一个常用的标准测试数据集，广泛应用于机器学习算法评估中，使用量最广泛的应该是iris鸢尾花数据集。

​    UCI数据集包括非常各行各业的数据，不仅包括简单数据集，也包括大量非常复杂、来源实际应用的数据集，所有数据集均来自全世界各地的捐赠，可以免费使用。

​    每个数据文件（.data）包含以“属性-值”对形式描述的很多个体样本的记录。对应的.info文件包含的大量的文档资料 。（有些文件\_generate_ databases，他们不包含*.data文件。）作为数据集和领域知识的补充，在utilities目录里包含了一些在使用这一数据集时的有用资料，说明非常详细。其主页截图如下：

![1](./pics/dataset/1.png)

#### 1.1.3 数据案例

​    以iris数据集为例，分析数据格式。

​    首先点击相应的链接，可以看到数据集的简要说明和相应的下载目录：

![1](./pics/dataset/2.png)

可以看出：iris是多分类数据集，一共包括4个特征，150个样本，不存在缺失值以及其他有用信息。对于下载目录：

![1](./pics/dataset/3.png)

iris.data是数据集，里面是按照键值对形式保存的数据，iris.names是数据的详细描述，每一个特征代表的含义有详细说明。取前面几组可以看出：

```
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
```

前4列是特征，最后一列是类别标签。该文件格式可以当做普通的txt文件读入。

### 1.2 MNIST

#### 1.2.1 下载地址

​    http://yann.lecun.com/exdb/mnist/

#### 1.2.2 简要介绍

​    MNIST是深度学习非常经典的学术型手写数字分类数据集，有lecun收集提供，但对于目前发展来说，已经显得非常陈旧。数据保存格式是图片的字节数据，每张手写数字是28*28的像素，灰度数据，一共有60k张训练图片和10k张测试图片，官方网站有非常详细的说明。由于其使用非常广泛，目前各大深度学习框架例如tensorflow和pytorch等都自带了该数据集，无需用户手动处理，非常方便。

#### 1.2.3 数据案例

​    MNIST数据集总共包括4个文件，如下：

![1](./pics/dataset/4.png)

​    在tensorflow中一行代码即可导入数据并使用：

```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist.train.images, mnist.train.labels
```

### 1.3 CIFAR 10 和 CIFAR 100

#### 1.3.1 下载地址

​    https://www.cs.toronto.edu/~kriz/cifar.html

#### 1.3.2 简要介绍

​    cifar-10和cifar-100是比mnist复杂，应用非常广泛的基准分类数据集。和mnist不同，其是彩色数据，格式为$32*32*3$,cifar-10是指类别一共是10类，cifar-100则相对更复杂一些，是包括100类的数据集。

![1](./pics/dataset/5.png)

cifar-10简要图示如上图所示。cifar-10一共包括60k张图片，每个类别是6k张，其中训练样本是50k张，测试样本是10k张。提供了3种格式的下载数据，如下：

![1](./pics/dataset/6.png)

​    cifar-100数据格式和cifar-10是一样的，其中一共包括100个类，每个类是600张图片，一共60k张图片，每个类包括500张训练图片和100张测试图片。100个类的分布是：总共包括20个大类，每个大类底下是5个小类，如下图所示：

![1](./pics/dataset/7.png)

下载格式和cifar-10一致。

#### 1.3.3 数据案例

​    由于cifar-10数据集应用非常广泛，故而类似的深度学习框架例如tensorflow、pytorch都可以直接调用。如果你打算自己处理，则可以按照以下方法读取：

```
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
```

由于图片过多，数据存储是分为5个batch，每个batch是10000张图片，对于其中一个batch，dict中包括data和label作为键，其中data是一个包含10000x3072的numpy array数组，每一个包含一张32*32的彩色图片，排布规则是RGB。

### 1.4 Fashion-MNIST

#### 1.4.1 下载地址

​    https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion

#### 1.4.2 简要介绍

​    Fashion-MNIST 是时尚界的mnist，包括的数据图片都是和各种衣服和鞋子有关。其说明文档是：https://github.com/zalandoresearch/fashion-mnist，该数据集的提出是为了弥补MNIST存在的如下不足：

​    (1) MNIST数据过于简单。普通的卷积神经网络就可以达到99.7%，简单的机器学习算法也可以达到97%，导致会出现模型在MNIST上面效果显著，但是在其他数据集上面性能较差；

​    (2) MNIST被过度使用了。Ian Goodfellow 等都呼吁大家不要再用MNIST；

​    (3) MNIST不能代表现代的CV任务。

​    Fashion-MNIST提出的目的就是为了无缝替代MNIST，所以其设计的图片尺寸，格式，排列方式完全和MNIST一样，也就是说对于MNIST识别代码，只需要换一下数据就可以，代码完全不用做任何修改，非常方便。

![1](./pics/dataset/fashion-mnist-sprite.png)

数据分布如下图：

![1](./pics/dataset/embedding.gif)

#### 1.4.3 数据案例

​    Fashion-MNIST数据集的类别信息如下：

| Label | Description |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

下载格式如下：

![1](./pics/dataset/8.png)

为了方便使用，作者提供了非常完善的各大框架调用api，可以直接拷贝工具类到工程中即可一行代码搞定。

### 1.5 ImageNet

#### 1.5.1 下载地址

​    http://www.image-net.org/

​    由于数据集非常大，其内部包括很多种下载方式，具体见官网

#### 1.5.2 简要说明

​    ImageNet 是一个计算机视觉系统识别项目名称， 是目前世界上图像识别最大的数据库。是美国斯坦福的计算机科学家，模拟人类的识别系统建立的。ImageNet就像一个网络一样，拥有多个Node（节点），每一个node相当于一个item或者subcategory。据官网消息，一个node（目前）含有至少500个对应物体的可供训练的图片/图像。它实际上就是一个巨大的可供图像/视觉训练的图片库。ImageNet的结构基本上是金字塔型：目录->子目录->图片集，其结构图链接为：http://image-net.org/explore.php，截图如下：

![1](./pics/dataset/9.png)

​    Imagenet数据集有1400多万幅图片，涵盖2万多个类别；其中有超过百万的图片有明确的类别标注和图像中物体位置的标注，目前关于图像分类、定位、检测等研究工作大多基于此数据集展开。与Imagenet数据集对应的有一个享誉全球的“ImageNet国际计算机视觉挑战赛(ILSVRC)”，如果想使用，建议下载ILSVRC各年份的数据集，因为非常规范和完整。谨慎下载，因为非常大，以ILSVRC2016比赛全部数据为例，数据集大小：~1TB。

#### 1.5.3 数据案例

​    imageNet提供了非常多的下载形式，如下：

![1](./pics/dataset/10.png)

 下载符合自己要求的格式即可。以下是ILSVRC2017比赛的数据简要说明：

1. [Object localization](http://image-net.org/challenges/LSVRC/2017/index#loc) for 1000 categories.
2. [Object detection](http://image-net.org/challenges/LSVRC/2017/index#det) for 200 fully labeled categories.
3. [Object detection from video](http://image-net.org/challenges/LSVRC/2017/index#vid) for 30 fully labeled categories.

## 2.图像目标检测、分割数据集

### 2.1 PASCAL VOC

#### 2.1.1 下载地址

​    http://host.robots.ox.ac.uk/pascal/VOC/

#### 2.1.2 简要介绍

​    PASCAL VOC挑战赛是视觉对象的分类识别、检测和分割的一个基准测试，提供了检测算法和学习性能的标准图像注释数据集和标准的评估系统。目前该数据集可以进行：**分类、检测、分割、动作分类、无框动作分类、人布局分类(检测出手、脚和头等部位)**。PASCAL VOC图片集包括20个目录：人类；动物（鸟、猫、牛、狗、马、羊）；交通工具（飞机、自行车、船、公共汽车、小轿车、摩托车、火车）；室内（瓶子、椅子、餐桌、盆栽植物、沙发、电视）。PASCAL VOC挑战赛在2012年后便不再举办，但其数据集图像质量好，标注完备，非常适合用来测试算法性能，VOC挑战赛年份是2005~2012,目前常用的数据集是VOC2007和VOC2012，数据集不算很大，是目前图像目标检测算法的必备数据集之一。    

#### 2.1.3 案例说明

​    以下以VOC2012数据集为例，数据大小大约2GB，说明数据格式。

​    VOC2012数据的具体下载地址是：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit ，其中有对应的数据说明pdf下载，非常详细(已经下载到本地了)。

​    VOC数据集已经分好了训练、验证和测试数据，无需我们再区分。下载后解压得到的数据文件如下：

![1](./pics/dataset/11.png)

**(1) JPEGImages文件夹**

​    存放所有图片，一共17125张图片，包括了训练图片和测试图片。这些图像都是以“年份_编号.jpg”格式命名的。图片的像素尺寸大小不一，但是横向图的尺寸大约在$500*375$左右，纵向图的尺寸大约在$375*500$左右，基本不会偏差超过100。（在实际训练中，第一步就是将这些图片都resize到$300*300$或是$500*500$，所有原始图片不能离这个标准过远）

![1](./pics/dataset/12.png)

**(2) Annotations**

​    存放每张图片的标注信息，是最重要的文件夹，以2007_000129.xml文件为例进行分析

```xml
<annotation>
  <folder>VOC2012</folder>
  <filename>2007_000129.jpg</filename>
  <source>
      <database>The VOC2007 Database</database>
      <annotation>PASCAL VOC2007</annotation>
      <image>flickr</image>
  </source>
  <size>  # 图片尺寸
      <width>334</width>
      <height>500</height>
      <depth>3</depth>
  </size>
  <segmented>1</segmented>
  <object>  # 目标，可以一张图片中有好几个目标
      <name>bicycle</name>
      <pose>Unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult> # 是否是难分目标
      <bndbox>  # 边界框标注
          <xmin>70</xmin>
          <ymin>202</ymin>
          <xmax>255</xmax>
          <ymax>500</ymax>
      </bndbox>
  </object>
  <object>
      <name>bicycle</name>
      <pose>Unspecified</pose>
      <truncated>1</truncated>
      <difficult>1</difficult>
      <bndbox>
          <xmin>251</xmin>
          <ymin>242</ymin>
          <xmax>334</xmax>
          <ymax>500</ymax>
      </bndbox>
  </object>
  <object>
      <name>bicycle</name>
      <pose>Unspecified</pose>
      <truncated>1</truncated>
      <difficult>1</difficult>
      <bndbox>
          <xmin>1</xmin>
          <ymin>144</ymin>
          <xmax>67</xmax>
          <ymax>436</ymax>
      </bndbox>
  </object>
  <object>
      <name>person</name>
      <pose>Unspecified</pose>
      <truncated>1</truncated>
      <difficult>1</difficult>
      <bndbox>
          <xmin>1</xmin>
          <ymin>1</ymin>
          <xmax>66</xmax>
          <ymax>363</ymax>
      </bndbox>
  </object>
  <object>
      <name>person</name>
      <pose>Frontal</pose>
      <truncated>1</truncated>
      <difficult>0</difficult>
      <bndbox>
          <xmin>74</xmin>
          <ymin>1</ymin>
          <xmax>272</xmax>
          <ymax>462</ymax>
      </bndbox>
  </object>
  <object>
      <name>person</name>
      <pose>Unspecified</pose>
      <truncated>1</truncated>
      <difficult>0</difficult>
      <bndbox>
          <xmin>252</xmin>
          <ymin>19</ymin>
          <xmax>334</xmax>
          <ymax>487</ymax>
      </bndbox>
  </object>
</annotation>
```
对应的图片是：

![1](./pics/dataset/13.jpg)



(3) SegmentationClass

​    存放的是所有语义分割数据集，同样，对于2007_000129.jpg所对应的语义分割图片为：

![1](./pics/dataset/14.png)

(4) SegmentationObject

​    存放所有图片对应的实例分割图片，同样，对于2007_000129.jpg所对应的实例分割图片为：

![1](./pics/dataset/15.png)

**(5) ImageSets**

​    存放每一种类型的challenge对应的图像数据说明，包括训练集和测试集的划分，是最关键的文件，文件夹下依然包括4个文件夹，如图：

![1](./pics/dataset/16.png)

Action文件夹是关于Action分类的划分说明，Layout文件夹是关于perso Layout分类的划分说明，Segmentation是关于图像分割的划分说明，Main是关于分类和目标检测的划分说明，是我们最关心的文件夹：

![1](./pics/dataset/17.png)

Main文件夹下包含了20个分类的xxx_train.txt、xxx_val.txt和xxx_trainval.txt，其大概内容是：

~~~~2011_002492 -1~~~~
2011_002458 -1
2011_002460 -1
2011_002461 -1
2011_002462 -1
2011_002464 -1
2011_002470  1
2011_002474 -1
2011_002476 -1
2011_002484 -1
2011_002488 -1
~~~~

第一列是文件名，后面是标签，代表正负样本。对于目标检测而言，我们非常关心train.txt，val.txt，其分别代表训练样本和测试样本。

​    由于VOC数据集在目标检测中是几乎必备的，所以目前出现了非常多的工具类，可以通过引入工具类代码直接读取数据、解析数据、训练数据，而不需要我们从头开始解析文件。

### 2.2 COCO

#### 2.2.1 下载地址

   MSCOCO官方网站： http://cocodataset.org/#home

   MSCOCO API：https://github.com/pdollar/coco

   细节：http://cocodataset.org/#download

#### 2.2.2 简要介绍

​    COCO是微软提供的数据集，目前COCO挑战赛依然在进行，数据集还在扩增、发展，但总体来说标注比较粗超。该数据集非常庞大，可以完成的任务也非常多，其特性为：

![1](./pics/dataset/18.png)

可以看出，已经标注的图片就包括20W张，共80种物体类别，所有的物体实例都用详细的分割mask进行了标注， 共标注了超过 150w个物体实体，25W张人的keypoints图片。 

​    COCO的全称是Common Objects in Context，是微软团队提供的一个可以用来进行图像识别的数据集。MS COCO数据集中的图像分为训练、验证和测试集。COCO通过在Flickr上搜索80个对象类别和各种场景类型来收集图像，其使用了亚马逊的Mechanical Turk（AMT）。COCO数据集现在有3种标注类型：object instances（目标实例）, object keypoints（目标上的关键点）, and image captions（看图说话），使用JSON文件存储。

#### 2.2.3 数据案例    

#### ![1](./pics/dataset/19.png)

​    以上是下载地址和对应的标注annotations文件，实际使用需要先把images和Annotations文件下载，然后使用cooc api进行编译。COCO数据集有对应的官方pdf文章说明，已经下载到本地了。

​    以下截图是下载的2017 Train/Val annotations:

![1](./pics/dataset/20.png)

可以看出：每一种类型都分为训练集和验证集，一共6个json文件。这3种类型共享下面所列的基本类型，包括info、image、license，而annotation类型则呈现出了多态

##### (1) 基本的JSON结构体类型

```json
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
}
    
info{
    "year": int,
    "version": str,
    "description": str,
    "contributor": str,
    "url": str,
    "date_created": datetime,
}
license{
    "id": int,
    "name": str,
    "url": str,
} 
image{ # 比较重要
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
    "license": int,
    "flickr_url": str,
    "coco_url": str,
    "date_captured": datetime,
}
```

1) info类型，比如一个info类型的实例:

```json
"info":{
	"description":"This is stable 1.0 version of the 2014 MS COCO dataset.",
	"url":"http:\/\/mscoco.org",
	"version":"1.0","year":2014,
	"contributor":"Microsoft COCO group",
	"date_created":"2015-01-27 09:11:52.357475"
},
```

2) Images是包含多个image实例的数组，对于一个image类型的实例：

```json
{
	"license":3,
	"file_name":"COCO_val2014_000000391895.jpg",
	"coco_url":"http:\/\/mscoco.org\/images\/391895",
	"height":360,"width":640,"date_captured":"2013-11-14 11:18:45",
	"flickr_url":"http:\/\/farm9.staticflickr.com\/8186\/8119368305_4e622c8349_z.jpg",
	"id":391895
},
```

3) licenses是包含多个license实例的数组，对于一个license类型的实例：

```json
{
	"url":"http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/",
	"id":1,
	"name":"Attribution-NonCommercial-ShareAlike License"
},
```

##### (2) Object Instance 类型的标注格式

**1) 整体JSON文件格式**

   对应的是instances_train2017.json和instances_val2017.json这两个文件。

   Object Instance这种格式的文件从头至尾按照顺序分为以下段落：

```json
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}
```

其中info、licenses、images这三个结构体/类型 在上一节中已经说了，在不同的JSON文件中这三个类型是一样的，定义是共享的。不共享的是annotation和category这两种结构体，他们在不同类型的JSON文件中是不一样的。images数组、annotations数组、categories数组的元素数量是相等的，等于图片的数量。

**2) annotations字段**

​    annotations字段是包含多个annotation实例的一个数组，annotation类型本身又包含了一系列的字段，如这个目标的
category id和segmentation mask。segmentation格式取决于这个实例是一个单个的对象（即iscrowd=0，将使用polygons格式）还是一组对象（即iscrowd=1，将使用RLE格式）。如下所示：

```json
annotation{
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}
```

注意，单个的对象（iscrowd=0)可能需要多个polygon来表示，比如这个对象在图像中被挡住了。而iscrowd=1时（将标注一组对象，比如一群人）的segmentation使用的就是RLE格式。另外，每个对象（不管是iscrowd=0还是iscrowd=1）都会有一个矩形框bbox ，矩形框左上角的坐标和矩形框的长宽会以数组的形式提供，数组第一个元素就是左上角的横坐标值。area是area of encoded masks。annotation结构中的categories字段存储的是当前对象所属的category的id，以及所属的supercategory的name。

​    下面是从instances_val2017.json文件中摘出的一个annotation的实例：

```json
{
	"segmentation": [[510.66,423.01,511.72,420.03,510.45,416.0,510.34,413.02,510.77,410.26,\
			510.77,407.5,510.34,405.16,511.51,402.83,511.41,400.49,510.24,398.16,509.39,\
			397.31,504.61,399.22,502.17,399.64,500.89,401.66,500.47,402.08,499.09,401.87,\
			495.79,401.98,490.59,401.77,488.79,401.77,485.39,398.58,483.9,397.31,481.56,\
			396.35,478.48,395.93,476.68,396.03,475.4,396.77,473.92,398.79,473.28,399.96,\
			473.49,401.87,474.56,403.47,473.07,405.59,473.39,407.71,476.68,409.41,479.23,\
			409.73,481.56,410.69,480.4,411.85,481.35,414.93,479.86,418.65,477.32,420.03,\
			476.04,422.58,479.02,422.58,480.29,423.01,483.79,419.93,486.66,416.21,490.06,\
			415.57,492.18,416.85,491.65,420.24,492.82,422.9,493.56,424.39,496.43,424.6,\
			498.02,423.01,498.13,421.31,497.07,420.03,497.07,415.15,496.33,414.51,501.1,\
			411.96,502.06,411.32,503.02,415.04,503.33,418.12,501.1,420.24,498.98,421.63,\
			500.47,424.39,505.03,423.32,506.2,421.31,507.69,419.5,506.31,423.32,510.03,\
			423.01,510.45,423.01]],# 分别是x,y坐标
	"area": 702.1057499999998,
	"iscrowd": 0,
	"image_id": 289343,
	"bbox": [473.07,395.93,38.65,28.67],
	"category_id": 18,
	"id": 1768
},
```

**3) categories字段**

categories是一个包含多个category实例的数组，而category结构体描述如下：

```json
{
    "id": int,
    "name": str,
    "supercategory": str,
}
```

从instances_val2017.json文件中摘出的2个category实例如下所示：

```json
{
	"supercategory": "person",
	"id": 1,
	"name": "person"
},
{
	"supercategory": "vehicle",
	"id": 2,
	"name": "bicycle"
},
```

##### (3) COCO API

   COCO API非常强大，不仅可以读取json数据、解析数据，而且可以直接可视化数据，防止数据有误。具体操作是：下载Images和Annotitation数据，然后github下载coco api，进入PythonAPI目录下，执行make命令，即可将c++代码编译成so文件，后续才可以正确使用，具体见github。

### 2.3 KITTI

#### 2.3.1 下载地址

​    http://www.cvlibs.net/datasets/kitti/index.php

#### 2.3.2 简要介绍

​    KITTI数据集由德国卡尔斯鲁厄理工学院和丰田美国技术研究院联合创办，是目前国际上最大的自动驾驶场景下的计算机视觉算法评测数据集。该数据集用于评测立体图像(stereo)，光流(optical flow)，视觉测距(visual odometry)，3D物体检测(object detection)和3D跟踪(tracking)等计算机视觉技术在车载环境下的性能。KITTI包含市区、乡村和高速公路等场景采集的真实图像数据，每张图像中最多达15辆车和30个行人，还有各种程度的遮挡与截断。整个数据集由389对立体图像和光流图，39.2 km视觉测距序列以及超过200k 3D标注物体的图像组成，以10Hz的频率采样及同步。总体上看，原始数据集被分类为’Road’,  ’City’,  ’Residential’, ’Campus’ 和 ’Person’。对于3D物体检测，label细分为car, van, truck, pedestrian, pedestrian(sitting), cyclist, tram以及misc组成。**简单来说是一个测试交通场景中车辆检测，车辆追踪，语义分割等算法的公开数据集，**应用非常广泛。

#### 2.3.3 数据案例

​    对于目标检测而言，可以选择2D场景和3D场景，这里以2D场景为例，具体下载地址是：http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d，对应截图如下：

![1](./pics/dataset/21.png)

一般而言，我们只需要下载第一个数据集和对应的标注文件download training labels of object data set(5M)即可，一共12GB。然后将其解压，其中7481张训练图片有标注信息，而测试图片没有，即训练图片数量为7481，由于数据过多，我暂时没有下载，无法知道内部的数据格式，但是需要注意的是：由于该数据集经常用于目标检测，也就是说经常会和VOC等数据集一起使用，故为了方便官方有提供专门的转化工具，可以将kitti的数据格式直接转化为voc的数据格式。

## 3.姿态估计数据集

### 3.1 6D-DataSet

#### 3.1.1 下载地址

​    下载地址1：http://ptak.felk.cvut.cz/6DB/public/datasets/

​    SIXD Challenge 2017地址：http://cmp.felk.cvut.cz/sixd/challenge_2017/

#### 3.1.2 简要描述

​    SIXD Challenge 2017是一个专门仅仅使用RGB或者RGBD数据进行6D姿态估计的挑战赛。为了方便比赛，组办方规定了数据的格式和提供了一系列转化工具，方便用户使用。所有的3D目标模型以及训练、测试数据都是RGBD格式的，训练数据包含了3D模型的不同视角，来源于类似于Kinect传感器和CAD渲染的3D模型，测试图片来自各种复杂的场景，具体细节在对应的数据集的dataset_info.md 中。目前该比赛一共提供了7个数据集，其中包括了6D姿态估计最广泛使用的**Hinterstoisser**和**Tejani **数据集。其截图如下：

![1](./pics/dataset/22.png)

#### 3.1.3 数据案例

​    以**Hinterstoisser**为例进行分析，目前该数据已经下载到本地。

**(1) 标准数据格式**

​    官方提供了**sixd_toolkit**工具箱来将任意数据转化为标准格式，其说明地址为：https://github.com/thodan/sixd_toolkit/blob/master/doc/sixd_2017_datasets_format.md，下面针对具体数据集来说明。

**(2) 数据格式分析**

![1](./pics/dataset/23.png)

数据文件格式说明如下：

- **dataset_info.md** - Dataset-specific information.
- **camera.yml** - Camera parameters (only for simulation, see below).
- **models[_MODELTYPE]** - 3D object models.
- **train[_TRAINTYPE]/YY/{rgb,depth,obj,seg}** - Training images of object YY.
- **test[_TESTTYPE]/ZZ/{rgb,depth,mask}** - Test images of scene ZZ.
- **vis_gt_poses** - Visualizations of the ground truth object poses.


**MODELTYPE**, **TRAINTYPE** and **TESTTYPE** are optional and are used if more data types are available.

一共15个类别，每个类别下的文件夹组织如下：

- **rgb** - Color images.
- **depth** - Depth images (saved as 16-bit unsigned short, see info.yml forthe depth units).
- **obj** (optional) - Object coordinate images [4].
- **seg** (optional) - Segmentation masks of the objects (for training images).
- **mask** (optional) - Masks of the regions of interest (for test images).

**(3) 具体文件分析**

**1) camera.yml**

​     存储的是Microsoft Kinect v1摄像头的内部参数：

​		width: 640
​		height: 480
​		fx: 572.41140
​		fy: 573.57043
​		cx: 325.26110
​		cy: 242.04899   

**2) dataset_info.md**

​    存储数据集的说明信息，例如类别个数，采集角度等等信息，非常重要。

**3) models.zip**

​    模型参数文件。3D对象模型以PLY（ascii）格式提供。 所有模型都包含顶点法线， 大多数模型还包含顶点颜色或顶点纹理坐标，并将纹理保存为单独的图像。

**4) train.zip**

​    训练压缩文件。对于任何一致图片，其内部数据的排列规则如下：

- **cam_K** - 3x3 intrinsic camera matrix K (saved row-wise).
- **cam_R_w2c** (optional) - 3x3 rotation matrix R_w2c (saved row-wise).
- **cam_t_w2c** (optional) - 3x1 translation vector t_w2c.
- **depth_scale** (optional) - Multiply the depth images with this factor to getdepth in mm.
- **view_level** (optional) - Viewpoint subdivision level, see below.

​    而 ground truth 数据集格式如下：

- **obj_id** - Object ID.
- **cam_R_m2c** - 3x3 rotation matrix R_m2c (saved row-wise).
- **cam_t_m2c** - 3x1 translation vector t_m2c.
- **obj_bb** - 2D bounding box of projection of the 3D object model at theground truth pose. It is given by (x, y, width, height), where (x, y) is thetop-left corner of the bounding box.

## 4.人脸数据集

### 4.1 人脸识别数据集

#### 4.1.1 PubFig

​    下载链接：http://www.cs.columbia.edu/CAVE/databases/pubfig/。

​    2009年公布的哥伦比亚大学公众人物脸部数据集，包含有200个人的58,797人脸图像，主要用于非限制场景下的人脸识别，全部是真实场景下图片。

#### 4.1.2 LFW

​    下载链接：http://vis-www.cs.umass.edu/lfw/ 

​    Labeled Faces in the Wild Home (LFW)数据集是为了研究非限制环境下的人脸识别问题而建立的。这个数据集包含超过5749个人,共13233幅人脸图像。每个人脸均被标准化为一个人名。其中，大约1680个人包含两个以上的人脸。这个集合被广泛应用于评价Face Verification算法的性能。

#### 4.1.3  FaceDB

​    下载链接：https://www.bioid.com/facedb/

​    这个数据集包含了1521幅分辨率为384x286像素的灰度图像。 每一幅图像来自于23个不同的测试人员的正面角度的人脸。为了便于做比较，这个数据集也包含了对人脸图像对应的手工标注的人眼位置文件。

#### 4.1.4 MegaFace

​    下载链接：http://megaface.cs.washington.edu/dataset/download.html

​      MegaFace资料集包含一百万张图片，代表690000个独特的人。所有数据都是华盛顿大学从Flickr（雅虎旗下图片分享网站）组织收集的。这是第一个在一百万规模级别的面部识别算法测试基准。 现有脸部识别系统仍难以准确识别超过百万的数据量。这个项目旨在研究当数据库规模提升数个量级时，现有的脸部识别系统能否维持可靠的准确率。

### 4.2 人脸检测数据集 

#### 4.2.1 FDDB

​    下载链接：http://vis-www.cs.umass.edu/fddb/

​     FDDB 是公认的人脸检测评测集合。包含了数据集合和评测标准（benchmark）。这个集合包含了2845张图像（5171人脸）。

#### 4.2.2 MALF

​    下载链接：http://www.cbsr.ia.ac.cn/faceevaluation/

​     Multi-Attribute Labelled Faces(MALF)是为了细粒度的评估野外环境中人脸检测模型而设计的数据库。数据主要来源于Internet，包含5250个图像，11931个人脸。每一幅图像包含正方形边界框，俯仰、蜷缩等姿势等。该数据集忽略了小于20*20的人脸，大约838个人脸，占该数据集的7%。同时，该数据集还提供了性别，是否带眼镜，是否遮挡，是否是夸张的表情等信息。

### 4.3 人脸属性数据集

#### 4.3.1  CelebA

​    下载链接：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

​    这是由香港中文大学汤晓鸥教授实验室公布的大型人脸识别数据集。该数据集包含有200K张人脸图片，人脸属性有40多种，主要用于人脸属性的识别。

#### 4.3.2 CK+ 

​    下载链接：https://www.pitt.edu/~emotion/ck-spread.htm

​    Cohn-Kanade AU-Coded Expression Database(CK+)包括123个人, 593 个 image sequence，每个image sequence的最后一张 Frame 都有action units 的label，而在这593个image sequence中，有327个sequence 有 emotion的 label。这个数据库是人脸表情识别中比较流行的一个数据库，很多文章都会用到这个数据做测试。

### 4.4 人脸关键点检测数据集

#### 4.4.1LFPW

​    下载链接：http://neerajkumar.org/databases/lfpw/

​     Labeled Face Parts in the Wild (LFPW) Dataset(LFPW)包括1132幅图像,每个人脸标定29个关键点

#### 4.4.2 AFLW

​    下载链接：http://lrs.icg.tugraz.at/research/aflw/

​    Annotated Facial Landmarks in the Wild(AFLW)人脸数据库是一个包括多姿态、多视角的大规模人脸数据库，而且每个人脸都被标注了21个特征点。此数据库信息量非常大，包括了各种姿态、表情、光照、种族等因素影响的图片。AFLW人脸数据库大约包括25000万已手工标注的人脸图片，其中59%为女性，41%为男性，大部分的图片都是彩色，只有少部分是灰色图片。该数据库非常适合用于人脸识别、人脸[检](http://cpro.baidu.com/cpro/ui/uijs.php?adclass=0&app_id=0&c=news&cf=1001&ch=0&di=8&fv=18&is_app=0&jk=c619a41d10e6c998&k=%BC%EC%B2%E2&k0=%BC%EC%B2%E2&kdi0=0&luki=2&n=10&p=baidu&q=04007150_cpr&rb=0&rs=1&seller_id=1&sid=98c9e6101da419c6&ssp2=1&stid=0&t=tpclicked3_hc&td=2231957&tu=u2231957&u=http%3A%2F%2Fwww.thinkface.cn%2Fthread-1735-1-1.html&urlid=0)测、人脸对齐等方面的[研](http://cpro.baidu.com/cpro/ui/uijs.php?adclass=0&app_id=0&c=news&cf=1001&ch=0&di=8&fv=18&is_app=0&jk=c619a41d10e6c998&k=%D1%D0%BE%BF&k0=%D1%D0%BE%BF&kdi0=0&luki=1&n=10&p=baidu&q=04007150_cpr&rb=0&rs=1&seller_id=1&sid=98c9e6101da419c6&ssp2=1&stid=0&t=tpclicked3_hc&td=2231957&tu=u2231957&u=http%3A%2F%2Fwww.thinkface.cn%2Fthread-1735-1-1.html&urlid=0)究，具有很高的研究价值。

### 4.5 活体检测数据集

#### 4.5.1 CASIA

​    下载链接：http://www.cbsr.ia.ac.cn/english/FaceAntiSpoofDatabases.asp

​     总共包括50个人,每个人12段视频