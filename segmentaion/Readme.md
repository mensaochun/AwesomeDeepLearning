# 深度学习图像语义分割

参考：

1. [A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#dilation)
2. [空洞卷积](https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25)
3. [知乎关于空洞卷积](https://www.zhihu.com/question/54149221)

## 一、什么是语义分割

​	语义分割是在像素级别上的分类，属于同一类的所有物体的像素都要被归为一类。比如说如下的照片，属于人的像素都要分成一类，属于摩托车的像素也要分成一类，除此之外还有背景像素也被分为一类。注意语义分割不同于实例分割，举例来说，如果一张照片中有多个人，对于语义分割来说，只要将所由人的像素都归为一类，但是实例分割还要将不同人的像素归为不同的类。也就是说示例分割比语义分割更进一步。

除此之外，还需要分割出边界。

## 二、传统语义分割方法

​	在深度学习方法流行之前，TextonForest和基于随机森林的方法是用得比较多的方法。深度学习方法比传统方法好很多，所以这里就不讲传统的方法了。

CNN分类网络有两个问题：全连接网络的大小必须固定，这就要求输入的图像必须要有固定的尺寸。

CNN网络用语语义分割有个问题：pooling由于增大了感受野，可以整合context信息（context中文称为语境或者上下文，通俗的理解就是综合了更多的信息来进行决策），因此有利于分类，但是却削弱了位置信息。而语义分割需要更多的位置信息。由两种不同的架构来解决这个问题：

1. encoder-decoder架构。encoder通过pooling逐渐减少了空间维度，decoder逐渐恢复物体的细节和空间维度。通常从encoder到decoder还有shortcut connetction（捷径连接，也就是跨层连接）。其中U-net就是这种架构很流行的一种】


2. 第二种架构叫做dilated/atrous （空洞）卷积，这种结构取消了pooling。

条件随机场后处理用来改善分割效果。CFRs是一种图模型，它可以基于潜在的图像强度使分割更加平滑。工作原理就是，相同强度的像素应该被标为同一类。CRFs能够将分数提升1-2%左右。

## 三、深度学习语义分割方法

下面总结一些从FCN进行各种改进的分割架构。

主要包括以下几种架构：

1. [FCN](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#fcn)
2. [SegNet](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#segnet)
3. [Dilated Convolutions](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#dilation)
4. [DeepLab (v1 & v2)](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#deeplab)
5. [RefineNet](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#refinenet)
6. [PSPNet](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#pspnet)
7. [Large Kernel Matters](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#large-kernel)
8. [DeepLab v3](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#deeplabv3)

### 1. FCN

发表时间：14 Nov 2014

主要贡献：

1. 是端对端的卷积语义分割网络的鼻祖。
2. 通过deconvolutional layers进行上采样
3. 通过skip connection改善了上采样的粗糙度。

FCN文章发现，全连接层实际上可以看作是一种卷积层。

上采样产生一个很粗粒度的分割图像，因为在pooling的时候信息都损失了。

因此考虑通过shortcut/skip connection将更高分辨率的feature map信息引入进来。

*Benchmarks (VOC2012)*:

| Score | Comment                               | Source                                   |
| ----- | ------------------------------------- | ---------------------------------------- |
| 62.2  | -                                     | [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=6103#KEY_FCN-8s) |
| 67.2  | More momentum. Not described in paper | [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=6103#KEY_FCN-8s-heavy) |

评论：

FCN是基于深度学习的语义分割的开山之作，尽管现在很多方法都超越了FCN，但它的思想仍然有很重要的意义。



### 2. Segnet

创新点：



### 3. Dilated convolution

**论文信息**

> Multi-Scale Context Aggregation by Dilated Convolutions
>
> Submitted on 23 Nov 2015
>
> [Arxiv Link](https://arxiv.org/abs/1511.07122)



**创新点**

1. 使用空洞卷积用来进行稠密预测（dense prediction）。
2. 提出context module，它使用空洞卷积来进行多尺度的整合。



**简要解释**

​	pooling对于分类网络来说可以增大感受野，但是这个对语义分割来说可能降低分辨率。因此作者提出一种叫做dilated convolution的操作。

![dilation](E:\DeepLearning\segmentaion\pics\dilation.gif)

​	dilated卷积，这个在deeplab中称为atrous卷积。可以很好地提升感受野，但是仍然可以保持空间的维度。

VGG网络的最后两个pooling层给拿掉了，并且之后的卷积层被dilated 卷积取代。



***Benchmarks (VOC2012)*:**

| Score | Comment                      | Source                |
| ----- | ---------------------------- | --------------------- |
| 71.3  | frontend                     | reported in the paper |
| 73.5  | frontend + context           | reported in the paper |
| 74.7  | frontend + context + CRF     | reported in the paper |
| 75.3  | frontend + context + CRF-RNN | reported in the paper |



**评论**



### 4. DeepLab(v1,v2)

创新点：

1. 使用空洞卷积
2. 提出了arous空间金字塔池化。
3. 使用全连接CRF


### 5. RefineNet

- RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
- Submitted on 20 Nov 2016
- [Arxiv Link](https://arxiv.org/abs/1611.06612)

*Key Contributions*:

- Encoder-Decoder architecture with well thought-out decoder blocks
- 采用了残差连接的设计





### 6. deeplab v3

