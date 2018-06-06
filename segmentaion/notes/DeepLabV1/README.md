# DeepLab V1

## 论文信息

论文地址：

论文发表日期：

https://zhuanlan.zhihu.com/p/37124598



## 创新点

本文将深度卷积网络（DCNN）和概率图模型结合起来，解决pixel-level的语义分割问题。作者指出DCNN的最后层的特征通常是非常high-level的，对于需要精准定位的语义分割来说通常是不够的。因此作者提出将DCNN的最后一层加上全连接条件随机场（fully connected conditional random field(CRF)）的方法来解决语义分割中的定位不准确的问题。

## 思想

DCNN网络具有不变形特性，对于图像分类这些high-level的任务来说，这种不变形是非常有用的。但是对于语义分割和姿态估计等这些low-level的任务来说，需要非常精准的像素定位而非特征抽象，这种不变性反而是非常不利的。

DCNN如果直接用于语义分割会存在两个问题：

1. 下采样(如max pooling)会导致分辨率降低，不利于像素级别的语义分割问题。
2. DCNN的空间不变性（spatial insensitivity/invariance），分类器获取对象类别的信息是需要空间变换不变性的，但这限制了DCNN的空间定位精度。

针对以上两个问题，文章作者分别提出了解决方法：

对于1：使用atrous卷积（和空洞卷积是一个意思）来减少分辨率降低。

对于2：使用全连接CRF来提高模型捕获细节和边缘信息的能力。

## 基本概念

### 空洞卷积

- 参考资料： [知乎提问：如何理解空洞卷积（dilated convolution）？](https://www.zhihu.com/question/54149221/answer/192025860)

- 空洞卷积的引入：

- - 由于普通下采样（max pooling）方法导致分辨率下降、局部信息丢失。
  - 为什么要用max pooling：每个像素有较大receptive field，且减小图像尺寸。
  - 想使用一种方法，不进行max pooling，但也能使每个像素有较大receptive field。

- 论文中原图描述空洞卷积：

![img](https://pic3.zhimg.com/80/v2-88009e435ba92c35bf291758b701cc05_hd.jpg)

- 论文 [Multi-Scale Context Aggregation by Dilated Convolutions](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1511.07122)原图描述空洞卷积：

![img](https://pic1.zhimg.com/80/v2-3c6448f83a81daaafa5c00aeff033880_hd.jpg)

- 基于pre-trained model理解空洞卷积的意义

- - 由于max pooling存在问题，所以在DeepLab中，减少了原有VGG网络中的max pooling的数量。
  - 由于max pooling的减少，因此不能使用普通卷基并调用VGG pre-trained model中的权重来进行训练。
  - 使用空洞卷积时，其权重还可以使用VGG pre-trained model。

## FULLY-CONNECTED CRF

CRFs在传统图像处理中主要用于平滑具有噪声的图像分割结果(在这种应用场景下的CRFs作者称为short-range CRFs)。但是对于DCNN架构来说，网络输出的scope map通常是非常平滑的，相邻像素之间通常会具有同质的分类结果，因此short-range CRFs可能是有害的，我们的目标是去恢复局部的细节信息，而不是去平滑。

为了解决short-range的这些问题，作者将FULLY-CONNECTED CRF整合到网络中。

- 参考资料：

- - [知乎文章：FCN(3)——DenseCRF](https://zhuanlan.zhihu.com/p/22464586)。该专栏中还有其他几篇关于CRF的文章。

- CRF的引入

- - CRF在传统图像处理上主要做平滑处理。
  - 但对于CNN来说，short-range CRFs可能会起到反作用，因为我们的目标是恢复局部信息，而不是进一步平滑图像。
  - 引入fully connected CRF来解决这个问题，考虑全局的信息。

- 这部分还不明白，之后要单独学习。

## **3.3. Multi-scale Prediction**

- 多尺寸预测，希望获得更好的边界信息。

- 引入：与FCN skip layer类似。

- 实现：

- - 在输入图片与前四个max pooling后添加MLP（多层感知机，包括3*3*128以及1*1*128），得到预测结果。
  - 这四个预测结果与最终模型输出拼接（concatenate）到一起，相当于多了128*5=640个channel。

- 效果不如dense CRF，但也有一定提高。最终模型是结合了Desne CRF与Multi-scale Prediction。



## 模型

主要是对原有VGG网络进行了一些变换：

将原先的全连接层通过卷基层来实现。

- VGG网络中原有5个max pooling，先将后两个max pooling去除（看别的博客中说，其实没有去除，只是将max pooling的stride从2变为1），相当于只进行了8倍下采样。
- 将后两个max pooling后的普通卷基层，改为使用带孔卷积。


- 为了控制视野域（同时减少计算量），对于VGG中的第一个fully connected convlution layer，即7*7的卷基层，使用3*3或4*4的卷积来替代。

- - 计算时间减少了2-3倍。

- 其他训练信息

- - 损失函数：交叉熵之和。
  - 训练数据label：对原始Ground Truth进行下采样8倍，得到训练label。
  - 预测数据label：对预测结果进行双线性上采样8倍，得到预测结果。

------

## **4. 其他**

- 疑问：

- - multi-scale，输入图片以及几个max pooling的输出的分辨率都不一致，那是通过什么方法最终获取到同一分辨率的？

