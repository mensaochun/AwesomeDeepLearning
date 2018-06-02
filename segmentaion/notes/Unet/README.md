# U-Net

## 论文信息

论文地址：[U-Net: Convolutional Networks for Biomedical Image Segmentation]()

发表时间：

## 创新点

深度网络通常需要大量的数据进行训练，当样本量较少的情况下，深度网络可能表现没那么好。对于这个问题，本文提出了新的网络架构和图像增强策略。网络架构包括encode和decode（论文中称为浓缩路径contracting path和扩展路径expanding path，实际上就是encode和decode），encode可以有效地捕捉上下文信息，而decode可以较好地预测位置信息。

## 模型

### 1. 模型设计思想

Unet是基于FCN网络的思想设计的，整个网络只有卷积层，而没有全连接层。网络的浓缩路径，图像分辨率逐渐降低，上下文信息会逐渐增强。在扩展路径中，通过上下样的方式，让特征图的分辨率逐渐增大，为了结合低层feature map的强位置信息，Unet进行了横向连接，也就是将浓缩路径中的相应部分连接到扩展路径中。需要注意的是，这种横向连接是在通道上进行concate。

![1](/home/pi/stone/Notes/DeepLearning/segmentaion/notes/Unet/pics/1.png)

Unet做的修改有：

1. 在上采样部分，feature map的通道数非常大，作者认为这样可以将上下文信息传递到分辨率更高的层当中。这样做的一个结果就是它基本上和浓缩路径对称了，因此看上去像一个U形的结构。


### 2. 网络结构

网络结构包括浓缩路径和扩展路径。

浓缩路径：

1. 注意图像输入是经过tile的。因此网络的输出是和图像的输入大小是不一样的。
2. 每经过两个3x3conv（没有padding）之后，会跟上一个2x2的max-pooling进行下采样。

扩展路径：

1. 使用2x2的deconv来进行上采样，上采样的过程中，通道数减半。同时，将上采样的feature map和浓缩路径中相应的feature map进行concate（注意：由于浓缩路径中的conv都是没有做padding动作的，这会导致扩展路径中的feature map和浓缩路径中相应位置的feature map大小不一致，这个时候就要将浓缩路径中的feature map进行crop再和扩展路径中的feature map进行concate）。

2. 上采样之后，再跟上3x3的卷积。

3. 最后一层使用1x1的卷积，将通道数map到类别数。




## 训练

### 1. 损失函数计算

网络输出的是pixel-wise的softmax。表达式如下：

![2](/home/pi/stone/Notes/DeepLearning/segmentaion/notes/Unet/pics/2.png)

其中，$x$为二维平面（$Ω $）上的像素位置，$a_k(x)$表示网络最后输出层中pixel $x$对应的第$k$个通道的值，$K$是类别总数。$p_k(x)$表示像素$x$属于$k$类的概率。

损失函数使用negative cross entropy。cross entropy的数学表达式如下：

![3](/home/pi/stone/Notes/DeepLearning/segmentaion/notes/Unet/pics/3.png)

其中$p_l(x)$表示$x$在真实label所在通道上的输出概率。需要特别注意的是cross entropy中还添加一个权重项$w(x)$ 。这是因为考虑到物体间的边界需要更多的关注，所对应的损失权重需要更大。

### 2. 像素损失权重计算

在损失函数计算中我们讲到对于边界像素我们给的损失权重要更大，但怎么获取这个权重？

我们得到一张图片的ground truth是一个二值的mask，本文首先采用形态学方法去计算出物体的边界。然后通过以下的表达式去计算权重图。

![4](/home/pi/stone/Notes/DeepLearning/segmentaion/notes/Unet/pics/4.png)

其中wc(x)是类别权重，需要根据训练数据集中的各类别出现的频率来进行统计，类别出现的频率越高，应该给的权重越低，越高则给的权重越高（文章没有详细说是怎么计算的）。d1表示物体像素到最近cell的边界的距离，d2表示物体像素到第二近的cell的边界的距离。在本文中，设置w0=10。

 ### 3. 图像增强

用到了很多变形的图像增强方法。

