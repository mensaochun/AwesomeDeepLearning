# RefineNet

## 论文信息

论文地址：[RefineNet: Multi-Path Refinement Networks with Identity Mappings for High-Resolution Semantic Segmentation](https://arxiv.org/pdf/1611.06612.pdf)

发表时间：20 Nov 2016

## 创新点

1. 本文提出一种叫做RefineNet的网络模块，它是基于Resnet的残差连接的思想设计的，可以充分利用下采样过程损失的信息，使稠密预测更为精准。
2. 提出chained residual pooling，能够以一种有效的方式来捕捉背景上下文信息。

## 思想

目前流行的深度网络，比如VGG，Resnet等，由于pooling和卷积步长的存在，feature map会越来越小，导致损失一些细粒度的信息（低层feature map有较丰富的细粒度信息，高层feature map则拥有更抽象，粗粒度的信息）。对于分类问题而言，只需要深层的强语义信息就能表现较好，但是对于稠密预测问题，比如逐像素的图像分割问题，除了需要强语义信息之外，还需要高空间分辨率。

针对这些问题，很多方法都提出了解决方案：

1. 针对pooling下采样过程中的分辨率损失，采用deconvolution恢复。但是却很难恢复位置信息。
2. 使用空洞卷积保持分辨率，增大感受野，但是这么做有两个缺点：A.明显增加了计算代价。B.空洞卷积是一种coarse sub-sampling，因此容易损失重要信息。
3. 通过skip connection来产生高分辨率的预测。

作者认为高级语义特征可以更好地进行分类识别，而低级别视觉特征有助于生成清晰、详细的边界。所以作者认为第3点是很好的思路。基于此，作者提出了RefineNet，其主要贡献为：

1. 提出一种多路径refinement网络，称为RefineNet。这种网络可以使用各个层级的features，使得语义分割更为精准。
2. RefineNet中所有部分都利用residual connections（identity mappings），使得梯度更容易短向或者长向前传，使段端对端的训练变得更加容易和高效。
3. 提出了一种叫做chained residual pooling的模块，它可以从一个大的图像区域捕捉背景上下文信息。

## 模型

### Multi-path refinement 

根据feature map的分辨率将预训练RESNET划分为4个block，并采用4个RefineNet的级联结构，每个RefinetNet都接收一个相应的RESNET block的输出和之前的RefineNet。

注意：

1. 这样的设计不是唯一的。实际上每个RefineNet可以接收多个RESNET blocks。不过这里只将前者。
2. 虽然所有的RefineNet都具有相同的内部结构，但是它们的参数没有不一样，这样允许更灵活地适应各个级别的细节信息。

![1](./pics/1.png)

为了解决深度网络缺少细粒度信息的这个限制。

多路径refinement

### RefineNet

![2](./pics/2.png)

RefineNet包括以下几种小模块：

1. Residual convolution unit ：对ResNet block进行2层的卷积操作。注意这里有多个ResNet block作为输入。
2. Multi-resolution fusion：将1中得到的feature map进行加和融合。
3. Chained residual pooling ：该模块用于从一个大图像区域中捕捉背景上下文。注意：pooling的stride为1。
4. Output convolutions：由三个RCUs构成。

## 实验

本文在很多数据集上做了实验，效果比较好，这里仅看PASCAL VOC2012的结果：

![3](./pics/3.png)

与其他方法的对比：

![4](./pics/4.png)