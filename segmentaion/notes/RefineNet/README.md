# RefineNet

## 论文信息

论文地址：[RefineNet: Multi-Path Refinement Networks with Identity Mappings for High-Resolution Semantic Segmentation](https://arxiv.org/pdf/1611.06612.pdf)

发表时间：20 Nov 2016

## 创新点

1. 本文提出一种叫做RefineNet的网络模块，它是基于Resnet的残差连接的思想设计的，可以充分利用下采样过程损失的信息，使稠密预测更为精准。
2. 提出chained residual pooling，能够以一种有效的方式来捕捉背景上下文信息。

## 思想

目前流行的深度网络，比如VGG，Resnet等，由于pooling和卷积步长的存在，feature map会越来越小，导致损失一些细粒度的信息（低层feature map有较丰富的细粒度信息，高层feature map则拥有更抽象，粗粒度的信息）。对于分类问题而言，只需要深层的强语义信息就能表现较好，但是对于稠密预测问题，比如逐像素的图像分割问题，除了需要强语义信息之外，还需要高空间分辨率。

为了解决深度网络缺少细粒度信息的这个限制。