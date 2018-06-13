## 创新点

1. 提出金字塔场景解析网络（pyramid scene parsing network ）来增强上下文。
2. 使用一种有效的优化策略： deeply supervised loss

## 思想

首先，场景解析实际上就是语义分割，只不过场景解析场景比较复杂，物体的类别也比较多，因此场景解析的任务也比较难。FCN类方法在复杂场景的解析存在一些问题。比如下图：

上下文关系是非常中啊哟，举个例子，飞机要么在天上飞，要么在飞机场上跑，但不可能在一个公路上。对于第一行中的例子，fcn将船预测成car，但如果能知道car几乎不可能在水上这个信息的话，就不会预测错误。



**为了获取global contextual prior、sub-regions contextual prior作者提出了Pyramid pooling module。**



## 模型

1. 用CNN古剑网络提取图片特征，得到最后一层的卷积层的feature map（b）。骨架网络的选择是多样的，不过这篇文章中作者选择的是ResNet架构。

2. 然后通过空间金字塔进行pool。pool之后的feature map通道比较大，因此通过1x1的卷积进行降维处理。此时得到的feature map尺寸是不一样的，因此通过上采样方法将它们的尺寸变成一样，最后进行concat就得到了包含局部上下文信息和全局上下文信息的特征表达。

   （c）中红色部分代表global pooling，捕获的是 global contextual prior。2x2、3x3、6x6捕获的是不同尺度sub-regions contextual prior

3. 将2得到的结果输入到最后的卷积层中进行预测。最后得到的结果是原图的1/8。因此在测试的时候还要经过上采样得到最后的分割图。

特征金字塔融合了四种尺度的特征。

注意一点：作者说金字塔的层级数和和各个层级的大小是可以修改的。

## 训练

为了更好地训练网络，作者在ResNet的stage4增加了一个附属loss。附属loss可以更好地优化学习。为了平衡两个losss，作者在它们前面添加了权重。





https://www.cnblogs.com/everyday-haoguo/p/Note-PSPNet.html