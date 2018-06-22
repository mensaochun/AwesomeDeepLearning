# 基础网络

## Resnet

### 1.创新点

提出残差学习架构来解决深度网络的退化问题。

### 2.思想

​	神经网络越深，能提取更高层次的特征，拟合能力更强。因此神经网络的深度是很重要的。但是有一个问题：是不是越深的网络就越好呢？一个回答可能就是：并不是，因为网络越深，越可能导致梯度爆炸或者消失的问题。但是实际上，现在已经有很多输入层normalization的以及中间层normalization（比如batch normalization），可以比较有效地解决梯度问题。当深度网络可以收敛的时候，它实际上面临一个退化（degradation）的问题：网络深度增加，准确率却不增加，甚至下降。这个问题并不是由于过拟合导致的，因为网络越深训练误差反而越高。

​	退化问题表明，并不是所有的系统（systems）都容易优化。现在假设一个这样的问题，我有一个浅层的网络，它可以达到某个准确率。如果复制这个浅层的网络的全部参数，然后后面加上一些其他的层，如果后面的层能学习一个identity map的话，那么效果就会和浅层的一样。但实际上这个并不容易做到。这就是退化问题导致的。

​	这篇文章，提出深度残差学习框架来解决退化问题。思路是，我现在不是直接学习目标H（x）,而是转而学习F(x)=H(x)-x。这里假设学习残差比直接学习H（x)更加容易。考虑一个特殊的例子：假设H(x)就是一个identity map, 那么这个时候将残差变为0比直接通过叠加多层非线性层来直接学习identity map更加容易。

​	在实际的操作中，F(x)+x这种方式可以很容易实现，我们通过shortcut connections的方式将x直接进行跨层喂给后面的层。x实际上就是一个identity map，这个identity map不需要任何参数。

### 3.相关工作

**残差表示**

​	这个在之前很多的视觉问题中都有用到。

**shortcut connection**

​	这个也并不是这篇文章首次提出的。之前的神经网络就有研究这个。比较近的工作，inception网络也有用到。还有就是highway network，它也用到了shortcut connection，不过他们用的不是identity map，而是一个gating function。也就是说，不是将所有信息都跨层喂给后面的层，而是设置了一定的比例。这个和lstm的思想有点像了。

### 4.模型

**Residual learning**

如上所述，不在赘述。

**Identity Mapping by Shortcuts**

​	这里要考虑跨层连接的维度问题。当跨层连接的feature map的维度相同，可以很容易做到Identity map。

当feature map的大小不同的时候可以采用两种方式进行跨层连接。

- 对x进行补零


- 将x进行线性投影

两种方式，在feature map大小变化的时候，都是卷积的步长为2进行的。



**Network Architectures**

![1](./pics/1.jpg)

在每个卷积之后，激活之前。都做bn。

没有使用dropout。

网络是全卷积的，没有全连接层？与VGG19网络相比，FLOP（浮点运算，包括加法和乘法）更少。

**Implementation**

**实验**

Imagenet classification

plain network

验证集上看，34层的plain网络相比于18层的plain网络具有更大的validation误差。文章验证了这个优化问题不是因为gradient vanish造成的。因为BN已经比较好地解决了这个问题。作者推测这个可能是因为34的plain网络具有更慢的收敛速度，不过是否是这样还有待于未来的研究。

residual network



**deeper bottleneck架构**

BottleNeck的设计为了降低训练时间，降低网络的复杂度。