# 史上最通俗易懂的GAN入门

主要从两个部分来讲GAN，第一部分，直觉，从直觉上建立起对GAN的认识。再以此切入第二部分，GAN背后的数学原理。

## I. 直觉

## 一、Basic Idea of GAN

### 1. Generation

首先了解Generation(生成)的含义。Generation就是模型通过学习一些数据，然后生成类似的数据。比如图像生成，语句生成。

![幻灯片8](E:\DeepLearning\GAN\pics\Intro\8.PNG)

GAN中用于Generation的部分称为Generator。Generator（生成器）实际上是一个函数，目前用得比较多的是神经网络。我们给Generator喂入一些随机向量，它就可以产生我们想要的结果。而且，随机输入的向量中，每一个维度都代表了某种意义或者特征，比如说画红框的维度实际上就分别代表了头发长短，头发颜色和嘴巴张开状态等特征。改变这些维度的数值会改变相应的特征。

![幻灯片9](E:\DeepLearning\GAN\pics\Intro\幻灯片9.PNG)

### 2. Discriminator

GAN中除了Generator之外，还有一个重要的组成部分叫做Discriminator(判别器)。Discriminator也是一个函数（神经网络是用得比较多的函数），它会对给定的输入，输出一个实数值，这个值越大，代表这个输入越真实，越小则越不真实。注：这个输出实数值一般会通过sigmoid压缩到0-1之间。

![幻灯片10](E:\DeepLearning\GAN\pics\Intro\幻灯片10.PNG)

### 3. 结合Generator和Discriminator

GAN中的Generator和Discriminator是相互不断进化的。如下图，第一代的Generator会比较弱，此时第一代的Discriminator可以比较好的辨别Generator生成的图片是真的假的。但是第二代的Generator会进化，生成更加真实的照片，以骗过第一代的Discriminator，此时Discriminator也不甘示弱，它也会进化得更强，用以辨别第二代Generator产生图片的真假。这个过程一直持续，直至Generator产生越来越真实的图片，而Discriminator的辨别能力也越来越强。

![幻灯片12](E:\DeepLearning\GAN\pics\Intro\幻灯片12.PNG)

### 4. 算法

现在将Generator和Discriminator相互进化的过程形式化为算法。

- 首先，初始化一个Generator和一个Discriminator。


- step1：在每次迭代中，先固定Generator，更新Discriminator。Discriminator会给Generator生成的图片一个很低的分数，而给真实的图片一个比较高的分数。


​

![幻灯片15](E:\DeepLearning\GAN\pics\Intro\幻灯片15.PNG)

- step2：固定Discriminator，更新Generator。通过更新Generator的参数，Discriminator会给Generator一个很高的得分值。这样最终的Generator就可以骗过Discriminator。


![幻灯片16](E:\DeepLearning\GAN\pics\Intro\幻灯片16.PNG)

伪代码如下（注意D(x)值一般为sigmoid输出，在0-1之间），已经解释得非常详细，就不多加解释了。

![幻灯片17](E:\DeepLearning\GAN\pics\Intro\幻灯片17.PNG)

### 5. 一个实例

二次元图像生成。我们可以看到，随着训练迭代次数的增多，生成的图片会越来越真实。

![幻灯片18](E:\DeepLearning\GAN\pics\Intro\幻灯片18.PNG)

![幻灯片19](E:\DeepLearning\GAN\pics\Intro\幻灯片19.PNG)

![幻灯片20](E:\DeepLearning\GAN\pics\Intro\幻灯片20.PNG)

![幻灯片21](E:\DeepLearning\GAN\pics\Intro\幻灯片21.PNG)

![幻灯片22](E:\DeepLearning\GAN\pics\Intro\幻灯片22.PNG)



![幻灯片23](E:\DeepLearning\GAN\pics\Intro\幻灯片23.PNG)



![幻灯片24](E:\DeepLearning\GAN\pics\Intro\幻灯片24.PNG)





![幻灯片20](E:\DeepLearning\GAN\pics\Intro\幻灯片20.PNG)

![幻灯片21](E:\DeepLearning\GAN\pics\Intro\幻灯片21.PNG)

![幻灯片22](E:\DeepLearning\GAN\pics\Intro\幻灯片22.PNG)

![幻灯片23](E:\DeepLearning\GAN\pics\Intro\幻灯片23.PNG)

![幻灯片24](E:\DeepLearning\GAN\pics\Intro\幻灯片24.PNG)



## 二、Can Generator Learn  by Itself

通过以上讲解，我们已经知道，GAN通过Generator和Discriminator相互竞争，可以生成很逼真的图片。我们现在想知道一个问题，就是如果没有Discriminator的话，Generator还可以生成图片吗？以下就对这个疑问进行解释。

### 1. 固定输入向量的Generator

实际上我们只用一个Generator就可以进行生成图片。并不需要Discriminator。可以这么做：假设我们从database中采样了m张图片，我们给每张图片都对应上一个向量。这样就构成了<向量，图片>对。这时候我们可以通过传统的监督学习得到一个Generator，对于每个输入向量，都对应着图片输出。

但是，这样的Generator有一个问题，比如对于同样是数字1的图片，输入向量在距离上应该比较接近，但是这个不容易做到。[todo]

![幻灯片35](E:\DeepLearning\GAN\pics\Intro\幻灯片35.PNG)

### 2. AutoEncoder中的Decoder作为Generator

这个时候可以考虑使用AutoEncoder。AutoEncoder中的Encoder可以将图片编码到一个低维空间。

![幻灯片36](E:\DeepLearning\GAN\pics\Intro\幻灯片36.PNG)

AutoEncoder的Decoder部分会将code解码为原始输入。然后编码和解码一起训练。

![幻灯片37](E:\DeepLearning\GAN\pics\Intro\幻灯片37.PNG)

训练好了AutoEncoder之后，就可以将Decoder拿出来作为一种Generator。

![幻灯片38](E:\DeepLearning\GAN\pics\Intro\幻灯片38.PNG)

在code空间中，通过随机输入一些向量，就可以生成相应的图片。

![幻灯片39](E:\DeepLearning\GAN\pics\Intro\幻灯片39.PNG)

![幻灯片40](E:\DeepLearning\GAN\pics\Intro\幻灯片40.PNG)

但是AutoEncoder存在一个问题，就是code之间存在一个gap。在code空间中，训练数据没有cover到的区域，很难生成一个好的图片。

![幻灯片41](E:\DeepLearning\GAN\pics\Intro\幻灯片41.PNG)

### 3. 变分自编码的Decoder作为Generator

这个时候需要用到variational AutoEncoder。变分自编码通过给code加上噪声，可以使训练数据cover到更多的code空间。其中噪声是网络Encoder输出的一个向量与高斯噪声产生的向量的积。为了不让Encoder输出的向量被训练为0，这时候还需要对其添加一个限制，如黄色框所示。

![幻灯片42](E:\DeepLearning\GAN\pics\Intro\幻灯片42.PNG)

但是variational antoEncoder也会存在一个问题，就是它不能学习到全局信息。因为它是通过最小化输出与输出之间的差距作为优化目标的，因此它的关注点只是让输入输出差距最小，而不是使原始图像完美复原。比如下图的上方两张图，输入输出只有一个pixel的差距，那么这个时候对于VAE来说，它的损失函数值会很小，这个时候对于VAE来说已经学习得很好，但是对于人类来说，这种图片实际上是很糟糕的。而对于下图的下方两张图来说，有6个pixel的差距，这个时候对于VAE来说，它的损失函数值会比较大，因此可以说它学习得不好，但是这种情况对于人类来说，是很合理的。

![幻灯片44](E:\DeepLearning\GAN\pics\Intro\幻灯片44.PNG)

## 三、Can Discriminator Generate

以上讲了单独的Generator也是可以进行生成的，那么单独的Discriminator能否也用来生成呢？答案是可以的！下面就进行详细介绍。

Discriminator实际上也是一个function。对于输入x，会输出一个scalar来判断这个x真不真。越真分值越高，越假分数越低。

![幻灯片48](E:\DeepLearning\GAN\pics\Intro\幻灯片48.PNG)

那Discriminator怎么用做生成呢？在之前我们讲到，Variational Autoencoder的生成器是不容易捕捉特征之间的相关性的，因此生成的图片经常不太真实。但是，如果用Discriminator用作Generator的话，可以很好第解决这个问题，因为假设Discriminator是CNN架构的话，它很容易学习。

![幻灯片49](E:\DeepLearning\GAN\pics\Intro\幻灯片49.PNG)



## II. 理论

### 1.MLE

首先从MLE讲起。假设我们现在有一个$P_{data}(x)$的分布，其中$x$可以看成是image。如果现在要根据这个分布去产生一张图片的话要怎么做？我们可以去寻找一个分布$P_G(x;\theta)$（这个分布受控于参数$\theta$），使它和$P_{data}(x)$很像。这个$P_G(x;\theta)$中的G(Generator)可以是多种形式，比如可以是Gaussian mixture model，也可以是神经网络，等等。接着我们可以从$P_{data}(x)$中去采样出m个sample：$\{x_1,...x_m\}$，分别去计算相应的$P_G(x^i;\theta)$，再将它们相乘得到likelihood。最后最大化likelihood就可以解得参数$\theta$，从而得到$P_{G}(x;\theta)$。利用$P_{G}(x)$就可以进行随机采样产生图片。

![幻灯片23](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片23.PNG)

实际上，我们发现，MLE实际上就是最小化$P_G(x;\theta)$和$P_{data}(x)$的KL divergence。也就是说，MLE做的事情就是让$P_{data}(x)$和$P_G(x;\theta)$这两个分布更接近。

![幻灯片24](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片24.PNG)

实际上，如果假设$P_G(x;\theta)$的G为gaussian mixture model的话，通过MLE方法计算出的$P_G(x;\theta)$进行生成图像的话通常会非常模糊，原因就是gaussian mixture model和$P_{data}(x)$通常会离得比较远。但是，如果我们把$P_G(x;\theta)$的G设置为神经网络的话，由于神经网络有强大的拟合能力，它可以较好地近似$P_{data}(x)$。

NN作为Generator的方式很简单，给NN输入一个概率分布的采样值（这个概率分布可以是Gaussian distribution也可以是normal distribution，由于NN有强大的拟合能力，不用担心它不会拟合出$P_{data}(x)$），NN就可以产生相应的x。我们要做的就是寻找NN的参数$\theta$，使NN输出的x的分布与$P_{data}(x)$越接近越好。

​	可是现在Generator产生的distribution写成式子是什么样子的？可写成：$P_G(x)=\int_zP_{prior}(z)I_{[G(x)=x]}dz$。对于Gaussian mixture model给一个x，可以很容易算它的出现的概率，但是，对于NN的话，即使知道了$P_{prior}(z)$的概率，由于G的形式很复杂，很难算出x出现的概率，因此就很难通过MLE的方式求得而GAN最大的贡献就是解决了这个问题。

![幻灯片25](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片25.PNG)

现在重新来形式化我们的问题。

首先看Generator的形式，输入是$z$，输出是$x$。现在给定一个先验分布$P_{prior}(z)$，我们要得到一个概率分布$P_G(x)$，但是现在的问题是不知道怎么计算这个G。

所以这个时候要去定义一个Discriminator。它的输入是$x$，输出是实数。它做的事情就是衡量$P_{data}(x)$和$P_G(x)$有多相近。这个Discriminator实际上就是取代了MLE的作用。只不过MLE算的是KL divergence，而D算的是另外一种divergence（后面会详细讲到）。

那么要怎样才能让D去算出两种分布的divergence呢？这个时候需要去解这个问题：
$argmin_{G}max_DV(G,D)$

首先需要定义一个function $V(G,D)$，它吃$G,D$，output一个scalar。然后$max_DV(G,D)$,最后$argmin_{G}max_DV(G,D)$。



![幻灯片26](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片26.PNG)

下面先看$max_DV(G,D)$。它的意思是，给定$G$，求出使$V(G,D)$最大的$D$。现在我们先假定$G$的个数只有3个（实际上是连续的，有无穷多个），分别为$G_1,G_2,G_3$，现在我们要做的事情就是在$G_1,G_2,G_3$中分别找到使$V$最大时所对应的$D^*$，即图中红点处。接着，再看$argmin_{G}max_DV(G,D)$，它做的事情就是在刚才找的最大值中找出最小值对应的$G^\star$。



![QQ_1](E:\DeepLearning\GAN\pics\GAN (v11)\QQ_1.png)

![QQ_2](E:\DeepLearning\GAN\pics\GAN (v11)\QQ_2.png)

现在的问题是，怎么去定义这个V呢？假设我们现在直接定义V为（先不管怎么来的）：

![QQ_3](E:\DeepLearning\GAN\pics\GAN (v11)\QQ_3.png)

这样定义之后有什么好处呢？实际上$max_DV(G,D)$就是评估了$P_G$和$P_{data}$的差异性，也就是上图中红点的高度。现在我们想找一个$G^{\star}$让这个高度最小，也就是让$P_G$和$P_{data}$最相似。

![QQ_4](E:\DeepLearning\GAN\pics\GAN (v11)\QQ_4.png)

下面就解释为什么会这样。下面进行证明：

===============================证明开始===============================

首先的问题就是，给定G，什么样的D能够$max_DV(G,D)$ ？步骤如下：

![幻灯片28](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片28.PNG)

![幻灯片29](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片29.PNG)



![幻灯片30](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片30.PNG)

![幻灯片31](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片31.PNG)

![幻灯片32](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片32.PNG)

最终我们得到最优的$G^\star$, 它使$P_G(x)$和$P_{data}(x)$最相近。

![幻灯片33](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片33.PNG)

================================证明结束=====================================

那么怎么去寻找最佳的$G^\star$呢？实际上，就是最小化Generator的Loss function。

![QQ_5](E:\DeepLearning\GAN\pics\GAN (v11)\QQ_5.png)

但是，$L(G)$中有max操作，能够进行最小化吗？实际上是可以的。假设只有三个D。这个地方没懂。。。![img](file:///E:/DeepLearning/GAN/pics/GAN%20(v11)/QQ_5.png?lastModify=1526800316)

![QQ_6](E:\DeepLearning\GAN\pics\GAN (v11)\QQ_6.png)

我们可以先给定一个$G_0$，寻找一个$D^\star_0$使$V(G_0,D)最大化。这时候去更新$$\theta_G$使得JS divergence最小，得到了$G_1$。然后按照这个步骤重复，最终就可以得到最佳解。

![幻灯片35](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片35.PNG)

会有一个问题：

![幻灯片36](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片36.PNG)

实践中：1. 期望不能做积分，就通过sample。

这个实际上就是cross entropy。



![幻灯片37](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片37.PNG)

如果D的loss很大，就说明p_data和p_G的js divergence很大，反之很小。下面少一个参数。

![幻灯片38](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片38.PNG)

learning D: 固定G，那么表现为从G中进行sample。

learning G: 固定D，那么意味着和第一项没有关系。

注意：G不能update太多，否则可能导致JS divergence太小。但是ian goodfelloe，learning D只要一次。

![幻灯片39](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片39.PNG)



![幻灯片40](E:\DeepLearning\GAN\pics\GAN (v11)\幻灯片40.PNG)



training的时候的问题。D