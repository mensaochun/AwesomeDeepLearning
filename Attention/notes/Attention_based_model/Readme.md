# Attention based model

计划分为三个部分： 
浅谈Attention-based Model【原理篇】（你在这里） 
[浅谈Attention-based Model【源码篇】](http://blog.csdn.net/wuzqChom/article/details/77918780) 
浅谈Attention-based Model【实践篇】

### 1. 为什么需要Attention

最基本的seq2seq模型包含一个encoder和一个decoder，encoder部分通常将一个输入的句子编码成一个固定大小的state。decoder部分则可以有不同的设计。下图是比较常用的一种设计，将state作为decoder每个时刻的输入，当前时刻的输出作为下一个时刻的输入。这样的设计存在一个问题：decoder每个时刻的输入都是state，但是我们希望能够有注意力的引入，比如我们更希望在输出machine的时候，输入的state中包含更多关于“机器”的信息。 
![1](E:\DeepLearning\Attention\notes\Attention_based_model\pics\1.png)

### 2. Attention-based Model是什么

Attention-based Model其实就是一个相似性的度量，当前的输入与目标状态越相似，那么在当前的输入的权重就会越大，说明当前的输出越依赖于当前的输入。严格来说，Attention并算不上是一种新的model，而仅仅是在以往的模型中加入attention的思想，所以Attention-based Model或者Attention Mechanism是比较合理的叫法，而非Attention Model。

> 没有attention机制的encoder-decoder结构通常把encoder的最后一个状态作为decoder的输入（可能作为初始化，也可能作为每一时刻的输入），但是encoder的state毕竟是有限的，存储不了太多的信息，对于decoder过程，每一个步骤都和之前的输入都没有关系了，只与这个传入的state有关。attention机制的引入之后，decoder根据时刻的不同，让每一时刻的输入都有所不同。

### 3. 机器翻译中的Attention

对于机器翻译来说，比如我们翻译“机器学习”,在翻译“machine”的时候，我们希望模型更加关注的是“机器”而不是“学习”。那么，就从这个例子开始说吧。

![2](E:\DeepLearning\Attention\notes\Attention_based_model\pics\2.png) 

刚才说了，attention其实就是一个当前的输入与输出的匹配度。在上图中，即为$h_1$和$z_0$的匹配度（**$h_1$为当前时刻RNN的隐层输出向量，而不是原始输入的词向量**，$z_0$初始化向量，如rnn中的initial memory），其中的match为计算这两个向量的匹配度的模块，出来的$α^1_0$即为由match算出来的相似度。好了，基本上这个就是attention-based model 的attention部分了。那么，match什么呢？

对于“match”, 理论上任何可以计算两个向量的相似度都可以，比如：

- 余弦相似度
- 一个简单的 神经网络，输入为$h$和$w$，输出为$α$
- 或者矩阵变换$α=h^TWz$ (Multiplicative attention，Luong et al., 2015) 
  现在我们已经由match模块算出了当前输入输出的匹配度，然后我们需要计算当前的输出（实际为decoder端的隐状态）和每一个输入做一次match计算，分别可以得到当前的输出和所有输入的匹配度，由于计算出来并没有归一化，所以我们使用softmax，使其输出时所有权重之和为1。那么和每一个输入的权重都有了（由于下一个输出为“machine”，我们希望第一个权重和第二个权权重越大越好），那么我们可以计算出其加权向量和，作为下一次的输入。

> 这里有一个问题：就是如果match用后面的两种，那么参数应该怎么学呢？
>
> 就是加入match是一个简单地神经网络或者一个矩阵，神经网络的权值和矩阵里面的值怎么来？ 
> 其实这些都是可以BP的时候就可以自动学到的。比如我们明确输出是什么，在第一时刻的时候，那就会调整$z_0$和$c_0$的值，进而就会调整所有的$α$值，之前说过$α$是match的输出，如果match是后两种的话，就会进而调整match中的参数。

如下图所示： 

![3](E:\DeepLearning\Attention\notes\Attention_based_model\pics\3.png) 

那么再算出了$c_0$之后，我们就把这个向量作为rnn的输入（如果我们decoder用的是RNN的话），然后d第一个时间点的输出的编码$z_1$由$c_0$和初始状态$z_0$共同决定。我们计算得到$z_1$之后，替换之前的$z_0$再和每一个输入的encoder的vector计算匹配度，然后softmax，计算向量加权，作为第二时刻的输入……如此循环直至结束。 

![4](E:\DeepLearning\Attention\notes\Attention_based_model\pics\4.png)

![5](E:\DeepLearning\Attention\notes\Attention_based_model\pics\5.png)

![6](E:\DeepLearning\Attention\notes\Attention_based_model\pics\6.png)

### 4.image caption

image caption要做的事情就是输入一张图片，输出关于这张图片内容的描述。

对于这个任务，可以考虑这么做：将图像通过CNN编码成一个向量，然后将这个向量作为decoder每个时刻的输入，同时将上一时刻的输出也作为该时刻的输入。但这样做同样会遇到一个问题，就是decoder每个时刻的输入都是相同的向量，因此很难将注意力focus在局部有用的信息上。

![7](E:\DeepLearning\Attention\notes\Attention_based_model\pics\7.png)

因此，可以考虑将图像分割成不同的子区域，每个子区域都对应一个向量。具体来说，可以将CNN网络的全连接层去掉，留下最后的卷积层，feature map上每个点都对应一个向量，每个向量又对应原图的一个区域。同样，可以将初始化的$z_0$ 和这些子区域的向量进行match，得到相应的权重。然后对不同子区域进行加权和，作为decoder第一个时刻的输入，其他时刻的做法一样。

![8](E:\DeepLearning\Attention\notes\Attention_based_model\pics\8.png)

![9](E:\DeepLearning\Attention\notes\Attention_based_model\pics\9.png)

一些好的例子

![10](E:\DeepLearning\Attention\notes\Attention_based_model\pics\10.png)

不好的例子

![11](E:\DeepLearning\Attention\notes\Attention_based_model\pics\11.png)



### 5.vedio caption



