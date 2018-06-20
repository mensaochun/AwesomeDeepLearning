# LSTM

## RNN的学习目标

每个时间节点都可以有学习目标。学习目标的形式与一半神经网络没有什么差别。

![1](E:\DeepLearning\RNN\pics2\1.png)

## RNN的训练

RNN是通过（Backpropagation through time）BPTT来进行训练的。

![2](E:\DeepLearning\RNN\pics2\2.png)

### BPTT

在将BPTT之前，先回顾一下BP。

![8](E:\DeepLearning\RNN\pics2\8.png)

注意：三角形表示放大器，实际上就是乘以一个scale。

![9](E:\DeepLearning\RNN\pics2\9.png)

将RNN展开之后，实际上就是一个很深的神经网络。所以RNN也可以看成神经网络，只是它很深。因此BPTT也可以跟BP一样。

![10](E:\DeepLearning\RNN\pics2\10.png)

![11](E:\DeepLearning\RNN\pics2\11.png)

![12](E:\DeepLearning\RNN\pics2\12.png)

现在有一个问题就是，网络中有很多参数是共享的，因此共享的参数不能分开更新，需要同步更新，因此更新公式有所改变：

![13](E:\DeepLearning\RNN\pics2\13.png)

最后BPTT的前向和反向传播可以表示为：

![14](E:\DeepLearning\RNN\pics2\14.png)

### 训练容易出现的问题

但是RNN的训练是非常不稳定的。

![3](E:\DeepLearning\RNN\pics2\3.png)

原因是因为RNN的loss function是非常崎岖的。有的地方很陡峭，有的地方很平坦。当刚好落在陡峭的地方的时候，梯度会非常大，然后梯度就会飞走。有时候跨过峭壁，导致loss变得很大。解决这个问题的一种方法就是将梯度clip。

![4](E:\DeepLearning\RNN\pics2\4.png)

为什么会出现这种情况？这是因为RNN的结构致使参数可能进行多次连乘。如果参数w如果为1.01，多次相乘却会使一个很大的数字，如果w是一个比1小一点的数，比如0.99，结果会变得非常小。这样计算梯度的时候就可能会非常大或者非常小，前者需要一个很小的学习率，后者需要一个很大的学习率。

![5](E:\DeepLearning\RNN\pics2\5.png)

解决RNN梯度容易消失的问题，就提出了LSTM。注意LSTM只能解决梯度消失的问题，并不能解决梯度爆炸的问题。

RNN和LSTM处理memory不一样

RNNmemory每次都被洗掉。

LSTM的memory和input是进行相加的。（问题，Forget gate也可以洗掉啊，LSTM最初的版本也是没有forget gate的。但实际上forget gate通常都是不会打开的。）

![6](E:\DeepLearning\RNN\pics2\6.png)

还有其他方法来解决梯度消失的问题。参考以下两篇文章。

![7](E:\DeepLearning\RNN\pics2\7.png)







