# 图像语义分割准确率度量方法

衡量图像语义分割准确率主要有三种方法：

1. 像素准确率（pixel accuracy, PA）
2. 平均像素准确率（mean pixel accuracy, MPA）
3. 平均IOU（Mean Intersection over Union, MIOU ）

在介绍三种方法之前，需要先说明一些符号表示的意义。

$k$：类别总数，如果包括背景的话就是$k+1$

$p_{ij}$：真实像素类别为$i$的像素被预测为类别$j$的总数量，换句话说，就是对于类别为$i$的像素来说，被错分成类别$j$的数量有多少。

$P_{ii}$ ：真实像素类别为$i$的像素被预测为类别$i$的总数量，换句话说，就是对于真实类别为$i$的像素来说，分对的像素总数有多少。 

## PA

PA的意义很简单，和我们常规的分类准确率计算没有区别，就是把分对的像素总量除以像素总数。

![1](E:\DeepLearning\segmentaion\notes\Metric\pics\1.png)

## MPA

MPA是对PA的改进，它是先对每个类计算PA，然后再对所有类的PA求平均。

![2](E:\DeepLearning\segmentaion\notes\Metric\pics\2.png)

## MIoU

在语义分割中，MIoU才是标准的准确率度量方法。它是分别对每个类计算（真实标签和预测结果的交并比）IOU，然后再对所有类别的IOU求均值。

![3](E:\DeepLearning\segmentaion\notes\Metric\pics\3.png)