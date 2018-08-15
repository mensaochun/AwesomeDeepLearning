---
typora-copy-images-to: pics
---

# 6d

在上篇yolo-6d中已经讲了如何通过物体3D bounding box在2D图像上的投影的8个角点和一个中心点来得到物体的6d姿态，这是一种思路，但deep6d提供了另外一种思路，直接回归6d pose。



## 为什么要基于目标检测框架

我们发现，很多6d姿态估计的方法都是基于2D图像的目标检测的。那为什么要这么做呢？我的看法是：

1. 为了检测多目标。目前主流的目标检测方法都是基于多目标检测设计的，而6d姿态估计也涉及多目标，并且也是基于2D图像来进行预测的，只是目标检测预测的目标是xywh，而6d姿态估计预测的是xyzuvw（可以其他表示姿态的方式），因此基于目标检测框架来进行姿态估计也是没有问题的。



## 怎么表示物体的姿态

物体的姿态可以用旋转矩阵R和平移向量t来表示。直接回归t没有问题，但是直接回归旋转矩阵R就比较麻烦，因为旋转矩阵需要满足单位正交的条件，网络回归的结果很难会满足这种限制，因此需要考虑用其他方式来表示旋转。本文提出使用李代数来表示旋转。



## 方法

一个物体的回归目标用一个4维向量来表示，前三个元素表示与旋转有关的李代数，最后一个元素表示平移向量中的z。注意，xy是不用回归的，因为我们可以通过目标检测得到2D bounding box的中心坐标，由这个中心坐标映射到相机的坐标系就可以得到3D坐标。

直接预测xy是不容易的：

考虑这样一种情况，如果两个物体的z大小一致，但是xy不太一样，这样将3D物体投影到2D图像上之后，从图像上看表面非常相似，大小也相似，只是会在图片中的不同位置。因此很难通过2D图像中的物体来预测xy。

欧拉角的问题：

1. 用2π弧度表示，一个角可以用多个值表示。
2. 众所周知的万向锁的问题。

四元数：

必须要求4维向量为一个单位向量，这个限制也不利于优化。

这篇文章用李代数so(3)来表示旋转，只需要3个值就可以了。

将GT 旋转矩阵映射到Rodrigues logarithm mapping。

## 模型

![1534312272666](pics/1534312272666.png)

模型的架构和mask RCNN是一致的，只是多了一个pose预测分支。我们先来分别看每个分支。

class和box分支：这两个分支是公用一个head的，共用一个head代表分类和回归任务之间的gap比较小，本质上做的事情很像，所以可以共用一个分支。

Mask分支：mask分支和MaskRCNN的网络设计是一样的，唯一不同的是，输出的mask是与类别无关的，只是一个二值mask。而Mask RCNN中，mask的预测是和类别相对应起来的。为什么要这么做？我觉得这是因为mask预测在6d pose预测中是一个完全没用的东西，但作者仍然保留下来。既然没什么用，但还保留下来，好吧，那就减少mask预测的复杂度，不管对于哪一类物体，只要预测mask就好了。。。

Pose分支：pose分支是本文新增的分支。与mask分支不同的是，pose预测是和类别有关的。也就是说每个类别的物体，都会对应一个pose预测。因此pose预测的输出维度是4×num_class。为什么这么做？我觉是因为pose我们的最根本任务就是预测pose，那么我pose预测针对每个类别来做，会更专业化，效果也会更好。这种做法在很多论文里都有这么做。
