# Faster RCNN code study

## 1. 骨架网路

骨架网络的选择可以多种，这里使用了ResNet50，ResNet100, MobileNetV2。

这里以ResNet为例进行讲解。

ResNet50或者ResNet101都可以分为5个block。前四个block用来作为骨架网络，用于特征提取，然后进行region proposal。最后一个block用做检测网络。

## 2. 阶段一：RPN

### 2.1. 生成anchor

**anchor的种类**

anchor的ratios（宽高比）有三种：0.5, 1, 2。anchor的scales有三种：0.5, 1, 2。因此总共有3×3=9个anchor。

三种scales的anchor的面积分别为：128x128，256x256，512x512。由此可计算出所有anchor的宽高。

**anchor的总数**

根据骨架网络的最后一层的feature_width和feature_height，总共可以生成feature_width×feature_heightx9个anchor(feature map上的每一个点都对应9个anchor)。

在代码的实现中，将所有的anchor用anchors表示，anchor的尺寸是all_Nx4，维度4表示xmin, ymin, xmax, ymax（注意这个是在原图尺度上的），维度all_N表示feature_width×feature_heightx9，没有任何过滤。

> 关于anchor的中心设置需要注意一下，在github上提出issue得到的解答是，anchor的中心，有两种选择，假设stride=32，左上角的anchor可以选择（0,0）也可以选择（16,16）。

### 2.2. 对RPN输出的处理

注意：对RPN输出的处理需要用到anchor，因为对输出进行转换的时候需要用到anchor的宽高信息。

**RPN输出**：rpn_box_pred ，rpn_cls_prob， 维度分别为[feature_width×feature_height, 4*9],

[feature_width×feature_height, 2x9]。对这个预测结果需要进行一些处理。

1. 这个输出是变换，而不是真正的bbox的预测值。所以需要将结果进行转换。

~~~python
# 转换到原图尺度上。
predict_xcenter = t_xcenter * reference_w + reference_xcenter
predict_ycenter = t_ycenter * reference_h + reference_ycenter
predict_w = tf.exp(t_w) * reference_w
predict_h = tf.exp(t_h) * reference_h
# 计算xmin,ymin,xmax,ymax
predict_xmin = predict_xcenter - predict_w / 2.
predict_xmax = predict_xcenter + predict_w / 2.
predict_ymin = predict_ycenter - predict_h / 2.
predict_ymax = predict_ycenter + predict_h / 2.
~~~

2. 将以上得到的结果进行裁剪，如果大于边界或者小于边界就裁剪掉。
3. 在NMS之前，先通过预测的前景概率值筛选出pre_nms_topN（12000）个结果。
4. 然后进行NMS，留下post_nms_topN（2000）个结果。注意这里进行NMS并没有根据类来做，因为类就只有一个。

由以上四步操作，得到保留下来的ROI 最大2000个，并且相应的分数也保留下来。

```python
# RPN的输出
rois[2000,4], roi_scores[2000] 
```

### 2.3. 生成anchor的训练目标

1. 首先由2.1得到的所有all_anchors，进行处理，超出边界的anchor的舍去不要，得到剩下的anchors[N, 4]。

2. 建立anchors的标签labels[N]，初始化其中的元素为-1.

3. 计算真实gt_boxes[M, 4]与anchor的IOU，得到一个IOU矩阵：max_overlaps[N,M]。

4. 对max_overlaps每行取最大值，这个值如果大于0.7，则给这个anchor打上1。如果小于0.3，就打上背景label 0。为了确保每个gt都有anchor来负责预测，需要计算每列的最大值，列最大值对应的那个anchor，就打上1。

5. 计算需要的前景的数量256x0.5，如果前景数量超过256，则随机选择多出的那部分前景anchor，设置label为-1。同样，计算背景anchor的数量，如果背景anchor的数量大于256-256x0.5，那么随机选择多出的部分，设置label -1。这样，对anchors的label的设置就完成了。

6. 现在要计算anchor的回归目标。对于每个anchor，根据max_overlaps中每行的最大值对应的gt，计算相应的回归目标。注意，这个时候计算的目标是gt和anchor之间的变换。

   ~~~python
   reference_w = reference_xmax - reference_xmin + 1e-8
   reference_h = reference_ymax - reference_ymin + 1e-8
   reference_xcenter = reference_xmin + reference_w/2.0
   reference_ycenter = reference_ymin + reference_h/2.0
   # w + 1e-8 to avoid NaN in division and log below

   t_xcenter = (x_center - reference_xcenter) / reference_w
   t_ycenter = (y_center - reference_ycenter) / reference_h
   t_w = np.log(w/reference_w)
   t_h = np.log(h/reference_h)
   ~~~

7. 最后，将anchors的结果，整合到all_anchors当中去。除了anchors部分，其余部分的labels都设置为-1。回归的部分值，则设置为0.

最终得到了：

~~~python
# anchor的训练目标
rpn_labels[num_all_anchors], rpn_bbox_targets[num_all_anchors,4]
~~~

需要特别注意的是，RPN产生的2000个ROI和anchor的训练目标没有本质上的联系。

## 3. 阶段二：fast RCNN

喂进去骨架网络的最后层的feature，以及RPN产生的2000个roi，进行ROI pooling，得到ROI pooling的结果，以及stage 2的训练目标。

### 3.1 proposal的训练目标

Proposal的训练目标，喂进去: RPN产生的2000个roi，还有gt：gtboxes_batch(1, Num_Of_objects, 5] ，维度5表示：[x1, y1, x2, y2, label]。

1. 考虑是否把gt也加到rois中进行训练。

2. 设置每张图片上roi的数量为256。

3. 对rois进行采样，选出256个。

   > 1. 计算真实gt_boxes[M, 4]与rois的IOU，得到一个IOU矩阵：overlaps[2000,M]。
   > 2. 计算overlaps中每一行的最大值对应的gt，则设置这个roi的label为这个gt的label。
   > 3. 对overlaps每行取最大值，这个值如果大于FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5则保留下来，得到该roi的index，集齐全部index得到fg_inds。如果小于这个值，进行类似操作得到bg_inds.
   > 4. 选择前景fg_inds的数量为总共的rois_per_image的0.25，剩余的则是bg_inds的数量。
   > 5. 计算保留下来的roi的训练target。
   > 6. 最终得到：rois[256,4], labels[256], bbox_targets[256, 4*(num_cls+1)]

RPN产生的2000个roi并不是都拿来训练，全都拿出来训练太多了，因此要经过筛选。

### 3.2 ROI pooling

喂进去骨架网络的最后层的feature，以及RPN产生的256个roi，通过tensorflow中的tf.image.crop_and_resize进行roi pooling，得到[256, 7, 7, channels]的结果。其中7是代表ROI pooling之后的大小。

### 3.3 检测

经过最后一个ResNet的block，这个block做的是检测的任务。得到结果：bbox_pred[256,4×num_classes], cls_score[256xnum_classes]。**注意，每个ROI并不是只预测一个回归结果，而是对每个类都进行回归预测**

### 3.4 检测结果后处理

**测试**

对于**每个类别**进行如下操作：

1. 将结果bbox_pred转换到原图尺度上。转换方式如2.2。
2. 将超过边界的bbox裁剪到边界。
3. 进行NMS。最多保留100个结果：rois,  roi_scores 

这个时候，每个类都有剩下ROI，最后再根据ROI的得分SHOW_SCORE_THRSHOLD = 0.5将得分小的结果进行滤除。

**注意，每个roi可以预测多个结果！**

**评估**

评估的时候，是不需要通过阈值将结果进行滤除去的，而是都保留着去计算mAP。



## 4. 损失函数







