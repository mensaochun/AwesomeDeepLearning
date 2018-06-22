# Code Study

## coco数据集

class COCO
数据成员的意义：

| 私有成员           | 意义                                       |
| -------------- | :--------------------------------------- |
| self.dataset   | 将annotation的json文件load到内存中，用dict表示。有'info' (139690875497920)，'licenses'，'images'，'type' (139690876792144) = {str}，'instances'，'annotations'，'categories' |
| self.anns      | dict：ann_id-->{'segmentation': [[382.48, 268.63, 330.24, 229.93, 278.97, 205.75, 228.66, 143.83, 214.15, 140.93, 225.76, 134.16, 257.69, 123.52, 277.03, 82.89, 328.3, 48.06, 433.75, 41.29, 502.43, 79.99, 561.44, 168.02, 547.9, 216.39, 562.41, 246.38, 542.09, 285.07, 510.17, 285.07, 467.61, 223.16, 419.24, 253.15, 394.09, 264.76]], 'area': 53481.5118, 'iscrowd': 0, 'image_id': 42, 'bbox': [214.15, 41.29, 348.26, 243.78], 'category_id': 18, 'id': 1817255} |
| self.cats      | cat_id-->{'supercategory': 'person', 'id': 1, 'name': 'person'}。注：总共80个类。实际上coco数据有91个类，由于有些物体不好分割，就没选用，只选了其中80个。 |
| self.imgs      | img_id-->{'license': 3, 'url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg', 'file_name': 'COCO_val2014_000000391895.jpg', 'height': 360, 'width': 640, 'date_captured': '2013-11-14 11:18:45', 'id': 391895} |
| self.imgToAnns | img_id-->multiple anns.                  |
| self.catToImgs | cat_id-->multiple img_ids                |

class CocoDataset

self.image_info

~~~python
{'id': 262145, 
 'source': 'coco', 
 'path': '/home/pi/data/coco/zipfile/train2014/COCO_train2014_000000262145.jpg', 'width': 640, 
 'height': 427, 
 'annotations': [{'segmentation': [[214.27, 79.01, 220.54, 82.35, 235.99, 91.54, 247.69, 97.39, 255.2, 101.15, 265.64, 109.08, 261.89, 98.22, 256.04, 86.53, 248.94, 79.85, 244.76, 74.0, 237.66, 67.73, 232.23, 65.65, 228.89, 63.56, 231.4, 55.62, 229.31, 56.04, 226.38, 62.72, 220.12, 60.63, 215.94, 60.63, 214.27, 61.05, 218.87, 66.06, 216.36, 70.66, 212.6, 75.67, 212.6, 78.59]], 'area': 970.0430999999998, 'iscrowd': 0, 'image_id': 262145, 'bbox': [212.6, 55.62, 53.04, 53.46], 'category_id': 28, 'id': 284647}, ...,]
~~~

self.class_info

~~~python
00 = {dict} {'source': '', 'id': 0, 'name': 'BG'}
01 = {dict} {'source': 'coco', 'id': 1, 'name': 'person'}
02 = {dict} {'source': 'coco', 'id': 2, 'name': 'bicycle'}
03 = {dict} {'source': 'coco', 'id': 3, 'name': 'car'}
04 = {dict} {'source': 'coco', 'id': 4, 'name': 'motorcycle'}
~~~

self.class_from_source_map

~~~python
'.0' (140486453530440) = {int64} 0
'coco.1' (140486453530328) = {int64} 1
'coco.2' (140486453529824) = {int64} 2
'coco.3' (140486453530048) = {int64} 3
'coco.4' (140486453526912) = {int64} 4
'coco.5' (140486453530216) = {int64} 5
'coco.6' (140486453529936) = {int64} 6
'coco.7' (140486453528592) = {int64} 7
'coco.8' (140486453528480) = {int64} 8
'coco.9' (140486453528760) = {int64} 9
'coco.10' (140486453528536) = {int64} 10
...
'coco.88' (140486453486568) = {int64} 78
'coco.89' (140486453486960) = {int64} 79
'coco.90' (140486453487128) = {int64} 80
~~~

self. image_from_source_map

~~~python
'coco.262145' (140483409391344) = {int64} 0
'coco.262146' (140483409367472) = {int64} 1
'coco.524291' (140483409367088) = {int64} 2
'coco.131074' (140483409367344) = {int64} 3
'coco.393221' (140483409369712) = {int64} 4
...
~~~

self.source_class_ids

~~~python
{
  '': [0],
 'coco': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]}
~~~





## 一、下载，安装

1. 项目下载地址：https://github.com/matterport/Mask_RCNN.git
2. 这个项目需要用到pycocotols。到这个网站进行下载：https://github.com/waleedka/coco

然后到coco/PythonAPI下运行：

~~~shell
make
~~~

进行编译。注意，如果不是使用系统默认python，那么要修改相应的python路径。

3. 下载coco数据训练好的预训练模型：https://github.com/matterport/Mask_RCNN/releases，直接放在project的root下。

4. 如果想用气球数据来训练我们的模型的话，可以到这里下载数据：https://github.com/matterport/Mask_RCNN/releases

   ​

## 二、Training

Training代码以coco数据集为例。

### 配置信息

配置信息：主要配置信息都在父类Config中，配置参数均为类变量。

有一些参数可以根据已有参数计算出来，包括batch_size,image_shape,image_meta_size，则在`__init__`中进行设置，如下：

~~~python
def __init__(self):
    """Set values of computed attributes."""
    # Effective batch size
    self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
    # Input image size
    if self.IMAGE_RESIZE_MODE == "crop":
        self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
    else:
        self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])
    # Image meta data length
    # See compose_image_meta() for details
    self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
~~~

如果我们有一个新的数据集，则可以继承Config类，然后添加一些新的成员或者覆盖成员变量。比如Coco数据集，如下：

~~~python
class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)  #FIXME
    NUM_CLASSES = 1 + 3  # COCO has 80 classes
~~~

### Model

模型部分+目标+损失+训练都在model里面定义了。

#### 1. build函数

build函数定义了training和test两个阶段的模型。我们这里只看training的模型。

build做的事情主要包括以下部分：

1. 网络输入。

|      |    input_image    |   input_rpn_match   | input_image_meta |    input_rpn_bbox     |
| ---- | :---------------: | :-----------------: | :--------------: | :-------------------: |
| 用途   |       输入的图像       | anchors输出的前景背景label |     输入图像的信息      | anchors输出的bbox的target |
| 形状   | [batch,1024,1024] | [batch,anchors, 1]  |    [batch,16]    |  [batch, anchors,4]   |

|      | input_gt_class_ids |   input_gt_boxes   |  input_gt_masks  |
| ---- | :----------------: | :----------------: | :--------------: |
| 用途   | Head部分的bbox类别label | Head部分的bbox的target | Head部分的bbox的mask |
| 形状   | [batch, ?,?]两个问号？  |    [batch,?,4]     | [batch,56,56,?]  |

#### 2. Resnet+FPN的输出

rpn_feature_maps：一个list，里面装了各层级的输出。

~~~python
 0 = {Tensor} Tensor("fpn_p2/BiasAdd:0", shape=(?, 256, 256, 256), dtype=float32)
 1 = {Tensor} Tensor("fpn_p3/BiasAdd:0", shape=(?, 128, 128, 256), dtype=float32)
 2 = {Tensor} Tensor("fpn_p4/BiasAdd:0", shape=(?, 64, 64, 256), dtype=float32)
 3 = {Tensor} Tensor("fpn_p5/BiasAdd:0", shape=(?, 32, 32, 256), dtype=float32)
~~~

mrcnn_feature_maps：一个list，里面装了多个层级的输出，比rpn_feature_maps多一个p6输出。

~~~python
 0 = {Tensor} Tensor("fpn_p2/BiasAdd:0", shape=(?, 256, 256, 256), dtype=float32)
 1 = {Tensor} Tensor("fpn_p3/BiasAdd:0", shape=(?, 128, 128, 256), dtype=float32)
 2 = {Tensor} Tensor("fpn_p4/BiasAdd:0", shape=(?, 64, 64, 256), dtype=float32)
 3 = {Tensor} Tensor("fpn_p5/BiasAdd:0", shape=(?, 32, 32, 256), dtype=float32)
 4 = {Tensor} Tensor("fpn_p6/MaxPool:0", shape=(?, 16, 16, 256), dtype=float32)
~~~

#### 3. anchors生成

输入：输入图像的大小→`image_shape`，anchor的尺度（正方形的边长）→`RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)`，anchor的长宽比→`RPN_ANCHOR_RATIOS = [0.5, 1, 2]` 。

输出：anchor字典，存在`self.anchor_cache`中，key为形状tuple，value为tensor。

所做的事情：

- 计算骨架网络的各个层级的feature map大小
- 生成anchors
- 将anchor进行normalize



### 生成proposal的target,label以及生成实例mask

target是bbox回归用的，label是bbox分类用的。mask是进行实例分割用的。

输入。输入是一张图片的所有ROI，还有这张图片中的所有bbox的类别，bbox坐标和实例的二值mask。

~~~python
# 如果proposal不够，补零。
proposals: [N, (y1, x1, y2, x2)] in normalized coordinates.
# 类别id，MAX_GT_INSTANCES表示一张图片中包含object的数量
gt_class_ids: [MAX_GT_INSTANCES], int type。
gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
gt_masks: [height, width, MAX_GT_INSTANCES]，boolean type.
~~~

输出。输出是

~~~python
Returns: Target ROIs and corresponding class IDs, bounding box shifts,and masks.
rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Class-specific bbox refinements.
masks: [TRAIN_ROIS_PER_IMAGE, height, width] Masks cropped to bbox
boundaries and resized to neural network output size.
~~~

步骤：

1. 首先，将proposal和gt中有0 padding的部分都给移除了。为什么会有0 padding，可能因为Batch中的image必须对齐。
2. 将label为-1的crowd box给移除了，crowd box包含了好几个实例。猜测是因为太难搞了。
3. 计算proposal和gt的overlap，也就是IOU，得到一个矩阵。
4. 给proposal打正负标签。正标签：与gt的IOU大于0.5的话，就是正样本，小于0.5并且不是crowd box就是负样本。
5. 对正负样本进行随机采样，正样本比例1/3，负样本比例为2/3。
6. 给正样本分配label和target。注意之前打的是正负label，这里是类别label。
7. 给正样本分配mask target。这里有一个选项，是否所有的mask都需要进行normalize。