# faster RCNN代码阅读

1. 首先根据get_network(name)来获得特定的网络对象，net=VGGnet_train()。网络对象定义了tensorflow的图，可以看成是吃placeholder，吐output的一个黑盒。

2. net的初始化包括网络吃进来的placeholder，以及setup函数来进行网络结构的构建。

3. feed()的功能，将东西（可能多个）放到input列表（列表进行清空初始化）里面，喂给下一层。

4. layer装饰器

   实际干的事情就是给op搞出喂进去的东西，调用完op之后，在把这个op的output加入到self.layers，并且放入input列表中，供给下一层用。

5. py_func这个函数吃python的函数（这个函数是完全对numpy进行操作的），以及tensor，吐出来的tensor要经过tf.convert_to_tensor()转换？

6. anchor_target_layer这个函数搞出anchors的target，包括regression和分类的，用来算loss的时候使用。注意，由于输入的图片的大小有可能是不同的，所以这个函数也吃上一层吃进来的东西，主要是获取宽度和高度，再用来计算不同anchor所在的位置，这样就可以得到label。

7. proposal_layer这个函数，挑出fg，将相应的回归框进行nms（两次），得到最终的ROI。

8. proposal_target_layer这个函数，随机地选择一些proposal来进行train，因为有可能实在太多负样本了！

9. 构建完net之后，就进入solver进行train了。

   ​

## 代码阅读

1. 出现`libgtk-x11-2.0.so.0: cannot open shared object file`

~~~shell
apt-get install libgtk2.0-0
~~~

这个`apt-get install libgtk2.0-0:i386`不行

2. error libGL.so: cannot open shared object file: No such file or directory

~~~shell
apt-get install libgl1-mesa-dev
~~~

https://stackoverflow.com/questions/17417211/android-error-libgl-so-cannot-open-shared-object-file-no-such-file-or-direct

3. 出错：tensorflow.python.framework.errors_impl.NotFoundError: /root/anaconda2/bin/../lib/libstdc++.so.6: version `CXXABI_1.3.8' not found (required by /root/stone/TFFRCNN/lib/roi_pooling_layer/roi_pooling.so)

懒得再弄了...







