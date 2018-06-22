# yolo v2 code study

## 本地安装

会出现一个报错：libstdc++.so.6: version `CXXABI_1.3.9' not found。经过很久的研究，发现是因为gcc和g++的版本过低造成的。其中一个帖子，如下，验证了确实是这个原因造成的。

https://blog.csdn.net/ccbrid/article/details/78979878

但是按照这个帖子并没有成功解决问题。

转而寻求其他的解决：https://askubuntu.com/questions/466651/how-do-i-use-the-latest-gcc-on-ubuntu/581497#581497

https://stackoverflow.com/questions/20357033/how-to-fix-program-name-usr-lib-x86-64-linux-gnu-libstdc-so-6-version-cxx

直接升级gcc和g++到4.9，这个问题就不再出现。

## k80 docker上安装

使用镜像：



ImportError: libgtk-x11-2.0.so.0: cannot open shared object file: No such file or directory

https://github.com/jupyter/docker-stacks/issues/228

cudaCheckError() failed : invalid device function

https://github.com/longcw/yolo2-pytorch/issues/15



## 代码细节

