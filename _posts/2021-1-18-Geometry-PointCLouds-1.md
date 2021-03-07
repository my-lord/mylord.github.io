---
layout: post
title: "Open3D - PointCloud-1"
date: 2021-1-18 
description: "Open3D（Geometry->PointCloud）"
tag: Open3D 
---   



## PointCloud 补充篇

## 序言

 废话系列开始，正式从源码开始梳理Open3D。17年那会简略的过过一遍Open3D的源码，不过那个时候还没有成型，只有几个配准算法所以就被忽略了这么久。现在重新看Open3D已经具备了算法框架的形态，同时Python基因也无缝对接了tensorflow等深度学习框架。Open3D-ML分支也是拓展了3D点云在深度学习方向的应用。当然，这也是我最近关注Open3D的主要原因。

深度学习在图像处理的成功在于他能够在图像角点提取、边缘提取等传统算法的基础上将一个逻辑对象所拥有的所有角边缘特征整合在一起形成一个高维的特征描述子。然后在高维空间对不同逻辑对象进行聚类分割，进而实现高层次的逻辑对象识别。这样一个巨大的突破也是为什么18/19年滴滴AI实验室的负责人在review近20年的目标识别算法时将经典的角点、边缘等提取算法称为冷兵器，而深度学习后的目标识别开启了火药时代。

3D 识别所面临的困境与图像处理一致，需要根据全局特征来实现不同逻辑对象的区分。在多数点云分析中，仅根据固定邻域内点云分布形态能得到的分析能力也相对有限。而且很多逻辑对象其局部特征可能具备较强的相似度，因此仅根据局部特征很难实现高层次的逻辑对象识别功能。基于相同的需求，深度学习在3D点云上的运用也就应运而生。与图像不同的是，点云表达的信息隐藏在点与点之间相互位置关系内，而且缺少拓扑关系。因此，如何将在图像上表现优异的深度学习算法移植到点云上是当前众多学者的主要研究方向。

### Point Cloud　

Point Cloud类继承于Geometry3D ,Geometry3D又公有继承自Geometry。先简要介绍下这两个基类，详细介绍可以对呀的专题介绍页：

* Geometry :

> Geometry 类封装了一些几何类型标识，未标识、点云、体素、八叉树、线集、网格、三角网、半边三角网、图像、RGBD图像、四面体网格、OBB和AABB。
>
> Geometry定义了几何物体的维度和名称等信息。

* Geometry3D：

  > Geometry3D 定义了一些三维几何模型进行空间旋转变换的函数，包括对点和法向的旋转变尺度等操作。因为未指明几何模型类型，因此定义的都是虚函数/纯虚函数。

  ### 基类纯虚函数重写

PointCloud 类中重写了父类的一些纯虚函数，具体如下：

> Clear() : 清空点云、法向和颜色数组
>
> IsEmpty() : 判断点云是否为空
>
> GetMinBound() : 最小点
>
> GetMaxBound() : 最大点
>
> GetCenter() : 中心点
>
> GetAxisAlignedBoundingBox() : 计算包围盒的过程中使用了std::accumulate 函数，该函数定义为 
>
> ``` c++
> accumulate(InputIterator first , InputIterator last, T init , BinaryOperation binary_op)
> ```
>
> 其中，binary_op 传入回掉函数（int x,int y）,init  传入x, 前面的范围值传入y。
>
> GetOrientedBoundingBox():利用特征向量来计算OBB包围盒，计算过程中用到了std::iota()函数来批量递增赋值vector的元素。
>
>  ``` c++
> std::iota(vec.begin() , vec.end(),0)
>  ```
>
> 这个函数赋予了给vector 数组赋予序列的初值的功能（以前都是自己写代码赋值，感觉好蠢）。
>
> 顺便再来整合一下std::tuple() 和std::tie() 的使用方法，tuple 即元组可以理解为pair的扩展，用来将不同类型的元素组合在一起，常用来多返回值（弥补了我以前对多返回值的疑问）。通常采用 std::get<常量表达式>（tuple_name）来访问或者修改tuple里面的元素。  std::tie可以将变量的引用整合为一个tuple，从而实现批量赋值。
>
> VoxelDownSample() : 体素降采样。实现过程中采用了无序图的方式还是挺秀的但是有点看不懂，具体实现的时候可以参考一下。
>
> VoxelDownSampleAndTrace(): 带索引的点云降采样，每个格子自带存储在该体素内的点索引。
>
> RemoveRadiusOutliers() : 半径滤波；
>
> RemoveStatisticalOutliers() : 统计滤波，算法实现过程中用到了std::inner_product()计算矩阵的内积，该函数有两种形式：
>
> ​    （1）inner_product(beg1,end1,beg2,init):要求两个数组长度一致
>
> ​    （2）inner_product(beg1,end1,beg2,init,BinOp1,BinOp2):定义了两个回掉函数，第一个定义加，第二个定义乘。
>
> OrientNormalsConsistentTangentPlane(): 基于一致正切平面的法向一致定向方法，里面还用到了表面重建的过程，没仔细看，看起来挺复杂的。
>
> ComputeMahalanobisDistance() : 计算马氏距离。
>
> ComputeConvexHull() : 计算点云凸包。
>
> HiddenPointRemoval(): 移除视野内的隐藏点，利用视点和点云构造一个凸包，移除不与视点相关联的节点。
>
> SegmentPlane() : 从点云中分割平面，计算平面方程用的不是主成分分析，而是一种非常简约快速的方法，参考：[https://www.ilikebigbits.com/2015_03_04_plane_from_points.html](https://www.ilikebigbits.com/2015_03_04_plane_from_points.html)
>
> 

### 结语

Point cloud 类封装的函数大致包含：包围盒计算、法向估计、法向一致定向、重采样、平面分割和有深度图恢复点云等操作。Open3D代码实现对STL和C++11的使用频率相对较高，可以借助代码阅读过程中更好的理解STL中提供的函数使用方法。



2021.03.0  于上海