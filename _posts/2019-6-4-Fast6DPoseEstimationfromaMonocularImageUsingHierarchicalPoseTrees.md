---
layout: post
title: "Fast 6D Pose Estimation from a Monocular Image Using Hierarchical Pose Trees"
date: 2019-6-4 
description: "单目图像的6D位姿估计"
tag: 3D视觉 
---   

# 主体思想介绍

<img src="/doc_imgae/2019-6-4/1.png">


这篇文章的主要工作是从二维图像中估计目标物体的三维位姿（6D信息，X、Y、Z、Rx、Ry、Rz）。
如上图所示，该方法可以在CPU上以平均150ms的效率实现单目图像上三维物体的位姿估计。这篇文章的主题思路如下：
> * <p>构造PCOF：Perspectively Cumulateed Orientation Feature特征</p>
> * <p>构造平衡位姿树 HPT:Hierarchical Pose Trees</p>

## PCOF：Perspectively Cumulateed Orientation Feature

<img src="/doc_imgae/2019-6-4/2.png">

PCOF 相当于模板图像，在三维模型中由于存在空间位置和姿态的变换，模板图像会不止一张。
而且，每张模板图像也是由多张三维模型的投影图像构造而成，该作者在另外一篇文章介绍的每个模板图像大概使用200张投影图。
具体的构造流程如下：

> * 如上图所示，对三维模型构造一个环绕的三角网格的球模型。网格的每个顶点表示一个视点的位置。（注意，此处仅仅确定位置）
> * 每个顶点处沿着光轴方向旋转5度和沿着光轴方向平移30mm处各构造一个模板T。（基础距离是到三维模型680m）
> * 每个模板位置处随机确定四个变换参数，然后对其进行投影得到投影图像。四个参数分别是：沿着x、y轴旋转（+-12度），到三维模型中心的距离（+-40mm），沿着光轴方向旋转（+-7.5度）。大概随机200次，获取200张投影图像。
> * 对每个投影图像求梯度（Sobel），生成COF描述图像。与COF不同的是，取每个像素的梯度方向数量大于一定阈值的主方向8bit描述子相应位置为1，否则为0.

<img src="/doc_imgae/2019-6-4/3.png">


## HPT:Hierarchical Pose Trees

<img src="/doc_imgae/2019-6-4/4.png">

平衡位姿树是为了加速模板匹配的过程。如上图所示，平衡位姿树的构造采用从下往上的过程。越往上，模板图像也就也少。检索的过程与之相反，采用由粗到精的过程。
构造过程可以分为：聚类、整合和降低分辨率三个步骤。具体步骤如下：
> * 首先，对所有的模板图像采用X-means 方法进行聚类。
> * 将相同聚类的模板图像整合到一个模板中。对相同像素的直方图进行累加和归一化。对处理后的模板图进行阈值化，得到新的特征和权重。
> * 将相邻2*2像素的直方图进行累加、归一化，探测新的特征和权重。
> * 重复以上步骤直到低分辨率模板图像T中包含的特征点个数少于一定的阈值Nmin(论文中介绍的是50).


## 实验结果

<img src="/doc_imgae/2019-6-4/5.png">
<img src="/doc_imgae/2019-6-4/6.png">
<img src="/doc_imgae/2019-6-4/7.png">
<img src="/doc_imgae/2019-6-4/8.png">





<br>
*转载请注明原地址，邓辉的博客：[https://github.com/my-lord/mylord.github.io](https://github.com/my-lord/mylord.github.io) 谢谢！*


<br>


Don't Panic.
-------------------------------------------
