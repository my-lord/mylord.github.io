---
layout: post
title: "geometry->Geometry3d"
date: 2021-02-05
description: "Open3D（Geometry--Geometry and Geometry3D）"
---



## 前言

又到了愉快的废话系列，写帖子最大的乐趣莫过于可以把平时无处述说的废话放到这里。本着做事做完整的心态，把Point Cloud依赖的两个基类Geometry 和Gemetry3D进行简单的介绍。

## Geometry

该基类的主要工作是构造了数据类型、数据名称和维度三个信息，同时可提供了配置这些信息的接口函数。

* 数据类型

``` c++
 /// \brief Specifies possible geometry types.
    enum class GeometryType {
        /// Unspecified geometry type.
        Unspecified = 0,
        /// PointCloud
        PointCloud = 1,
        /// VoxelGrid
        VoxelGrid = 2,
        /// Octree
        Octree = 3,
        /// LineSet
        LineSet = 4,
        /// MeshBase
        MeshBase = 5,
        /// TriangleMesh
        TriangleMesh = 6,
        /// HalfEdgeTriangleMesh
        HalfEdgeTriangleMesh = 7,
        /// Image
        Image = 8,
        /// RGBDImage
        RGBDImage = 9,
        /// TetraMesh
        TetraMesh = 10,
        /// OrientedBoundingBox
        OrientedBoundingBox = 11,
        /// AxisAlignedBoundingBox
        AxisAlignedBoundingBox = 12,
    };
```

一共定义了十三种数据类型，包含一种未知类型的说明符。Geometry默认的构造函数是需要提供数据类型和维度信息，内置了两个纯虚函数：Clear（）、IsEmpty（）。看字面意思是清空和判断是否为空，具体实现要看子类的重构吧。



## Geometry3D

Geometry3D是三维模型的基类，在Geometry的基础上添加了一些三维空间变换、获取模型中点、最大/小点等函数接口。

* 新增纯虚函数接口

  GetMinBound（）、GetMaxBound（）、GetCenter（）、GetAxisAlignedBoundingBox（）、GetOrientedBoundingBox（）、Transform（）、Translate（const Eigen::Vector3d& translation , bool relative = true）（relative ：true 直接对所有点进行平移，False 对模型中心进行平移。这个思想挺好！！！）

  Scale（const double scale, const Eigen::Vector3d& center）:s(p - c) +c

  Rotate(const Eigen::Matrix3d& R , const Eigen::Vector3d& center) : R ( p - c ) + c

  Rotate(const Eigen::Matrix3d& R );

* 新增静态函数接口

  GetRotationMatrixFromXYZ(const Eigen::Vector3d& rotation)

  GetRotationMatrixFromYZX(const Eigen::Vector3d& rotation)

  GetRotationMatrixFromZXY(const Eigen::Vector3d& rotation)

  GetRotationMatrixFromXZY(const Eigen::Vector3d& rotation)

  GetRotationMatrixFromZYX(const Eigen::Vector3d& rotation)

  GetRotationMatrixFromYXZ(const Eigen::Vector3d& rotation)

  : 真是不嫌麻烦，把所有的可能进行组合。

  GetRotationMatrixFromAxisAngle(const Eigen::Vector3d& rotation)

  GetRotationMatrixFromQuaternion(const Eigen::Vector3d& rotation)

  : 如字面意思所示，还是挺齐全的。

* 基本的数据处理

  ComputeMinBound(const std::vector<Eigen::Vector3d>& points)

  ComputeMaxBound(const std::vector<Eigen::Vector3d>& points)

  ComputeCenter(const std::vector<Eigen::Vector3d>& points)

  ResizeAndPaintUniformColor(std::vector<Eigen::Vector3d>& colors , const size_t size , const Eigen::Vector3d& color)

  TransformPoints(const Eigen::Matrix4d& transformation, std::vector<Eigen::Vector3d>& points)

  TransformNormals(const Eigen::Matrix4d& transformation , std::vector<Eigen::Vector3d>& normals)

  : 这里用到了STL中的accumulate()函数，该函数有两种用法：

   1. 累加求和

      int sum = accumulate(vector.begin() , vector.end() ,  42（初值）)

  2. 自定义数据类型处理

     ``` c++
     minBound = std::accumulate(points.begin(),points.end(),points[0],
                               [](const Eigen::Vector3d& a, const Eigen::Vector3d& b){
                                 return a.array().min(b.array()).matrix();
                               });
     ```





----

2021.2.7   于佛山