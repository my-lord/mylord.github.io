
----
layout:post

title:"Open3d - Geometry -- Point clouds"

data:2021-1-8

description:"Open3D 笔记 "

tag: Open3D

----



### 点云读取

```python
import open3d as o3d
print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")
print(pcd)
print(np.asarray(pcd.points))

```

Open3D 提供了一系列点云文件读取的函数接口，包括读取点云文件、模型文件和图像。

> o3d.io.read_point_cloud(path)			读取点云文件
>
> o3d.io.read_triangle_mesh(path)		读取模型文件
>
> o3d.io.read_image(path)				 	读取照片文件

* Point Cloud

  |  格式  |                             描述                             |
  | :----: | :----------------------------------------------------------: |
  |  xyz   | Each line contains `[x, y, z]`, where `x`, `y`, `z` are the 3D coordinates |
  |  xyzn  | Each line contains `[x, y, z, nx, ny, nz]`, where `nx`, `ny`, `nz` are the normals |
  | xyzrgb | Each line contains `[x, y, z, r, g, b]`, where `r`, `g`, `b` are in floats of range `[0, 1]` |
  |  pts   | The first line is an integer representing the number of points. Each subsequent line contains`[x, y, z, i, r, g, b]`, where `r`, `g`, `b` are in `uint8` |
  |  ply   | See [Polygon File Format](http://paulbourke.net/dataformats/ply), the ply file can contain both point cloud and mesh data |
  |  pcd   | See [Point Cloud Data](http://pointclouds.org/documentation/tutorials/pcd_file_format.html) |

* Mesh

  |   格式   |                             描述                             |
  | :------: | :----------------------------------------------------------: |
  |   ply    | See [Polygon File Format](http://paulbourke.net/dataformats/ply/), the ply file can contain both point cloud and mesh data |
  |   stl    | See [StereoLithography](http://www.fabbers.com/tech/STL_Format) |
  |   obj    |  See [Object Files](http://paulbourke.net/dataformats/obj/)  |
  |   off    | See [Object File Format](http://www.geomview.org/docs/html/OFF.html) |
  | gltf/glb | See [GL Transmission Format](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0) |

* Image

  支持Jpg和png格式的图像文件。

### 点云显示	

```python
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
```

Open3D 的visualization 模块中定义了一系列的绘制点云的函数：

> draw_geometries()
>
> draw_geometries_with_custom_animation()
>
> draw_geometries_with_animation_callback()
>
> draw_geometries_with_key_callback()

### 体素下采样

```python
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
```

体素下采样的基本思路是：

> 将点云分割到各个体素空间内
>
> 以体素内点云的均值，作为该体素的下采样点

### 法向估计

 ```python
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
 ```

点云数据内置法向估计函数，输入的参数是邻域检索对象。计算后的法向存在normals数组中。邻域检索输入的两个参数是邻域radius范围内的max_nn 个参与邻域计算。相应的配合法向方向调整函数：

> orient_normals_to_align_with_direction()
>
> orient_normals_towards_camera_location()

### 点云分割

```python
print("Load a polygon volume and use it to crop the original point cloud")
vol = o3d.visualization.read_selection_polygon_volume(
    "../../test_data/Crop/cropped.json")
chair = vol.crop_point_cloud(pcd)
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])
```

visualization中提供了读取选择多边形区域兑现文件，然后利用该对象实现点云分割（其他的多边形文件对象应该也具备这个函数功能）。

### 点云着色

```python
print("Paint chair")
chair.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])
```

> paint_uniform_color()  : 为所有的点云赋予一个统一的颜色，颜色区间在[0,1]之间。

### 点间距计算

```python
# Load data
pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")
vol = o3d.visualization.read_selection_polygon_volume(
    "../../test_data/Crop/cropped.json")
chair = vol.crop_point_cloud(pcd)

dists = pcd.compute_point_cloud_distance(chair)
dists = np.asarray(dists)
ind = np.where(dists > 0.01)[0]
pcd_without_chair = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd_without_chair],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
```

> compute_point_cloud_distance() : 计算源点云每个点到目标点云的最近点距离。

### 包围盒计算

```python
aabb = chair.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = chair.get_oriented_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([chair, aabb, obb],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])
```

Open3D 提供了AABB和OBB两种包围盒计算方式。

###  凸包

```python
pcl = o3dtut.get_bunny_mesh().sample_points_poisson_disk(number_of_points=2000)
hull, _ = pcl.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([pcl, hull_ls])
```

 open3D利用hull库实现的点云凸包算法实现，就上述代码而言，凸包的计算过程是先对进行点云重采样（代码中的o3dtut好像不是点云对象，等后续搞明白了再补上，该降采样方法应该就是randla-net论文中提到的那个最远距离降采样方法）。

### DBSCAN点云聚类

```python
pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
```

在PointCloud类中实现的DBSAN点云聚类算法，返回点云标示，-1表示噪声。

### 平面分割

```python
pcd = o3d.io.read_point_cloud("../../test_data/fragment.pcd")
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                  zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
```

平面分割输入的参数是：点到平面距离、构造平面方程需要的点个数、迭代次数（用的是RANSAN的方法）。返回的是一个元组，第一个是平面参数，第二个是内点的索引。

### 隐藏点移除

```python
#将模型转化为点云，计算包围盒对角线长度
print("Convert mesh to a point cloud and estimate dimensions")
pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(5000)
diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
o3d.visualization.draw_geometries([pcd])

print("Define parameters used for hidden_point_removal")
camera = [0, 0, diameter]
radius = diameter * 100

print("Get all points that are visible from given view point")
_, pt_map = pcd.hidden_point_removal(camera, radius)

print("Visualize result")
pcd = pcd.select_by_index(pt_map)
o3d.visualization.draw_geometries([pcd])
```

hidden_point_removal 函数输入参数是相机位置、投影球体半径。返回一个元组包含模型文件的索引和未被遮挡的点云。



----

2021.1.8  于佛山