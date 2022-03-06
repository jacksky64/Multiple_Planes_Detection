from string import hexdigits
from turtle import width
from utils import *
import open3d as o3d
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
import math

def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    """ Detect multiple planes from given point clouds

    Args:
        points (np.ndarray): 
        min_ratio (float, optional): The minimum left points ratio to end the Detection. Defaults to 0.05.
        threshold (float, optional): RANSAC threshold in (m). Defaults to 0.01.

    Returns:
        [List[tuple(np.ndarray, List)]]: Plane equation and plane point index
    """

    plane_list = []
    N = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        w, index = PlaneRegression(
            target, threshold=threshold, init_n=3, iter=iterations)
    
        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)

    return plane_list

# examples/Python/Advanced/outlier_removal.py

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def remove_outliers(pcd):
    print("Downsample the point cloud with a voxel of 0.01")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size = 0.01)
    o3d.visualization.draw_geometries([voxel_down_pcd])

    print("Every 50th points are selected")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points = 50)
    o3d.visualization.draw_geometries([uni_down_pcd])

    print("Statistical oulier removal")
    cl,ind = voxel_down_pcd.remove_statistical_outlier( nb_neighbors=20, std_ratio=2.0)
    display_inlier_outlier(voxel_down_pcd, ind)

    print("Radius oulier removal")
    cl,ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    display_inlier_outlier(voxel_down_pcd, ind)
    return

def draw_geom(geometries, pose):
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    viewer.create_window(width=500,height=500)
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])

    view_ctl = viewer.get_view_control()
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    cam.extrinsic = pose 
    view_ctl.convert_from_pinhole_camera_parameters(cam)

    viewer.run()
    viewer.destroy_window()
    return 


def build_mesh(pcd, voxel_size = 0.01, radii = [0.005, 0.01, 0.02, 0.04]):
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size = voxel_size)
    voxel_down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(voxel_down_pcd, o3d.utility.DoubleVector(radii))
    return rec_mesh, voxel_down_pcd
#    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

def pc_offset(pcd):
    mpcd = o3d.geometry.PointCloud()
    mpcd.colors = copy.deepcopy(pcd.colors)
    mpcd.points = copy.deepcopy(pcd.points)
    p = np.asarray(mpcd.points)
    p[0:,1] = p[0:,1] + 0.5
    mpcd.points = o3d.utility.Vector3dVector(p)
    
    return mpcd

def geom_transform(pcd):
    T = np.eye(4)
    T[:3, :3] = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0 ))
    T[2, 3] = -1.5
    print(T)
    pcd_t = copy.deepcopy(pcd).transform(T)
    return pcd_t

def test_crop(point_cloud):
        
    start_position = {'x': 0., 'y': 0.1, 'z': 0.}
    cuboid_points = getCuboidPoints(start_position)

    points = o3d.utility.Vector3dVector(cuboid_points)
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
    oriented_bounding_box.color = np.asarray([1.,0,0])
    point_cloud_crop = point_cloud.crop(oriented_bounding_box)

    pose2 = np.eye(4)
    pose2[:3, :3] = point_cloud.get_rotation_matrix_from_xyz((np.pi, 0, 0 ))
    pose2[2, 3] = 1.5        
    draw_geom([point_cloud, oriented_bounding_box], pose2)

    # View original point cloud with the cuboid
    o3d.visualization.draw_geometries([point_cloud, oriented_bounding_box],width=500,height=500)

    # View cropped point cloud with the cuboid
    o3d.visualization.draw_geometries([point_cloud_crop, oriented_bounding_box],width=500,height=500)

    # height hist
    z=np.asarray(point_cloud_crop.points)[:,2]
    d = z - np.min(z)
    plot_histogram(d)

    colors = np.c_[d/np.max(d) ,np.zeros((d.shape[0],2))]
    point_cloud_crop.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_crop, oriented_bounding_box],width=500,height=500)
    return point_cloud_crop

def plot_histogram(d):
    
    n, bins, patches = plt.hist(x=d, bins='auto', cumulative=False, density=True, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='xy', alpha=0.75)
    plt.xlabel('z')
    plt.ylabel('Frequency')
    plt.title('height distribution')
    maxfreq = n.max()
  
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 0.1) * 0.1 if maxfreq % 0.1 else maxfreq + 0.1)
    plt.show()

def getCuboidPoints(start_position):
    CUBOID_EXTENT_METERS = 0.4
    
    METERS_BELOW_START = 1.5
    METERS_ABOVE_START = -0.5
    return np.array([
        # Vertices Polygon1
        [start_position['x'] + (CUBOID_EXTENT_METERS / 2), start_position['y'] + (CUBOID_EXTENT_METERS / 2), start_position['z'] + METERS_ABOVE_START], # face-topright
        [start_position['x'] - (CUBOID_EXTENT_METERS / 2), start_position['y'] + (CUBOID_EXTENT_METERS / 2), start_position['z'] + METERS_ABOVE_START], # face-topleft
        [start_position['x'] - (CUBOID_EXTENT_METERS / 2), start_position['y'] - (CUBOID_EXTENT_METERS / 2), start_position['z'] + METERS_ABOVE_START], # rear-topleft
        [start_position['x'] + (CUBOID_EXTENT_METERS / 2), start_position['y'] - (CUBOID_EXTENT_METERS / 2), start_position['z'] + METERS_ABOVE_START], # rear-topright

        # Vertices Polygon 2
        [start_position['x'] + (CUBOID_EXTENT_METERS / 2), start_position['y'] + (CUBOID_EXTENT_METERS / 2), start_position['z'] - METERS_BELOW_START],
        [start_position['x'] - (CUBOID_EXTENT_METERS / 2), start_position['y'] + (CUBOID_EXTENT_METERS / 2), start_position['z'] - METERS_BELOW_START],
        [start_position['x'] - (CUBOID_EXTENT_METERS / 2), start_position['y'] - (CUBOID_EXTENT_METERS / 2), start_position['z'] - METERS_BELOW_START],
        [start_position['x'] + (CUBOID_EXTENT_METERS / 2), start_position['y'] - (CUBOID_EXTENT_METERS / 2), start_position['z'] - METERS_BELOW_START],
        ]).astype("float64") 


def uniformSampler(x,y,z, minx, maxx, miny, maxy, xSpace, ySpace) :
    interpolator = interp.CloughTocher2DInterpolator(np.array([x,y]).T, z)
    nxSample =  math.ceil(np.abs(maxx-minx)/xSpace)
    nySample =  math.ceil(np.abs(maxy-miny)/ySpace)

    xline = np.linspace(minx, maxx, nxSample)
    yline = np.linspace(miny, maxy, nySample)
    xgrid,ygrid = np.meshgrid(xline, yline)
    # interpolate z data; same shape as xgrid and ygrid
    z_interp = interpolator(xgrid, ygrid)
    xgrid,ygrid,z_interp = np.reshape(xgrid,-1),np.reshape(ygrid,-1),np.reshape(z_interp,-1)
    valid_point = (np.isnan(z_interp)==False) & (z_interp>=np.min(z)) & (z_interp<=np.max(z))

    return xgrid[valid_point],ygrid[valid_point],z_interp[valid_point]



if __name__ == "__main__":
    import random

    #  draw open3d Coordinate system 
    #axis_pcd = o3d.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])
    #  stay 3D Draw points on coordinates ： Coordinates [x,y,z] Corresponding R,G,B Color 
    points = np.array([[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = [[0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    test_pcd = o3d.geometry.PointCloud()  #  Defining point clouds 
    #  Method 2（ Blocking display ）： call draw_geometries Directly display the point cloud data to be displayed 
    test_pcd.points = o3d.utility.Vector3dVector(points)  #  Define the point cloud coordinate position 
    test_pcd.colors = o3d.utility.Vector3dVector(colors)  #  Define the color of the point cloud 
   

    pcd = ReadPointCloud('Data/test_camera3d_centered_bin.ply')

    rec_mesh, voxel_down_pcd = build_mesh(pcd, voxel_size = 0.02)
    uniform_point_cluod = o3d.geometry.TriangleMesh.sample_points_uniformly(rec_mesh, number_of_points=2500)
    #voxel_down_pcd.paint_uniform_color([1, 0, 0])
    pose = np.eye(4)
    pose[:3, :3] = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0 ))
    pose[2, 3] = 1.5   
    draw_geom([rec_mesh, uniform_point_cluod],pose)


    point_cloud_crop = test_crop(pcd)

    # uniform sampling
    cropped_points = np.asarray(point_cloud_crop.points)
    x,y,z = cropped_points[0:,0],cropped_points[0:,1],cropped_points[0:,2]
    xSpace, ySpace = 0.01,0.01
    xgrid, ygrid, z_interp = uniformSampler(x,y,z, np.min(x), np.max(x), np.min(y), np.max(y), xSpace, ySpace) 

    # build uniform sampled point cloud
    xyz = np.zeros((np.size(xgrid), 3))
    xyz[:, 0] = np.reshape(xgrid, -1)
    xyz[:, 1] = np.reshape(ygrid, -1)
    xyz[:, 2] = np.reshape(z_interp, -1)
    uniform_pcd = o3d.geometry.PointCloud()
    uniform_pcd.points = o3d.utility.Vector3dVector(xyz)
    uniform_pcd.paint_uniform_color([0, 0, 1])

    # show uniform sampling
    draw_geom([rec_mesh, uniform_pcd],pose)
    draw_geom([uniform_pcd, point_cloud_crop],pose)

    thoraxHeight = np.mean(z_interp[(z_interp>(np.quantile(z_interp,0.95))) & (z_interp<(np.quantile(z_interp,0.99)))])-np.min(z_interp)   

    d = z_interp - np.min(z_interp)
    plot_histogram(d)

    pose = np.array([ 1.        ,  0.        ,  0.        , 0,
                        -0.        , -1.        , -0.        , 0.0,
                        -0.        , 0.        , -1.        , 1.5,
                        0.        ,  0.        ,  0.        ,  1.]).reshape(4,4)

    pose2 = np.eye(4)
    pose2[:3, :3] = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0 ))
    pose2[2, 3] = 1.5        
    draw_geom([pcd], pose)
    
    pose = np.array([ 1.        ,  0.        ,  0.        , 0,
        -0.        , -1.        , -0.        , 0.0,
        -0.        , 0.        ,  1.        , 4,
        0.        ,  0.        ,  0.        ,  1.]).reshape(4,4)
    draw_geom([pcd], pose)
    

    

    pcd_t = geom_transform(pcd)

    DrawPointCloud(pcd_t)

    pfs = pc_offset(pcd)

    draw_geom([pcd, pfs],pose)

    rec_mesh, voxel_down_pcd = build_mesh(pcd, voxel_size = 0.02)
    voxel_down_pcd.paint_uniform_color([1, 0, 0])
    draw_geom([rec_mesh, voxel_down_pcd], pose)

    remove_outliers(pcd)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, test_pcd, mesh_frame ],width=500, height=500, window_name="Xray") # , zoom=0.5, front=[0.01,0,0], lookat=[0, 0, 2],  up=[0,0.01,0] )

    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(downpcd, depth=12, width=0, scale=1.1, linear_fit=False)[0]
    o3d.visualization.draw_geometries([poisson_mesh ])


    DrawPointCloud(pcd)
    
    DrawNormals(pcd,voxel_size=0.05)
    
    points = ReadPlyPoint('Data/test_camera3d_centered_bin.ply')

   
    # pre-processing
    #points = RemoveNan(points)
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    points = DownSample(points,voxel_size=0.003)
    points = RemoveNoiseStatistical(points, nb_neighbors=50, std_ratio=0.5)
    
    
    results = DetectMultiPlanes(points, min_ratio=0.05, threshold=0.005, iterations=2000)

    planes = []
    colors = []
    for _, plane in results:

        r = random.random()
        g = random.random()
        b = random.random()

        color = np.zeros((plane.shape[0], plane.shape[1]))
        color[:, 0] = r
        color[:, 1] = g
        color[:, 2] = b

        planes.append(plane)
        colors.append(color)
    
    planes = np.concatenate(planes, axis=0)
    colors = np.concatenate(colors, axis=0)
    DrawResult(planes, colors)

