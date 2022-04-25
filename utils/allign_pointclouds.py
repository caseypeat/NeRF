import numpy as np
import open3d as o3d
import  matplotlib.pyplot as plt
import copy

from matplotlib import cm


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4559,
    #                                   front=[0.6452, -0.3036, -0.7011],
    #                                   lookat=[1.9892, 2.0208, 1.8945],
    #                                   up=[-0.2779, -0.9482, 0.1556])

    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, source, target):
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


if __name__ == '__main__':

    thresh = 100

    points_front = np.load('./logs/priority3_real/20220423_152823/pointcloud/3000.npy')

    points_front, sigmas = points_front[..., :3], points_front[..., 3]
    points_front = points_front[np.broadcast_to(sigmas[..., None], (sigmas.shape[0], 3)) > thresh].reshape(-1, 3)
    sigmas = sigmas[sigmas > thresh][..., None]

    points_front = points_front[np.broadcast_to(points_front[..., 2, None], (points_front.shape[0], 3)) > -0.7].reshape(-1, 3)
    points_front = points_front[np.broadcast_to(points_front[..., 1, None], (points_front.shape[0], 3)) > -0.5].reshape(-1, 3)
    points_front = points_front[np.broadcast_to(points_front[..., 0, None], (points_front.shape[0], 3)) > -0.25].reshape(-1, 3)

    pcd_front = o3d.geometry.PointCloud()
    pcd_front.points = o3d.utility.Vector3dVector(points_front)

    pcd_front_d = pcd_front.uniform_down_sample(every_k_points=10)
    pcd_front_in, _ = pcd_front_d.remove_radius_outlier(nb_points=16, radius=0.05)

    points_back = np.load('./logs/priority3_real/20220423_164938/pointcloud/3000.npy')

    points_back, sigmas = points_back[..., :3], points_back[..., 3]
    points_back = points_back[np.broadcast_to(sigmas[..., None], (sigmas.shape[0], 3)) > thresh].reshape(-1, 3)
    sigmas = sigmas[sigmas > thresh][..., None]

    points_back = points_back[np.broadcast_to(points_back[..., 2, None], (points_back.shape[0], 3)) > -0.7].reshape(-1, 3)
    points_back = points_back[np.broadcast_to(points_back[..., 1, None], (points_back.shape[0], 3)) > -0.5].reshape(-1, 3)
    points_back = points_back[np.broadcast_to(points_back[..., 0, None], (points_back.shape[0], 3)) > -0.25].reshape(-1, 3)

    pcd_back = o3d.geometry.PointCloud()
    pcd_back.points = o3d.utility.Vector3dVector(points_back)

    pcd_back_d = pcd_back.uniform_down_sample(every_k_points=10)
    pcd_back_in, _ = pcd_back_d.remove_radius_outlier(nb_points=16, radius=0.05)

    voxel_size = 0.01

    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, pcd_front_in, pcd_back_in)

    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    print(result_ransac.transformation)

    # draw_registration_result(source, target, result_ransac.transformation)

    np.save('./data/bothsides_transform.npy', result_ransac.transformation)
    print(result_ransac.transformation.T)

    a = np.ones((points_front.shape[0], 1))
    print(a.shape, points_front.shape, result_ransac.transformation.shape)

    points_front_h = np.concatenate((points_front, a), axis=-1)

    # points_front_t = points_front_h
    # points_front_t = points_front_h @ result_ransac.transformation
    # points_front_t = points_front_h @ result_ransac.transformation
    points_front_t = (result_ransac.transformation @ points_front_h.T).T
    # points_front_t = points_front_h @ np.linalg.inv(result_ransac.transformation)

    print(points_front_t)

    # points = np.concatenate((points_front_t[..., :3]/points_front_t[..., 3, None], points_back), axis=0)

    pcd_f = o3d.geometry.PointCloud()
    pcd_f.points = o3d.utility.Vector3dVector(points_front_t[..., :3])
    # pcd_f.points = o3d.utility.Vector3dVector(points_front_t[..., :3]/points_front_t[..., 3, None])
    pcd_f.paint_uniform_color([1, 0, 0])

    print(points_back)

    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(points_back)
    pcd_b.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw([pcd_f, pcd_b])

    # draw_registration_result(source, target, result_ransac.transformation)
