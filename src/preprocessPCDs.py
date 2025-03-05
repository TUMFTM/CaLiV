import numpy as np
import open3d as o3d
import pickle
import argparse
import yaml
import os
import glob
from scipy.spatial.transform import Rotation

min_ground_dist = 0.1
distance_threshold = 0.03
ransac_n = 3
num_iterations = 1000
radius=0.2
max_nn=30
camera_loc = [0.0, 0.0, 0.0]

def filter_pcd(pcd, bounds, min_area):
    min_bound = bounds[0]
    max_bound = bounds[1]

    roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    cropped_pcd = pcd.crop(roi)

    remaining_cloud = cropped_pcd

    planes_to_keep = []

    is_first_plane = True

    while len(remaining_cloud.points) >= ransac_n:
        _, inliers = remaining_cloud.segment_plane(distance_threshold=distance_threshold,
                                                             ransac_n=ransac_n,
                                                             num_iterations=num_iterations)
        
        if len(inliers) == 0:
            break
        
        inlier_cloud = remaining_cloud.select_by_index(inliers)

        inlier_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        inlier_cloud.orient_normals_towards_camera_location(np.array(camera_loc))
        inlier_cloud.normalize_normals()

        bbox = inlier_cloud.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()
        bbox_area = bbox_extent[0] * bbox_extent[1]

        mean_z = np.mean(inlier_cloud.points, axis=0)[2]

        if is_first_plane:
            is_first_plane = False
            ground_z = mean_z
            inlier_cloud = remaining_cloud.select_by_index(inliers)
        elif mean_z - ground_z >= min_ground_dist:
            inlier_cloud = remaining_cloud.select_by_index(inliers)
            planes_to_keep.append(inlier_cloud)
        
        remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)

    for plane in planes_to_keep:
        remaining_cloud += plane

    if not remaining_cloud.is_empty():
        bbox = remaining_cloud.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()
        bbox_area = bbox_extent[0] * bbox_extent[1]

        if bbox_area < min_area:
            return o3d.geometry.PointCloud()
        filtered_pcd, _ = remaining_cloud.remove_radius_outlier(nb_points=10, radius=0.5)
    else:
        filtered_pcd = o3d.geometry.PointCloud()

    return filtered_pcd

def transform_pcds(loc_lidar_f, rot_lidar_f, loc_lidar_r, rot_lidar_r, 
                   bounds, args):
    
    ego_poses_filepath = args.input + 'ego_poses.yaml'
    out_poses_filepath = args.output + 'poses.pkl'

    openfile = open(ego_poses_filepath, "rb")
    data_storage = yaml.safe_load(openfile)
    openfile.close()

    # translation matrix of front lidar
    T_f_init = np.eye(4)
    T_f_init[:3, :3] = Rotation.from_euler('xyz', rot_lidar_f, degrees=True).as_matrix()
    T_f_init[:3, 3] = np.array(loc_lidar_f)

    # translation matrix of rear lidar
    T_r_init = np.eye(4)
    T_r_init[:3, :3] = Rotation.from_euler('xyz', rot_lidar_r, degrees=True).as_matrix()
    T_r_init[:3, 3] = np.array(loc_lidar_r)

    lidar_f_count = 0
    lidar_r_count = 0

    valid_scans_f = []
    valid_scans_r = []

    for i in range(1, len(data_storage["sensor_1"])):
        filename_f = data_storage["sensor_1"][i][0]
        ego_pose = data_storage["sensor_1"][i][1]

        t = np.asarray(ego_pose[:3])
        quat = ego_pose[3:]docker run -v $(pwd)/output:/app/output -it  gmmcalib:latest
        R = Rotation.from_quat(quat).as_matrix()

        T_ego = np.eye(4)
        T_ego[:3, 3] = t
        T_ego[:3, :3] = R

        filepath_f = args.input + "sensor_1/" + filename_f
        pcd_f = o3d.io.read_point_cloud(filepath_f)

        T_f = T_ego @ T_f_init

        pcd_f.transform(T_f)
                
        filtered_pcd_f = filter_pcd(pcd_f, bounds, args.min_area)

        if not filtered_pcd_f.is_empty():
            lidar_f_count += 1
            new_filpath_f = args.output + 'sensor_1/' + str(lidar_f_count) + '.pcd'

            valid_scans_f.append((new_filpath_f, filtered_pcd_f, T_ego))

    for i in range(1, len(data_storage["sensor_2"])):
        filename_r = data_storage["sensor_2"][i][0]
        ego_pose = data_storage["sensor_2"][i][1]

        t = np.asarray(ego_pose[:3])
        quat = ego_pose[3:]
        R = Rotation.from_quat(quat).as_matrix()

        T_ego = np.eye(4)
        T_ego[:3, 3] = t
        T_ego[:3, :3] = R

        filepath_r = args.input + "sensor_2/" + filename_r
        pcd_r = o3d.io.read_point_cloud(filepath_r)

        T_r = T_ego @ T_r_init

        pcd_r.transform(T_r)
                
        filtered_pcd_r = filter_pcd(pcd_r, bounds, args.min_area)

        if not filtered_pcd_r.is_empty():
            lidar_r_count += 1
            new_filpath_r = args.output + 'sensor_2/' + str(lidar_r_count) + '.pcd'

            valid_scans_r.append((new_filpath_r, filtered_pcd_r, T_ego))

    max_len = len(valid_scans_f)
    len_r = len(valid_scans_r)

    if len_r < max_len:
        max_len = len_r

    T_egos_f = []
    T_egos_r = []

    for i in range(max_len):
        new_filepath_f, pcd_f, T_ego_f = valid_scans_f[i]
        new_filepath_r, pcd_r, T_ego_r = valid_scans_r[i]

        o3d.io.write_point_cloud(new_filepath_f, pcd_f)
        o3d.io.write_point_cloud(new_filepath_r, pcd_r)

        T_egos_f.append(T_ego_f)
        T_egos_r.append(T_ego_r)

    T_egos = T_egos_f + T_egos_r

    with open(out_poses_filepath, 'wb') as out_file:
        pickle.dump(T_egos, out_file)
        pickle.dump(T_f_init, out_file)
        pickle.dump(T_r_init, out_file)
        out_file.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    
    argparser.add_argument(
        '--config',
        default='../config/config.yaml',
        help='path of config yaml')
    argparser.add_argument(
        '--input',
        default='../input/',
        help='path of input data')
    argparser.add_argument(
        '--output',
        default='../data/',
        help='path of gmmcalib data')
    argparser.add_argument(
        '--min-area',
        default=0.3,
        type=float,
        help='minimum bound for expected target area (x-y plane)')
    args = argparser.parse_args()

    # Remove all files in ../data/sensor_1/
    for file in glob.glob("../data/sensor_1/*"):
        try:
            os.remove(file)
        except:
            pass

    # Remove all files in ../data/sensor_2/
    for file in glob.glob("../data/sensor_2/*"):
        try:
            os.remove(file)
        except:
            pass

    # Remove file: ../data/poses.pkl
    poses_file = "../data/poses.pkl"
    if os.path.exists(poses_file):
        try:
            os.remove(poses_file)
        except:
            pass

    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)

    transform_lidar_f = config_data.get("transform_sensor_1", "")[0]
    transform_lidar_r = config_data.get("transform_sensor_2", "")[0]
    min_bound = config_data.get("min_bound", "")[0]
    max_bound = config_data.get("max_bound", "")[0]

    loc_lidar_f = transform_lidar_f[3:]
    rot_lidar_f = transform_lidar_f[:3]
    loc_lidar_r = transform_lidar_r[3:]
    rot_lidar_r = transform_lidar_r[:3]
    bounds = [min_bound, max_bound]
    
    transform_pcds(loc_lidar_f, rot_lidar_f, loc_lidar_r, rot_lidar_r, bounds, args)