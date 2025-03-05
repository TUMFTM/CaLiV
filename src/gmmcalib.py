# forked from https://github.com/TUMFTM/GMMCalib

import argparse
import os
import generatePCDs
import transformPCDs
from modelgenerator import jgmm
import create_gt
import numpy as np
import pickle
import open3d as o3d
import minimization as min

def calibrate(data_path, config_file_path, sequence):
    pcds, T_egos, T_f_init, T_r_init = generatePCDs.generate_data(data_path, config_file_path, sequence)
    Xin = create_gt.create_init_pc(box_size=(0.5, 0.5, 0.5), num_points=400) + np.array([0.0 - 0.25, 0.0 - 0.25, 0.0 -0.25])

    V = [np.array(cloud.points) for cloud in pcds]
    nObs = len(V)

    print("####### Perform Calibration and Model Generation. ########")
    X, _, AllT, _= jgmm(V=V, Xin=Xin, maxNumIter=100)
 
    T_1 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2)]
    T_2 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2, nObs)]

    X = X.transpose()
    pcd_x = o3d.geometry.PointCloud()
    pcd_x.points = o3d.utility.Vector3dVector(X)
    obb = pcd_x.get_oriented_bounding_box()
    obb_points = np.asarray(obb.get_box_points())

    T_Rs = []
    T_egos_reordered = []
    lidar_ids = []

    for i in range(nObs // 2):
        T_ego_1 = np.array(T_egos[i])
        T_ego_2 = np.array(T_egos[i + (nObs // 2)])

        T_Rs.append(T_1[i])
        T_Rs.append(T_2[i])

        T_egos_reordered.append(T_ego_1)
        T_egos_reordered.append(T_ego_2)

        lidar_ids.append(0)
        lidar_ids.append(1)

    k=10
    
    print("Minimization Iteration 1/2...")
    T_sts, T_L2_error = min.compute_lidar_errors(T_Rs, T_egos_reordered, lidar_ids, obb_points, k)
    T_f_sts = T_sts @ T_f_init

    print("Minimization Iteration 2/2...")
    T_L2_error = min.compute_lidar_errors_iter2(T_Rs, T_egos_reordered, lidar_ids, T_sts, obb_points, k)

    T_f_calib = np.linalg.inv(T_L2_error) @ T_f_sts
    T_r_calib = np.linalg.inv(T_L2_error) @ T_r_init

    print("T_f_calib:")
    print(T_f_calib)
    print("T_r_calib:")
    print(T_r_calib)

    with open("../output/gmmcalib_result.pkl", "wb") as f:
        pickle.dump(T_f_calib, f)
        pickle.dump(T_r_calib, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration script")
    parser.add_argument("--data_path", type=str, help="Path to data", default="../data")
    parser.add_argument("--config_file_path", type=str, help="Path to config file", default="../config/config.yaml")
    parser.add_argument("--sequence", nargs='+', type=int, help="Sequence sequence of pcds")

    args = parser.parse_args()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_path))
    config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.config_file_path))

    if args.sequence is None:
        sequence = list(range(1, len(os.listdir(str(data_path+"/sensor_1"))) + 1))
    else:
        sequence = args.sequence

    calibrate(data_path, config_file_path, sequence)
