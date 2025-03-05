import numpy as np
from scipy.optimize import minimize
import transformPCDs

def compute_error(T1, T2, X):
    T1 = np.linalg.inv(T1)
    T2 = np.linalg.inv(T2)

    targets = np.array(X)

    rot_error = 0.0

    for target in targets:
        transformed_target_1 = T1[:3, :3] @ target
        transformed_target_2 = T2[:3, :3] @ target
        errors = transformed_target_1 - transformed_target_2
        rot_error += np.sum(np.square(errors))

    t_error = np.sum(np.square(T1[:3, 3] - T2[:3, 3])) * len(targets)

    return rot_error + t_error

def cost_function(errors, T_Rs, T_egos, lidar_ids, X, k):
    t_rel = errors[:3]
    roll_rel, pitch_rel, yaw_rel = errors[3:6]
    T_rel_error = transformPCDs.euler_to_homogeneous(roll_rel, pitch_rel, yaw_rel, t_rel)

    t_L2 = errors[6:9]
    roll_L2, pitch_L2, yaw_L2 = errors[9:12]
    T_L2_error = transformPCDs.euler_to_homogeneous(roll_L2, pitch_L2, yaw_L2, t_L2)

    T_globs = []

    for T_R, T_ego, lidar_id in zip(T_Rs, T_egos, lidar_ids):
        if lidar_id == 0:
            T_error_loc = np.linalg.inv(T_rel_error) @ T_L2_error
        else:
            T_error_loc = T_L2_error

        T_glob = T_R @ T_ego @ T_error_loc @ np.linalg.inv(T_ego)
        T_globs.append(T_glob)

    errors = []

    for i in range(len(T_globs) - 1):
        for j in range(i + 1, len(T_globs)):
            error = compute_error(T_globs[i], T_globs[j], X)
            errors.append(error)

    errors = np.array(errors)

    lower_threshold = np.percentile(errors, k)
    upper_threshold = np.percentile(errors, 100 - k)
    mask = (errors >= lower_threshold) & (errors <= upper_threshold)
    errors = errors[mask]

    return np.sum(errors)

def cost_function_simplified(errors, T_Rs, T_egos, lidar_ids, T_sts, X, k):
    t = np.zeros(3)
    roll, pitch, yaw = errors
    T_L_error = transformPCDs.euler_to_homogeneous(roll, pitch, yaw, t)

    T_globs = []

    for T_R, T_ego, lidar_id in zip(T_Rs, T_egos, lidar_ids):
        if lidar_id == 0:
            T_error_loc = np.linalg.inv(T_sts) @ T_L_error
        else:
            T_error_loc = T_L_error

        T_glob = T_R @ T_ego @ T_error_loc @ np.linalg.inv(T_ego)
        T_globs.append(T_glob)

    errors = []

    for i in range(len(T_globs) - 1):
        for j in range(i + 1, len(T_globs)):
            errors.append(compute_error(T_globs[i], T_globs[j], X))

    errors = np.array(errors)

    lower_threshold = np.percentile(errors, k)
    upper_threshold = np.percentile(errors, 100 - k)
    mask = (errors >= lower_threshold) & (errors <= upper_threshold)

    errors = errors[mask]

    return np.sum(errors)

def compute_lidar_errors(T_Rs, T_egos, lidar_ids, X, k):
    errors_initial = np.zeros(12)

    result = minimize(cost_function, errors_initial, args=(T_Rs, T_egos, lidar_ids, X, k), method='Powell')

    t_rel = result.x[:3]
    roll_rel, pitch_rel, yaw_rel = result.x[3:6]
    T_rel_error = transformPCDs.euler_to_homogeneous(roll_rel, pitch_rel, yaw_rel, t_rel)

    t_L2 = result.x[6:9]
    roll_L2, pitch_L2, yaw_L2 = result.x[9:12]
    T_L2_error = transformPCDs.euler_to_homogeneous(roll_L2, pitch_L2, yaw_L2, t_L2)

    return T_rel_error, T_L2_error

def compute_lidar_errors_iter2(T_Rs, T_egos, lidar_ids, T_sts, X, k):
    errors_initial = np.zeros(3)

    result = minimize(cost_function_simplified, errors_initial, args=(T_Rs, T_egos, lidar_ids, T_sts, X, k), method='Powell')

    t = np.zeros(3)
    roll, pitch, yaw = result.x
    T_L_error = transformPCDs.euler_to_homogeneous(roll, pitch, yaw, t)

    return T_L_error

