# forked from https://github.com/TUMFTM/GMMCalib

import open3d as o3d
import pickle
import yaml

def generate_data(data_path, config_file_path, sequence):
    # Read the parameters from the YAML file
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    number_of_sensors = config_data.get("number_of_sensors", "")

    poses_filepath = data_path + '/poses.pkl'

    openfile = open(poses_filepath, "rb")
    T_egos = pickle.load(openfile)
    T_f_init = pickle.load(openfile)
    T_r_init = pickle.load(openfile)
    openfile.close()
    
    sensors = [data_path + "/sensor_" + str(i+1) + "/" for i in range(number_of_sensors)]

    pcds = []
    for sensor in sensors:
        for frame in range(sequence[0], sequence[-1]+1):
            pcd = o3d.io.read_point_cloud(sensor + str(frame) + ".pcd")
            pcds.append(pcd)
        
    return pcds, T_egos, T_f_init, T_r_init