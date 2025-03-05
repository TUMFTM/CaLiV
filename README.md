# CaLiV
LiDARto-Vehicle Calibration of Arbitrary Sensor Setups via Object Reconstruction

## Prerequisites
To begin, prepare the PCD files (in the sensor frame) for each sensor, following the example provided in the `/input` directory.  
The corresponding vehicle poses must be specified for each pcd in the `/input/ego_poses.yaml` file.  
For the poses, we expect the coordinate system (x-forward, y-left, z-up).  
The poses follow the schema  
    - x (translation)  
    - y (translation)  
    - z (translation)  
    - x (quat)  
    - y (quat)  
    - z (quat)  
    - w (quat)  

## Configuration
The config file currently has four entries:  
- **Initial Transformations:** Represents the initial roll, pitch, yaw angles as well as the translation vector of both sensors.  
- **Target position:** The parameters `min_bound` and `max_bound` define the global position of the target in all point clouds.

## Use CaLiV
A Dockerfile is provided for CaLiV:

You can build the Docker file with a defined `<tag>`: 

    docker build -t caliv:latest -f docker/Dockerfile .


Then run the calibration with: 

    docker run -v $(pwd)/output:/app/output -it  caliv:latest


The results, namely the corrected LiDAR transformations are saved in the `/output` path.