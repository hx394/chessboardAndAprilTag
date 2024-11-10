import cv2

import gtsam
import numpy as np
import os




# Known 3D coordinates of the points in the world (e.g., in meters)
world_points = np.array([
    [-0.005, -0.005, 0],  # Lower-left corner
    [0.005, -0.005, 0],   # Lower-right corner
    [0.005, 0.005, 0],    # Upper-right corner
    [-0.005, 0.005, 0]  ]  # Upper-left corner
, dtype=np.float32)

# Detected 2D points in the image (pixel coordinates)
image_points = np.array([
[703.0389404295757, 541.346862793088],
[882.4196166991014, 534.966247558486],
[875.2590942383895, 358.52453613269824],
[696.3630371094946, 362.72912597667397]
], dtype=np.float32)

# Load the calibration matrix from Problem 1 results
fx, fy, s, px, py = 1.46425299e+03, 1.45896397e+03, 0, 5.40797911e+02, 9.46673188e+02  
# Replace with your actual values from calibration

# Initialize camera calibration in GTSAM
K = np.array([[fx,0,px],[0,fy,py],[0,0,1]],dtype=np.float32)

# Camera intrinsic parameters (from calibration)
camera_matrix = K
dist_coeffs = np.array([0.24908392, -0.99844089, -0.00290131,  0.00389668,  1.07896648],dtype=np.float32)  

# Estimate the pose using solvePnP
success, rotation_vector, translation_vector = cv2.solvePnP(
    world_points, image_points, camera_matrix, dist_coeffs
)

if success:
    # rotation_vector and translation_vector represent the pose of the camera
    print("Rotation Vector:\n", rotation_vector)
    print("Translation Vector:\n", translation_vector)

    rotation_matrix,_=cv2.Rodrigues(rotation_vector)
    print("Rotation Matrix:\n",rotation_matrix)
    
    rotation=gtsam.Rot3(rotation_matrix)
    translation=gtsam.Point3(translation_vector[0][0],translation_vector[1][0],translation_vector[2][0])
    pose=gtsam.Pose3(rotation,translation)



    print("Estimated camera pose relative to AprilTag:")
    print(pose)
else:
    print("fail to solve PnP")

