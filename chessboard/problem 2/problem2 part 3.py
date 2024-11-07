import gtsam
import numpy as np
import cv2
import apriltag

R= [
        0.998356, -0.0376212, 0.0432434,
        -0.0349875, -0.997577, -0.0601261,
        0.0454007, 0.0585143, -0.997254
]
t=[  0.0135327, -0.0275992 , 0.0834204]

# Load the calibration matrix from Problem 1 results
fx, fy, s, px, py = 1467.34104, 1462.30880, 0, 545.234279, 943.926077  
# Replace with your actual values from calibration

# Initialize camera calibration in GTSAM
K = gtsam.Cal3_S2(fx, fy, s, px, py)


# Load image and detect AprilTag
image = cv2.imread('chessboard/frame/frame_0.jpg', cv2.IMREAD_GRAYSCALE)
detector = apriltag.Detector()
detections = detector.detect(image)

# Check if the AprilTag with ID 0 is detected
observed_2D = []
for det in detections:
    if det.tag_id == 0:
        # Extract corner points in the specified order: lower-left, lower-right, upper-right, upper-left
        # for pt in det.corners:
        #     observed_2D.append(np.array([[pt[0]], [pt[1]]], dtype=np.float64))
        # Extract corner points in the specified order: lower-left, lower-right, upper-right, upper-left
        observed_2D = [gtsam.Point2(pt[0], pt[1]) for pt in det.corners]
        observed_2D.reverse()
        #print(observed_2D)
        break

if not observed_2D:
    raise ValueError("AprilTag with ID 0 not found in the image")

# Define AprilTag 3D points in the tag's body-centric coordinate system
#scale_factor=fx
points_3D = [
    gtsam.Point3(-0.005, -0.005, 0),  # Lower-left corner
    gtsam.Point3(0.005, -0.005, 0),   # Lower-right corner
    gtsam.Point3(0.005, 0.005, 0),    # Upper-right corner
    gtsam.Point3(-0.005, 0.005, 0)    # Upper-left corner
]

# define the camera observation noise model
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
 
# create factor graph
graph = gtsam.NonlinearFactorGraph()
 
# get tag 0 corners in the first image
tag_0_corners = observed_2D#cpe.apriltag_corners['x0'][0]
corners_world = points_3D#gtg.landmark_ground_truth[0][0]
 
#print(corners_world)
#print(tag_0_corners)
 
# add a noise to world corners
#corners_world = corners_world + np.random.normal(0, 0.1, corners_world.shape)
#for i in range(len(corners_world)):
#    corners_world[i] = corners_world[i] + np.random.normal(0, 0.1)
#corners_world = corners_world + np.random.normal(0, 0.1)
 
# init values
initial_values = gtsam.Values()
 
# add measurement factor to the graph for each corner of the apriltag
for i in range(len(tag_0_corners)):
    graph.add(gtsam.GenericProjectionFactorCal3_S2(gtsam.Point2(tag_0_corners[i][0], tag_0_corners[i][1]), measurement_noise, gtsam.symbol('x', 0), gtsam.symbol('l', (0*10)+(i+1)), K))
    initial_values.insert(gtsam.symbol('l', (0*10)+(i+1)), gtsam.Point3(corners_world[i]))
 
# set prior factor on the camera pose
prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1, 1, 1, 1]))
#graph.add(gtsam.PriorFactorPose3(gtsam.symbol('x', 0), gtsam.Pose3(gtsam.Rot3.Rodrigues(cpe.camera_pose['x0'][0,:]), gtsam.Point3(cpe.camera_pose['x0'][1,:])), prior_noise_model))
graph.add(gtsam.PriorFactorPose3(gtsam.symbol('x', 0), gtsam.Pose3(gtsam.Rot3(*R),gtsam.Point3(*t)), prior_noise_model))
# initial value for camera pose
#initial_values.insert(gtsam.symbol('x', 0), gtsam.Pose3(gtsam.Rot3.Rodrigues(cpe.camera_pose['x0'][0,:]), gtsam.Point3(cpe.camera_pose['x0'][1,:])))
initial_values.insert(gtsam.symbol('x',0),gtsam.Pose3(gtsam.Rot3(*R),gtsam.Point3(*t)))
# save poses from graph before optimization
landmarks = []
point_noise=gtsam.noiseModel.Isotropic.Sigma(3,0.000001)
for i in range(4):
    landmarks.append(initial_values.atPoint3(gtsam.symbol('l', (0*10)+(i+1))))
    
    factor=gtsam.PriorFactorPoint3(gtsam.symbol('l', (0*10)+(i+1)),landmarks[i],point_noise)
    #initial_estimate.insert(P(i),landmarks[i])
    graph.push_back(factor)
camera_pose = initial_values.atPose3(gtsam.symbol('x', 0))
 
 
 
# optimize the graph
params = gtsam.LevenbergMarquardtParams()
#params.setVerbosityLM("SUMMARY")
#print(camera_pose)
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
result = optimizer.optimize()
print(result)