import cv2
import apriltag
import gtsam
import gtsam.noiseModel
import numpy as np
import os

R= [
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, -1.0
]
t=[  0, 0 , 0.1]

def X(i):
    return gtsam.symbol('x',i)

def L(j):
    return int(gtsam.symbol('o',j))
def P(k):
    return int(gtsam.symbol('p',k))

print("Current working directory:", os.getcwd())

# Load the calibration matrix from Problem 1 results
fx, fy, s, px, py = 1.46425299e+03, 1.45896397e+03, 0, 5.40797911e+02, 9.46673188e+02
# Replace with your actual values from calibration

# Initialize camera calibration in GTSAM
K = gtsam.Cal3_S2(fx, fy, s, px, py)
#print(K)
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



# Set up GTSAM factor graph
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()


# Define noise model for the projection factors
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # Example: standard deviation of 1 pixel


pose_prior=gtsam.Pose3(gtsam.Rot3(*R),gtsam.Point3(*t))
initial_estimate.insert(X(0), pose_prior)
pose_noise=gtsam.noiseModel.Diagonal.Sigmas(np.array(
    [1,1,1,1,1,1]))
graph.push_back(gtsam.PriorFactorPose3(X(0),pose_prior,pose_noise))

point_noise=gtsam.noiseModel.Isotropic.Sigma(3,0.000001)
# Insert 3D points (AprilTag corners) into initial estimate
for i, point_3D in enumerate(points_3D):



    
    factor=gtsam.PriorFactorPoint3(P(i),point_3D,point_noise)
    initial_estimate.insert(P(i),point_3D)
    graph.push_back(factor)
    
# Add projection factors for each corner point

# Define noise models


for i, (point_3D, point_2D) in enumerate(zip(points_3D, observed_2D)):
    
 
    
     factor=gtsam.GenericProjectionFactorCal3_S2(point_2D, measurement_noise, X(0), P(i), K)
    
     graph.push_back(factor)
    


# Provide an initial estimate for the camera pose
#initial_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0.5))  # Guess the camera is 0.5 meters in front of the tag
#initial_pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, -0.1, 0.1), gtsam.Point3(0.1, 0.1, 1.0))  # Vary initial estimate
#initial_estimate.insert(gtsam.symbol('x', 0), initial_pose)
#print("graph:")
#print(graph)
#print("initial estimate:")
#print(initial_estimate)
#initial_error = graph.error(initial_estimate)
#print("Initial error before optimization:", initial_error)
# Optimize the factor graph using Levenberg-Marquardt optimizer
params=gtsam.LevenbergMarquardtParams()
#params.setVerbosityLM("SUMMARY")
#params.setVerbosity("TERMINATION") 
#params.setVerbosityLM("TRYDELTA") 
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate,params)

#optimizer=gtsam.DoglegOptimizer(graph,initial_estimate)
result = optimizer.optimize()



#print("result:")
#print(result)
# Extract and print the optimized camera pose relative to AprilTag
camera_pose = result.atPose3(X(0))
print("Estimated camera pose relative to AprilTag:")
print(camera_pose)

print(optimizer.error())