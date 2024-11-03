import cv2
import apriltag
import gtsam
import numpy as np

# Load the calibration matrix from Problem 1 results
fx, fy, s, px, py = 1.46734104e+03, 1.46230880e+03, 0, 5.45234279e+02, 9.43926077e+02  
# Replace with your actual values from calibration

# Initialize camera calibration in GTSAM
K = gtsam.Cal3_S2(fx, fy, s, px, py)

# Load image and detect AprilTag
image = cv2.imread('frame/frame_0.jpg', cv2.IMREAD_GRAYSCALE)
detector = apriltag.Detector()
detections = detector.detect(image)

# Check if the AprilTag with ID 0 is detected
observed_2D = []
for det in detections:
    if det.tag_id == 0:
        # Extract corner points in the specified order: lower-left, lower-right, upper-right, upper-left
        observed_2D = [gtsam.Point2(pt[0], pt[1]) for pt in det.corners]
        break

if not observed_2D:
    raise ValueError("AprilTag with ID 0 not found in the image")

# Define AprilTag 3D points in the tag's body-centric coordinate system
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
noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # Example: standard deviation of 1 pixel

# Add projection factors for each corner point
for i, (point_3D, point_2D) in enumerate(zip(points_3D, observed_2D)):
    graph.add(gtsam.GenericProjectionFactorCal3_S2(point_2D, noise_model, gtsam.symbol('x', 0), point_3D, K))

# Provide an initial estimate for the camera pose
initial_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0.5))  # Guess the camera is 0.5 meters in front of the tag
initial_estimate.insert(gtsam.symbol('x', 0), initial_pose)

# Optimize the factor graph using Levenberg-Marquardt optimizer
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
result = optimizer.optimize()

# Extract and print the optimized camera pose relative to AprilTag
camera_pose = result.atPose3(gtsam.symbol('x', 0))
print("Estimated camera pose relative to AprilTag:")
print(camera_pose)

# Display detected corners on the image (for visualization purposes)
for corner in observed_2D:
    cv2.circle(image, (int(corner.x()), int(corner.y())), 5, (0, 255, 0), -1)

# Show the result
cv2.imshow('AprilTag Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()