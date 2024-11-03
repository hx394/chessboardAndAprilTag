import cv2
import numpy as np
import glob

# Chessboard dimensions and square size in meters
chessboard_size = (8, 6)  # inner corners in each row and column
square_size = 0.01  # each square is 1 cm (0.01 meters)

# Prepare object points based on the known 3D structure of the chessboard
object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
object_points *= square_size

# Lists to store object points and image points from all images
object_points_list = []  # 3D points in real world space
image_points_list = []   # 2D points in image plane

# Load images
image_paths = glob.glob("C:/Users/rain_/Desktop/chessboardAndAprilTag/chessboard/originals/*.JPEG")  # Update with the correct path

# Loop over all images to detect chessboard corners
for image_path in image_paths:
    # Read image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # Refine corner positions for more accuracy
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # Add object points and image points
        object_points_list.append(object_points)
        image_points_list.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', image)
        cv2.waitKey(500)  # Display each result for 500 ms
    else:
        print(f"Could not find corners in image {image_path}")

cv2.destroyAllWindows()

# Perform camera calibration to get the camera matrix, distortion coefficients, etc.
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points_list, 
                                                                    image_points_list, 
                                                                    gray.shape[::-1], 
                                                                    None, None)

# Print the camera matrix and distortion coefficients
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
print("rotation vectors:\n",rvecs)
print("translation vectors:\n",tvecs)