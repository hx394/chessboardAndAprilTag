import cv2
import numpy as np
import os
print("Current working directory:", os.getcwd())

# Chessboard dimensions (inner corners)
chessboard_size = (8, 6)

# Load the image
image = cv2.imread("C:/Users/rain_/Desktop/chessboardAndAprilTag/chessboard/originals/IMG_3918.JPEG")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#gray = cv2.equalizeHist(gray)

#gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)



# Detect chessboard corners
ret, corners = cv2.findChessboardCorners(gray, chessboard_size,None)

# If corners are found, refine and draw them
if ret:
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # Draw and display the corners
    cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
    for corner in corners:
        
        cv2.circle(image, (int(corner[0][0]), int(corner[0][1])), 10, (255, 255, 0), 5)
    save_path = "output_image.jpg"
    cv2.imwrite(save_path, image)
    print(corners)
    cv2.namedWindow("Chessboard Corners", cv2.WINDOW_NORMAL)
    cv2.imshow("Chessboard Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Chessboard corners not found")