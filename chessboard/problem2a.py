import cv2
import apriltag

# Load the image in grayscale
image = cv2.imread("chessboard/frame_0.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the AprilTag detector
detector = apriltag.Detector()

# Detect tags in the image
tags = detector.detect(gray)

# Check if any tags were found
if not tags:
    print("No AprilTags detected.")
else:
    # Loop through all detected tags
    for tag in tags:
        # Check if this tag has ID 0
        if tag.tag_id == 0:
            print("AprilTag ID 0 detected")
            
            # Extract the corners of the tag
            corners_2D = tag.corners
            print("Corner coordinates (in image coordinates):")
            
            for i, corner in enumerate(corners_2D):
                print(f"Corner {i}: (x={corner[0]}, y={corner[1]})")
                cv2.circle(image, (int(corner[0]), int(corner[1])), 10, (255, 0, 0), 5)
                
            
        
                
          
            
            # Draw the detected tag on the image for visualization
            for i in range(4):
                start_point = tuple(corners_2D[i].astype(int))
                end_point = tuple(corners_2D[(i+1) % 4].astype(int))
                cv2.line(image, start_point, end_point, (255, 0, 0), 2)
            
            save_path = "output_image.jpg"
            cv2.imwrite(save_path, image)
            cv2.namedWindow("AprilTag Detection", cv2.WINDOW_NORMAL)
            # Display the result
            cv2.imshow("AprilTag Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
    else:
        print("AprilTag with ID 0 was not found.")