import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Find the Center of the frame for our Region of Interest (ROI)
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    roi_width, roi_height = 700, 500
    top_left_x = center_x - roi_width // 2
    top_left_y = center_y - roi_height // 2
    bottom_right_x = center_x + roi_width // 2
    bottom_right_y = center_y + roi_height // 2

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Define a region of interest (ROI) for better performance
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        # Find the largest contour (assume it's the hand)
        max_contour = max(contours, key=cv2.contourArea)

        # Create a convex hull around the largest contour
        hull = cv2.convexHull(max_contour, returnPoints=False)

        # Find convexity defects
        defects = cv2.convexityDefects(max_contour, hull)

        if defects is not None:
            finger_count = 0

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Calculate the angle between the fingers
                a = np.linalg.norm(np.array(start) - np.array(far))
                b = np.linalg.norm(np.array(end) - np.array(far))
                c = np.linalg.norm(np.array(start) - np.array(end))
                angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

                # Count as a finger if the angle is less than 90 degrees and the defect depth is significant
                if angle <= np.pi / 2 and d > 10000: # Was 10,0000
                    finger_count += 1
                    cv2.circle(roi, far, 5, (0, 255, 0), -1)

            # Display the number of fingers
            cv2.putText(frame, f"Fingers: {finger_count + 1}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Finger Detection', frame)
    cv2.imshow('Threshold', thresh)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()