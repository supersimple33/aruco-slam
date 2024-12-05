"""Script written by Ed Towney to calibrate using Charuco boards in OpenCV.
https://medium.com/@ed.twomey1/using-charuco-boards-in-opencv-237d8bc9e40d
"""

import os

import cv2
import numpy as np

# ------------------------------
# ENTER YOUR REQUIREMENTS HERE:
ARUCO_DICT = cv2.aruco.DICT_5X5_250
SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 7
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
# ...
PATH_TO_YOUR_IMAGES = "./images"

# IMAGE_SIZE = (960, 540)
IMAGE_SIZE = (1920, 1080)
DISPLAY_SIZE = (960, 540)
# ------------------------------

def correct_rotation(image):
    h, w, _ = image.shape
    if h > w:
        image = cv2.transpose(image)
        image = cv2.flip(image, 1)
    return image

def calibrate_and_save_parameters():
    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Load PNG images from folder
    image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".jpg")]
    image_files.sort()  # Ensure files are in order

    all_charuco_corners = []
    all_charuco_ids = []

    for image_file in image_files:
        print(image_file)
        image = cv2.imread(image_file)
        image = correct_rotation(image)
        image = cv2.resize(image, IMAGE_SIZE)

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)

        # If at least one marker is detected
        if len(marker_ids) > 5:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

    # # Calibrate camera
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image.shape[:2], None, None)

    # # Save calibration data
    np.save("camera_matrix.npy", camera_matrix)
    np.save("dist_coeffs.npy", dist_coeffs)

    print("Camera Matrix:")
    print(camera_matrix)

    print("Distortion Coefficients:")
    print(dist_coeffs)

    # Iterate through displaying all the images
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = correct_rotation(image)

        image = cv2.resize(image, IMAGE_SIZE)

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
        image = cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

        undistorted_image = cv2.resize(undistorted_image, DISPLAY_SIZE)
        cv2.imshow("Undistorted Image", undistorted_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

calibrate_and_save_parameters()
