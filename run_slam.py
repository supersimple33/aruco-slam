import os
import cv2
import numpy as np
import time

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from aruco_slam import ArucoSlam
import viewer_3d as v3d
import viewer_2d as v2d

CALIB_MTX_FILE = 'calibration/calib_mtx.npy'
DIST_COEFFS_FILE = 'calibration/calib_dist.npy'

VIDEO_FILE = 'video.mp4'

IMAGE_SIZE = 1920, 1080
DISPLAY_SIZE = 960, 540

def load_matrices():
    # assert that the camera matrix and distortion coefficients are saved
    assert os.path.exists(CALIB_MTX_FILE), \
        'Camera matrix not found. Run calibration.py first.'
    assert os.path.exists(DIST_COEFFS_FILE), \
        'Distortion coefficients not found. Run calibration.py first.'
    
    calib_matrix = np.load(CALIB_MTX_FILE)
    dist_coeffs = np.load(DIST_COEFFS_FILE)

    return calib_matrix, dist_coeffs


def main():
    calib_matrix, dist_coeffs = load_matrices()

    # load the camera matrix and distortion coefficients, initialize the tracker
    initial_pose = np.array([0, 0, 0, 0, 0, 0]) # x, y, z, roll, pitch, yaw
    tracker = ArucoSlam(initial_pose, calib_matrix, dist_coeffs)

    # use the camera
    cap = cv2.VideoCapture(VIDEO_FILE)

    tracked_positions = []
    tracked_markers = []
    camera_markers = []

    camera_viewer_3d = v3d.Viewer3D(IMAGE_SIZE)
    image_viewer_2d = v2d.Viewer2D(calib_matrix, dist_coeffs)

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, state, marker_poses = tracker.process_frame(frame)

        camera_transform = state[:6]

        markers = state[6:].reshape(-1, 6)

        frame = cv2.resize(frame, IMAGE_SIZE)

        camera_viewer_3d.view(camera_transform, markers, marker_poses)

        frame = image_viewer_2d.view(frame, camera_transform, markers, marker_poses)

        frame = cv2.resize(frame, DISPLAY_SIZE)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return tracked_positions, tracked_markers, camera_markers

if __name__ == '__main__':
    positions, markers, camera_markers = main()