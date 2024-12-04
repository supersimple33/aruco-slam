"""
The main script to run the SLAM system. 
Reads the video file, 
processes each frame, 
and displays the results in 2D and 3D.
"""
import os
import cv2
import numpy as np
import tqdm
import argparse

from aruco_slam import ArucoSlam
import viewers.viewer_3d as v3d
import viewers.viewer_2d as v2d

DISPLAY_3D = True
DISPLAY_2D = True

# contingent on the display flags
SAVE_2D = True
SAVE_3D = True

CALIB_MTX_FILE = 'calibration/camera_matrix.npy'
DIST_COEFFS_FILE = 'calibration/dist_coeffs.npy'

IMAGE_SIZE = 1920, 1080
DISPLAY_SIZE = 960, 540

# set numpy to print only 3 decimal places
np.set_printoptions(precision=3)

# set numpy to print in scientific notation only if the number is very large
np.set_printoptions(suppress=True)

def load_matrices():
    """
    loads camera calibration matrices
    """

    # assert that the camera matrix and distortion coefficients are saved
    assert os.path.exists(CALIB_MTX_FILE), \
        'Camera matrix not found. Run calibration.py first.'
    assert os.path.exists(DIST_COEFFS_FILE), \
        'Distortion coefficients not found. Run calibration.py first.'

    calib_matrix = np.load(CALIB_MTX_FILE)
    dist_coeffs = np.load(DIST_COEFFS_FILE)

    return calib_matrix, dist_coeffs

def main(args):
    """
    runs main thread
    """
    calib_matrix, dist_coeffs = load_matrices()

    # load the camera matrix and distortion coefficients, initialize the tracker
    initial_pose = np.array([0, 0, 0, 0, 0, 0]) # x, y, z, roll, pitch, yaw
    tracker = ArucoSlam(initial_pose, calib_matrix, dist_coeffs, args.filter)

    # use the camera
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    if DISPLAY_3D:
        camera_viewer_3d = v3d.Viewer3D(IMAGE_SIZE, SAVE_3D)
    if DISPLAY_2D:
        image_viewer_2d = v2d.Viewer2D(calib_matrix, dist_coeffs, SAVE_2D)

    # number of frames
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    iterator = tqdm.tqdm(range(frames), desc='Processing frames', unit='frames')

    while True:
        iterator.update(1)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, IMAGE_SIZE)

        # find markers, update system state
        frame, camera_pose, marker_poses, detected_poses = \
            tracker.process_frame(frame)

        if DISPLAY_3D:
            camera_viewer_3d.view(
                camera_pose,
                marker_poses,
                detected_poses
            )
        if DISPLAY_2D:
            q = image_viewer_2d.view(
                frame,
                camera_pose,
                marker_poses,
                detected_poses
            )
            if q:
                break

    if DISPLAY_3D:
        camera_viewer_3d.close()
    if DISPLAY_2D:
        image_viewer_2d.close()

    cap.release()

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description='Run the SLAM system')
    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file',
        default="input_video.mp4"
        )
    parser.add_argument(
        '--filter',
        type=str,
        help='Filter to use (kalman, factorgraph)',
        default='kalman'
        )

    args = parser.parse_args()

    video_file = args.video

    main(args)
