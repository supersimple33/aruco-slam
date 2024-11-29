"""
Handles the detecting ArUco. Acts as simple wrapper for EKF
"""

from typing import Tuple, List

import numpy as np
import cv2

from filters.extended_kalman_filter import EKF

class ArucoSlam(object):
    """
    Slam object
    """

    def __init__(
            self, 
            initial_pose: np.ndarray,
            calib_matrix: np.ndarray,
            dist_coeffs: np.ndarray
            ) -> None:
        """
        Initializes the slam object
        
        params:
        - initial_pose: the initial pose of the camera
        - calib_matrix: the camera calibration matrix
        - dist_coeffs: the camera distortion coefficients
        """
        self.calib_matrix = calib_matrix
        self.dist_coeffs = dist_coeffs
        self.detector = self.init_aruco_detector()

        self.filter = EKF(initial_pose)

        self.camera_pose = initial_pose

    def init_aruco_detector(self):
        """
        Creates an aruco detector

        returns:
        - detector: cv2.aruco.ArucoDetector object
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.cornerRefinementWinSize = 3
        aruco_params.cornerRefinementMaxIterations = 3
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 30
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        return detector

    def estimate_pose_of_markers(self, corners, marker_size):
        '''
        https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get
        
        params:
        - corners: the corners of the markers
        - marker_size: the size of the markers
        '''

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

        tvecs = []
        rvecs = []

        for c in corners:
            _, rot, t = cv2.solvePnP(
                marker_points,
                c,
                self.calib_matrix,
                self.dist_coeffs,
                False,
                cv2.SOLVEPNP_IPPE_SQUARE
            )

            tvecs.append(t.flatten())
            rvecs.append(rot.flatten())

        tvecs = np.array(tvecs)
        rvecs = np.array(rvecs)

        poses = np.hstack((tvecs, rvecs))

        return poses

    def process_frame(
            self,
            frame: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a frame

        params:
        - frame: the image frame to process

        returns:
        - frame: the frame with the markers drawn on it
        - camera_pose: the pose of the camera as estimated by the filter
        - marker_poses: the poses of the markers as estimated by the filter
        - detected_poses: the detected poses of markers
        """
        corners, ids, _ = tuple(self.detector.detectMarkers(frame))

        detected_poses = np.array([])
        camera_pose = self.filter.state[:6]
        marker_poses = np.array([])
        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            ids = ids.flatten()

            detected_poses = self.estimate_pose_of_markers(corners, 0.2)

            camera_pose, marker_poses = self.filter.observe(ids, detected_poses)

        return frame, camera_pose, marker_poses, detected_poses
    
    