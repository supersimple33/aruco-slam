"""Handles the detecting ArUco. Acts as simple wrapper for EKF."""

import cv2
import numpy as np

from filters.extended_kalman_filter import EKF
from filters.factor_graph import FactorGraph

KALMAN_FILTER = "kalman"
FACTOR_GRAPH = "factorgraph"


class ArucoSlam:
    """Class for performing SLAM with ArUco markers."""

    def __init__(
        self,
        initial_pose: np.ndarray,
        calib_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        filter_type: str,
    ) -> None:
        """Initialize the slam object.

        Arguments:
            initial_pose: the initial pose of the camera
            calib_matrix: the camera calibration matrix
            dist_coeffs: the camera distortion coefficients
            filter_type: the type of filter to use

        """
        self.calib_matrix = calib_matrix
        self.dist_coeffs = dist_coeffs
        self.detector = self.init_aruco_detector()

        if filter_type == KALMAN_FILTER:
            self.filter = EKF(initial_pose)
        elif filter_type == FACTOR_GRAPH:
            self.filter = FactorGraph(initial_pose)
        else:
            error_message = "Invalid filter type. Use -h for help."
            raise ValueError(error_message)

        self.camera_pose = initial_pose

    def init_aruco_detector(self) -> cv2.aruco.ArucoDetector:
        """Create an aruco detector.

        Returns:
            detector: cv2.aruco.ArucoDetector object

        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.cornerRefinementWinSize = 3
        aruco_params.cornerRefinementMaxIterations = 3
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 30
        return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    def estimate_pose_of_markers(
        self,
        corners: np.ndarray,
        marker_size: float,
    ) -> np.ndarray:
        """Estimate the pose of the markers.

        https://stackoverflow.com/questions/74964527/attributeerror-module- ...
        cv2-aruco-has-no-attribute-dictionary-get

        Arguments:
            corners (np.ndarray): the corners of the markers.
            marker_size (float): the size of the markers.

        Returns:
            np.array: the estimated poses of the markers

        """
        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )

        tvecs = []
        rvecs = []

        for c in corners:
            _, rot, t = cv2.solvePnP(
                marker_points,
                c,
                self.calib_matrix,
                self.dist_coeffs,
                False,  # noqa: FBT003
                cv2.SOLVEPNP_IPPE_SQUARE,
            )

            tvecs.append(t.flatten())
            rvecs.append(rot.flatten())

        tvecs = np.array(tvecs)
        rvecs = np.array(rvecs)

        return np.hstack((tvecs, rvecs))

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process a frame.

        Arguments:
            frame: the image frame to process.

        Returns:
            frame: the frame with the markers drawn on it.
            camera_pose: the pose of the camera as estimated by the filter.
            marker_poses: the poses of the markers as estimated by the filter.
            detected_poses: the detected poses of markers.

        """
        corners, ids, _ = tuple(self.detector.detectMarkers(frame))

        detected_poses = np.array([])
        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            ids = ids.flatten()

            detected_poses = self.estimate_pose_of_markers(corners, 0.2)

            self.filter.observe(ids, detected_poses)

        camera_pose, marker_poses = self.filter.get_poses()

        return frame, camera_pose, marker_poses, detected_poses
