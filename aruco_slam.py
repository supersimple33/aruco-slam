"""Handles the detecting ArUco. Acts as simple wrapper for EKF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from filters.extended_kalman_filter import EKF
from filters.factor_graph import FactorGraph

# Filter types
KALMAN_FILTER = "ekf"
FACTOR_GRAPH = "factorgraph"


class ArucoSlam:
    """Class for performing SLAM with ArUco markers."""

    def __init__(
        self,
        initial_pose: np.ndarray,
        calib_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        filter_type: str,
        map_file: str | None,
    ) -> None:
        """Initialize the slam object.

        Arguments:
            initial_pose: the initial pose of the camera
            calib_matrix: the camera calibration matrix
            dist_coeffs: the camera distortion coefficients
            filter_type: the type of filter to use
            map_file: the file to load the map from

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

        if map_file is not None:
            self.load_map(map_file)

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

    def save_map(self, filename: str) -> None:
        """Save the map to a file.

        Arguments:
            filename: the name of the file to save the map to.

        """
        _, marker_poses = self.filter.get_poses()

        index_to_id = {v: k for k, v in self.filter.landmarks.items()}

        uncertainties = self.filter.get_lm_uncertainties()

        with Path(filename).open("w", encoding="utf-8") as file:
            # write the header:
            file.write("# landmark_id\n")
            file.write("# x y z\n")
            file.write("# uncertainty\n")
            file.write("\n")

            for i, pose in enumerate(marker_poses):
                # write the id
                file.write(f"{index_to_id[i]}\n")

                # write the pose
                file.write(f"{', '.join(map(str, pose))}\n")

                # write the uncertainty for pose variables
                file.write(
                    f"{', '.join(map(str, uncertainties[i, :len(pose)]))}\n",
                )

                # write a newline
                file.write("\n")

    def load_map(self, filename: str) -> None:
        """Load the map from a file.

        Arguments:
            filename: the name of the file to load the map from.

        """
        with Path(filename).open("r", encoding="utf-8") as file:
            lines = file.readlines()

            # skip the header
            lines = lines[4:]

            for i in range(0, len(lines), 4):
                id_ = int(lines[i].strip())
                pose = np.array(lines[i + 1].strip().split(", "), np.float32)
                uncertainty = np.array(
                    lines[i + 2].strip().split(", "),
                    np.float32,
                )

                # TODO (ssilver):    # noqa: TD003 FIX002
                # ensure this is in map frame, not camera frame
                self.filter.add_marker(id_, pose, uncertainty)
