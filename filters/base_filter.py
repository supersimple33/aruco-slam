"""Handles the detecting ArUco. Acts as simple wrapper for EKF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# calibration files
CALIB_MTX_FILE = "./calibration/camera_matrix.npy"
DIST_COEFFS_FILE = "./calibration/dist_coeffs.npy"

# Filter types
KALMAN_FILTER = "ekf"
FACTOR_GRAPH = "factorgraph"

NOT_IMPLEMENTED_ERROR = """
                        This method is not implemented in the base class and
                        should be implemented in a subclass.
                        """


CAM_DIMS = 10  # x, y, z, qw, qx, qy, qz, ex, ey, ez
XYZ_DIMS = slice(0, 3)  # x, y, z
QUAT_DIMS = slice(3, 7)  # qw, qx, qy, qz
ERROR_DIMS = slice(7, 10)  # ex, ey, ez

LM_DIMS = 10  # x, y, z

QUAT_THRESHOLD = 50


class BaseFilter:
    """Class for performing SLAM with ArUco markers."""

    def __init__(
        self,
        initial_pose: np.ndarray,
        map_file: str | None,
        aruco_dict = None
    ) -> None:
        """Initialize the slam object.

        Arguments:
            initial_pose: the initial pose of the camera
            calib_matrix: the camera calibration matrix
            dist_coeffs: the camera distortion coefficients
            filter_type: the type of filter to use
            map_file: the file to load the map from

        """
        # assert that the camera matrix and distortion coefficients are saved
        if not Path(CALIB_MTX_FILE).exists():
            msg = "Camera matrix not found. Run calibration.py first."
            raise FileNotFoundError(msg)
        if not Path(DIST_COEFFS_FILE).exists():
            msg = "Distortion coefficients not found. Run calibration.py first."
            raise FileNotFoundError(msg)

        calib_matrix = np.load(CALIB_MTX_FILE)
        dist_coeffs = np.load(DIST_COEFFS_FILE)

        self.calib_matrix = calib_matrix
        self.dist_coeffs = dist_coeffs
        self.detector = self.init_aruco_detector(aruco_dict)

        self.camera_pose = initial_pose

        if map_file is not None:
            self.load_map(map_file)

    def init_aruco_detector(self, aruco_dict) -> cv2.aruco.ArucoDetector:
        """Create an aruco detector.

        Returns:
            detector: cv2.aruco.ArucoDetector object

        """
        if aruco_dict is None:
            aruco_dict = cv2.aruco.DICT_5X5_50
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
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
        ids: np.ndarray,
        marker_size: float,
    ) -> np.ndarray:
        """Estimate the pose of the markers.

        https://stackoverflow.com/questions/74964527/attributeerror-module- ...
        cv2-aruco-has-no-attribute-dictionary-get

        Arguments:
            corners (np.ndarray): the corners of the markers.
            ids (np.ndarray): the ids of the markers.
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

        for c, idx in zip(corners, ids):
            has_prior = False
            rot_prior = None

            # TODO(ssilver):
            # the following attempt at orientation ambiguity only works for
            # ekf_with_rotations, not ekf (no orientation), nor factorgraph
            # (no state). Leaving it here while I think about it.

            # if idx in self.landmarks:
            #
            #     # want to use the existing landmark pose to avoid ambiguity
            #     lm_index = self.landmarks[idx]

            #     # TODO(ssilver): get rid of all this. And, the code is repeated
            #     lm_i = LM_DIMS * lm_index + CAM_DIMS
            #     # get the landmark pose in the map frame
            #     q_ml = self.state[lm_i + 3 : lm_i + 7]  # qw, qx, qy, qz
            #     q_mc = self.state[QUAT_DIMS]  # qw, qx, qy, qz

            #     rot_ml = Rotation.from_quat(q_ml, scalar_first=True)
            #     rot_mc = Rotation.from_quat(q_mc, scalar_first=True)

            #     # convert to camera frame
            #     rot_cl = rot_ml.inv() * rot_mc

            #     # convert to rotation vector
            #     rot_prior = rot_cl.as_rotvec()

            #     has_prior = True

            _, rot, t = cv2.solvePnP(
                marker_points,
                c,
                self.calib_matrix,
                self.dist_coeffs,
                rvec=rot_prior,
                useExtrinsicGuess=has_prior,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            tvecs.append(t.flatten())
            rvecs.append(rot.flatten())

        tvecs = np.array(tvecs)
        rvecs = np.array(rvecs)

        return np.hstack((tvecs, rvecs))

    def process_frame(
        self,
        frame: np.ndarray,
        should_filter: bool = True,  # noqa: FBT001 FBT002
        iteration: int = 0,
        marker_size: float = 0.16,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process a frame.

        Arguments:
            frame: the image frame to process.
            should_filter: whether to filter the poses or not.
            iteration: the frame iteration for offline retrieval of that pose.

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

            detected_poses = self.estimate_pose_of_markers(corners, ids, marker_size)

            if should_filter:
                self.observe(ids, detected_poses)

        if should_filter:
            camera_pose, marker_poses = self.get_poses()
        else:
            _, marker_poses = self.get_poses()
            camera_pose = self.get_cam_estimate(iteration)

        return frame, camera_pose, marker_poses, detected_poses

    def save_map(self, filename: str) -> None:
        """Save the map to a file.

        Arguments:
            filename: the name of the file to save the map to.

        """
        _, marker_poses = self.get_poses()

        index_to_id = {v: k for k, v in self.get_lm_estimates()}

        uncertainties = self.get_lm_uncertainties()

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

                # TODO(ssilver):    # noqa: TD003 FIX002
                # ensure this is in map frame, not camera frame
                self.filter.add_marker(id_, pose, uncertainty)

    def correct_poses(
        self,
        ids: list[int],
        poses: list[np.ndarray],
    ) -> tuple[list[int], list[np.ndarray]]:
        """Correct the poses of the markers based on the current state.

        Arguments:
            ids: list of ids of the markers
            poses: list of poses of the markers

        Returns:
            corrected_ids: list of ids of the markers
            corrected_poses: list of corrected poses of the markers

        """
        corrected_ids = []
        corrected_poses = []

        cam_quat = self.state[QUAT_DIMS]  # Assuming (w, x, y, z)
        rot_mc = Rotation.from_quat(cam_quat, scalar_first=True)

        for idx, pose in zip(ids, poses):
            if idx not in self.landmarks:
                continue

            index = self.landmarks[idx]
            lm_i = LM_DIMS * index + CAM_DIMS

            # state landmark orientation
            state_quat = self.state[lm_i + 3 : lm_i + 7]  # qw, qx, qy, qz
            state_rot_ml = Rotation.from_quat(state_quat, scalar_first=True)

            # observed landmark rotation in camera frame
            rot_cl = Rotation.from_euler("xyz", pose[3:])

            # estimate landmark rotation in map frame
            rot_ml = rot_mc * rot_cl

            # difference between estimated and expected
            r_diff = rot_ml * state_rot_ml.inv()

            angle = np.degrees(r_diff.magnitude())
            angle = min(angle, 360 - angle)  # handle quaternion double cover

            if angle < QUAT_THRESHOLD:
                corrected_ids.append(idx)
                corrected_poses.append(pose)
            else:
                pass

        return corrected_ids, corrected_poses

    def observe(
        self,
        ids: np.ndarray,
        poses: np.ndarray,
    ) -> None:
        """Observe the markers.

        Arguments:
            ids: the ids of the markers.
            poses: the poses of the markers.

        """
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR)

    def get_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the poses of the camera and the landmarks.

        Returns:
            camera_pose: the pose of the camera.
            marker_poses: the poses of the markers.

        """
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR)

    def get_lm_uncertainties(self) -> np.ndarray:
        """Get the uncertainties of the landmarks.

        Returns:
            lm_uncertainties: the uncertainties of the landmarks.

        """
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR)

    def get_lm_estimates(self) -> np.ndarray:
        """Get the estimates of the landmarks.

        Returns:
            lm_estimates: the estimates of the landmarks.

        """
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR)

    def get_cam_estimate(self, iteration: int) -> np.ndarray:
        """Get the pose estimate at a specific timestamp/iteration.

        Should only be used for offline processing (FaCTOR_GRAPH).

        Arguments:
            iteration: the id of the landmark

        Returns:
            The pose of the landmark in the map frame.

        """
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR)
