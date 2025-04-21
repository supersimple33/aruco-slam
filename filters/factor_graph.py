"""Defines an EKF class."""

import gtsam
import numpy as np
from gtsam import Point3, Pose3, Rot3
from gtsam.symbol_shorthand import L, X
from scipy.spatial.transform import Rotation

from filters.base_filter import BaseFilter

# Uncertainty values
PRIOR_NOISE_XYZ = 1.0  # meters
PRIOR_NOISE_RPY = 20 * np.pi / 180  # degrees -> radians
ODOM_NOISE_XYZ = 1.0  # meters
ODOM_NOISE_RPY = 60 * np.pi / 180  # degrees -> radians
MEASUREMENT_NOISE_XYZ = 1.0  # meters
MEASUREMENT_NOISE_RPY = 30 * np.pi / 180  # degrees -> radians

HISTORICAL_FREQUENCY = 10
SLIDING_WINDOW_SIZE = 3


class FactorGraph(BaseFilter):
    """Factor Graph that tracks the positions of the cameras and landmarks."""

    def __init__(self, initial_camera_pose: np.ndarray) -> None:
        """Initialize filter."""
        super().__init__(initial_camera_pose, None)

        self.position = initial_camera_pose[:3]
        self.rotation = initial_camera_pose[3:7]

        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [
                    PRIOR_NOISE_RPY,
                    PRIOR_NOISE_RPY,
                    PRIOR_NOISE_RPY,
                    PRIOR_NOISE_XYZ,
                    PRIOR_NOISE_XYZ,
                    PRIOR_NOISE_XYZ,
                ],
            ),
        )

        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [
                    ODOM_NOISE_RPY,
                    ODOM_NOISE_RPY,
                    ODOM_NOISE_RPY,
                    ODOM_NOISE_XYZ,
                    ODOM_NOISE_XYZ,
                    ODOM_NOISE_XYZ,
                ],
            ),
        )

        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [
                    MEASUREMENT_NOISE_RPY,
                    MEASUREMENT_NOISE_RPY,
                    MEASUREMENT_NOISE_RPY,
                    MEASUREMENT_NOISE_XYZ,
                    MEASUREMENT_NOISE_XYZ,
                    MEASUREMENT_NOISE_XYZ,
                ],
            ),
        )

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = gtsam.Values()
        self.isam = gtsam.ISAM2()

        self.graph.add(
            gtsam.PriorFactorPose3(
                X(0),
                Pose3(
                    Rot3.Rodrigues(self.rotation),
                    Point3(self.position),
                ),
                self.prior_noise,
            ),
        )

        # add the inital pose to the estimates
        pose = Pose3(
            Rot3.Rodrigues(self.rotation),
            Point3(self.position),
        )
        self.initial_estimate.insert(X(0), pose)
        self.current_estimate.insert(X(0), pose)

        # keep track of landmark indices
        self.num_landmarks = 0
        self.landmarks = {}

        # keep track of what timestep we are on
        self.i = 0
        self.historical_timestep = True
        self.historical_factors = []

        # window to keep track of the last few poses and their factors
        self.sliding_window = []

    def observe(
        self,
        ids: list[int],
        poses: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Take in a set of observations and update the state of the filter.

        Arguments:
            ids: list of ids of the markers
            poses: list of poses of the markers

        Return:
            camera_pose: the updated camera pose
            marker_poses: the updated marker poses in the map frame

        """
        self.historical_timestep = self.i % HISTORICAL_FREQUENCY == 0
        camera_pose = self.current_estimate.atPose3(X(self.i))

        for idx, pose in zip(ids, poses):
            self.add_landmark_observation(idx, pose, camera_pose)

        # use last camera pose and zero motion model to add odometry factor
        self.add_odom_factor_and_estimate(camera_pose)

        # TODO(ssilver): historical factors should be  # noqa: TD003 FIX002
        #                present at every timestep.
        # Clear graph and add timesteps manually at every timestep?
        if self.historical_timestep:
            for factor in self.historical_factors:
                self.graph.push_back(factor)

        # don't optimize on the first iteration
        if self.i == 0:
            self.current_estimate = self.initial_estimate
        else:
            self.isam.update(self.graph, self.initial_estimate)
            self.current_estimate = self.isam.calculateEstimate()
            self.initial_estimate.clear()

        self.i += 1

        self.prune_graph()

    def add_odom_factor_and_estimate(
        self,
        camera_pose: gtsam.Pose3,
    ) -> None:
        """Add an odometry factor to the graph.

        Arguments:
            camera_pose: the pose of the camera

        """
        # TODO(ssilver): Add moving average model. # noqa: TD003 FIX002

        self.initial_estimate.insert(
            X(self.i + 1),
            camera_pose,
        )

        self.graph.push_back(
            gtsam.BetweenFactorPose3(
                X(self.i + 1),
                X(self.i),
                Pose3(
                    Rot3.Rodrigues([0, 0, 0]),
                    Point3(0, 0, 0),
                ),
                self.odom_noise,
            ),
        )

    def get_poses(self) -> None:
        """Return the poses of the camera and the landmarks."""
        camera_pose = self.current_estimate.atPose3(X(self.i))
        camera_rot = camera_pose.rotation().toQuaternion().coeffs()
        # x, y, z, w -> w, x, y, z
        camera_rot = camera_rot[[3, 0, 1, 2]]
        camera_translation = camera_pose.translation()
        camera_pose = np.hstack((camera_translation, camera_rot))

        lm_translations = []
        for idx in range(self.num_landmarks):
            landmark_pose = self.current_estimate.atPose3(L(idx))
            lm_translations.append(landmark_pose.translation())
        lm_translations = np.array(lm_translations)

        return camera_pose, lm_translations

    def get_lm_uncertainties(self) -> np.ndarray:
        """Return the uncertainties of the landmarks."""
        lm_uncertainties = []
        for idx in range(self.num_landmarks):
            key = L(idx)
            if self.current_estimate.exists(key):
                lm_uncertainties.append(
                    self.isam.marginalCovariance(key).diagonal(),
                )

        return np.array(lm_uncertainties)

    def prune_graph(self) -> None:
        """Prune the graph by removing old nodes."""
        # TODO(ssilver): implement using timestep # noqa: TD003 FIX002
        # aware way to prune the graph
        self.graph.resize(100)

    def add_marker(
        self,
        idx: int,
        pose: np.ndarray,
        uncertainity: np.ndarray = None,
    ) -> None:
        """Add a new marker to the state.

        Arguments:
            idx: the id of the marker
            pose: the pose of the marker
            uncertainity: the uncertainity of the pose, if known (from map)

        """
        camera_pose = self.current_estimate.atPose3(X(self.i))

        _ = uncertainity

        self.landmarks[idx] = self.num_landmarks
        idx = self.num_landmarks
        self.num_landmarks += 1

        camera_translation = camera_pose.translation()
        camera_rotation = camera_pose.rotation().rpy()

        # update the state
        rot_cm = Rotation.from_euler("xyz", camera_rotation).as_matrix()
        rot_mc = np.linalg.inv(rot_cm)

        # put the landmark's pose in map frame
        t_ml = rot_mc @ pose[:3] + camera_translation

        self.initial_estimate.insert(
            L(idx),
            Pose3(
                Rot3.Rodrigues([0, 0, 0]),
                Point3(t_ml),
            ),
        )

    def add_landmark_observation(
        self,
        idx: int,
        pose: np.ndarray,
        camera_pose: gtsam.Pose3,
    ) -> None:
        """Add a new landmark observation to the graph.

        Arguments:
            idx: the id of the landmark
            pose: the pose of the landmark
            camera_pose: the pose of the camera

        """
        if idx in self.landmarks:
            idx = self.landmarks[idx]
        else:
            self.add_marker(idx, pose)
            idx = self.landmarks[idx]

        cam_rot_inv = camera_pose.rotation().inverse()
        factor = gtsam.BetweenFactorPose3(
            X(self.i),
            L(idx),
            Pose3(
                cam_rot_inv,
                Point3(pose[:3]),  # landmark in camera frame
            ),
            self.measurement_noise,
        )

        # add the observation to the graph
        self.graph.push_back(factor)

        if self.historical_timestep:
            self.historical_factors.append(factor)

    def get_lm_estimates(self) -> np.ndarray:
        """Return the estimates of the landmarks."""
        return self.landmarks.items()
