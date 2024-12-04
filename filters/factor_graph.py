"""
Defines an EKF class that uses the position of the detected markers to create
a map and localize the camera in the map
"""

from typing import (
    List,
    Tuple
)

import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation

import gtsam
from gtsam.symbol_shorthand import X, L
from gtsam import Pose3, Point3, Rot3, Cal3_S2, Point2, noiseModel

# Uncertainty values
PRIOR_NOISE_XYZ = 1.0 # meters
PRIOR_NOISE_RPY = 20 * np.pi / 180 # degrees -> radians
ODOM_NOISE_XYZ = 1.0 # meters
ODOM_NOISE_RPY = 60 * np.pi / 180 # degrees -> radians
MEASUREMENT_NOISE_XYZ = 1.0 # meters
MEASUREMENT_NOISE_RPY = 30 * np.pi / 180 # degrees -> radians

HISTORICAL_FREQUENCY = 10
SLIDING_WINDOW_SIZE = 3

class FactorGraph(object):
    """
    Implements a Kalman Filter that tracks the positions of the cameras and 
    the aruco landmarks
    """

    def __init__(self, initial_camera_pose):
        """
        initializes filter
        """

        self.position = initial_camera_pose[:3]
        self.rotation = initial_camera_pose[3:]

        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([PRIOR_NOISE_RPY, PRIOR_NOISE_RPY, PRIOR_NOISE_RPY,
                      PRIOR_NOISE_XYZ, PRIOR_NOISE_XYZ, PRIOR_NOISE_XYZ])
        )

        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([ODOM_NOISE_RPY, ODOM_NOISE_RPY, ODOM_NOISE_RPY,
                      ODOM_NOISE_XYZ, ODOM_NOISE_XYZ, ODOM_NOISE_XYZ])
        )

        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([MEASUREMENT_NOISE_RPY, MEASUREMENT_NOISE_RPY, MEASUREMENT_NOISE_RPY,
                      MEASUREMENT_NOISE_XYZ, MEASUREMENT_NOISE_XYZ, MEASUREMENT_NOISE_XYZ])
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
                    Point3(self.position)
                ),
                self.prior_noise
            )
        )

        # add the inital pose to the estimates
        pose = Pose3(
            Rot3.Rodrigues(self.rotation),
            Point3(self.position)
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
        # self.sliding_window = []

    def observe(
            self,
            ids: List[int],
            poses: List[np.ndarray]
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes in a set of observations and updates the state of the filter

        params:
        - ids: list of ids of the markers
        - poses: list of poses of the markers

        returns:
        - camera_pose: the updated camera pose
        - marker_poses: the updated marker poses in the map frame
        """

        self.historical_timestep = self.i % HISTORICAL_FREQUENCY == 0
        camera_pose = self.current_estimate.atPose3(X(self.i))

        for idx, pose in zip(ids, poses):
            self.add_landmark_observation(idx, pose, camera_pose)

        # use last camera pose and zero motion model to add odometry factor
        self.add_odom_factor_and_estimate(camera_pose)

        # FIXME: historical factors should be present at every timestep. 
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
            camera_pose: gtsam.Pose3
            ) -> None:
        """
        Adds an odometry factor to the graph

        params:
        - camera_pose: the pose of the camera
        """
        self.initial_estimate.insert(
            X(self.i + 1),
            camera_pose
        )

        self.graph.push_back(
            gtsam.BetweenFactorPose3(
                X(self.i + 1),
                X(self.i),
                Pose3(
                    Rot3.Rodrigues([0, 0, 0]),
                    Point3(0, 0, 0)
                ),
                self.odom_noise
            )
        )

    def get_poses(self):
        """
        Returns the poses of the camera and the landmarks
        """

        camera_pose = self.current_estimate.atPose3(X(self.i))
        camera_rot = camera_pose.rotation().rpy()
        camera_translation = camera_pose.translation()
        camera_pose = np.hstack((camera_translation, camera_rot))

        lm_translations = []
        for idx in range(self.num_landmarks):
            landmark_pose = self.current_estimate.atPose3(L(idx))
            lm_translations.append(landmark_pose.translation())
        lm_translations = np.array(lm_translations)

        return camera_pose, lm_translations
    
    def prune_graph(self):
        """
        Prunes the graph by removing old nodes
        """

        # TODO: implement a more intelligent, timestep aware
        #  way to prune the graph
        self.graph.resize(600)

    def add_landmark_observation(
            self,
            idx: int, 
            pose: np.ndarray,
            camera_pose: gtsam.Pose3
            ) -> None:
        """
        Adds a new landmark observation to the graph

        params:
        - idx: the id of the landmark
        - pose: the pose of the landmark

        returns:
        - None
        """


        if idx in self.landmarks:
            idx = self.landmarks[idx]
        else:
            self.landmarks[idx] = self.num_landmarks
            idx = self.num_landmarks
            self.num_landmarks += 1

            camera_translation = camera_pose.translation()
            camera_rotation = camera_pose.rotation().rpy()

            # update the state
            rot_cm = Rotation.from_euler('xyz', camera_rotation).as_matrix()
            rot_mc = np.linalg.inv(rot_cm)

            # put the landmark's pose in map frame
            t_ml = rot_mc @ pose[:3] + camera_translation

            self.initial_estimate.insert(
                L(idx),
                Pose3(
                    Rot3.Rodrigues([0, 0, 0]),
                    Point3(t_ml)
                )
            )

        cam_rot_inv = camera_pose.rotation().inverse()
        factor = gtsam.BetweenFactorPose3(
                    X(self.i),
                    L(idx),
                    Pose3(
                        cam_rot_inv,
                        Point3(pose[:3]) # landmark in camera frame
                    ),
                    self.measurement_noise
                )

        # add the observation to the graph
        self.graph.push_back(factor)

        if self.historical_timestep:
            self.historical_factors.append(factor)