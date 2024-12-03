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

INITIAL_CAMERA_UNCERTAINTY = 0.1
INITIAL_LANDMARK_UNCERTAINTY = 0.7

PRIOR_NOISE_XYZ = 0.3 # meters
PRIOR_NOISE_RPY = 10 * np.pi / 180 # degrees -> radians

ODOM_NOISE_XYZ = 0.3 # meters
ODOM_NOISE_RPY = 10 * np.pi / 180 # degrees -> radians

MEASUREMENT_NOISE_XYZ = 0.3 # meters
MEASUREMENT_NOISE_RPY = 10 * np.pi / 180 # degrees -> radians


CAM_DIMS = 6 # x, y, z, roll, pitch, yaw
LM_DIMS = 3 # x, y, z

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

        self.initial_estimate.insert(
            X(0),
            Pose3(
                Rot3.Rodrigues(self.rotation),
                Point3(self.position)
            )
        )
        self.current_estimate.insert(
            X(0),
            Pose3(
                Rot3.Rodrigues(self.rotation),
                Point3(self.position)
            )
        )

        self.num_landmarks = 0
        self.landmarks = {}

        self.i = 0

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

        for idx, pose in zip(ids, poses):
            self.add_landmark_observation(idx, pose)

        current_pose = self.current_estimate.atPose3(X(self.i))

        self.initial_estimate.insert(
            X(self.i + 1),
            current_pose
        )
        self.graph.push_back(
            gtsam.BetweenFactorPose3(
                X(self.i + 1),
                X(self.i),
                current_pose, # TODO this needs to be from c_m to c_c,t-1
                self.odom_noise
            )
        )

        # don't optimize on the first iteration
        if self.i == 0:
            self.current_estimate = self.initial_estimate
        else:
            self.isam.update(self.graph, self.initial_estimate)
            self.current_estimate = self.isam.calculateEstimate()
            self.initial_estimate.clear()

        self.i += 1

    def add_landmark_observation(
            self,
            idx: int, 
            pose: np.ndarray
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
            print(idx)
            idx = self.num_landmarks
            self.num_landmarks += 1

            self.initial_estimate.insert( # TODO this needs from lm_c to lm_m
                L(idx),
                Pose3(
                    Rot3.Rodrigues(pose[3:]),
                    Point3(pose[:3])
                )
            )

        self.graph.add(
            gtsam.BetweenFactorPose3(
                X(self.i),
                L(idx),
                Pose3(
                    Rot3.Rodrigues(pose[3:]),
                    Point3(pose[:3])
                ),
                self.measurement_noise
            )
        )

    def parse_poses(
            self,
            ids: List[int],
            poses: List[np.ndarray]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parses the observations into observed, predicted, and jacobian values.

        params:
        - ids: list of ids of the markers
        - poses: list of poses of the markers

        returns:
        - z: the observed values
        - h: the predicted values
        - dh: the jacobian matrix
        """
        pass