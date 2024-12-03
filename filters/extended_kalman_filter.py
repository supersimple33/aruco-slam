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

INITIAL_CAMERA_UNCERTAINTY = 0.1
INITIAL_LANDMARK_UNCERTAINTY = 0.7

R_UNCERTAINTY = 0.8
Q_UNCERTAINTY = 0.3

CAM_DIMS = 6 # x, y, z, roll, pitch, yaw
LM_DIMS = 3 # x, y, z

class EKF(object):
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

        # 6n + 6, where n is the number of landmarks
        self.state = np.array(initial_camera_pose)

        self.uncertainty = np.eye(CAM_DIMS) * INITIAL_CAMERA_UNCERTAINTY

        self.num_landmarks = 0
        self.landmarks = {}

        # initialize the H and h functions
        h, dh = self.initialize_h()
        self.h = h
        self.partial_jacobian = dh

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

        # add any new markers
        for idx, pose in zip(ids, poses):
            if idx in self.landmarks:
                continue
            else:
                self.add_marker(idx, pose)

        # perform prediction, update steps
        self.predict()
        self.update(ids, poses)

        camera_pose = self.state[:CAM_DIMS]
        marker_poses = self.state[CAM_DIMS:].reshape(-1, LM_DIMS)

        return camera_pose, marker_poses

    def predict(self):
        """
        Predicts the next state of the system
        """

        # no system model, no explicit state change
        # just update the uncertainity matrix for camera motion
        q_dims = LM_DIMS * self.num_landmarks + CAM_DIMS
        q = np.zeros((q_dims, q_dims))
        q[:CAM_DIMS, :CAM_DIMS] = np.eye(CAM_DIMS) * Q_UNCERTAINTY
        self.uncertainty += q

    def update(
            self,
            ids: List[int],
            poses: List[np.ndarray]
            ) -> None:
        """
        Updates the state of the filter

        params:
        - ids: list of ids of the markers
        - poses: list of poses of the markers
        """

        # perform update step
        z, h, dh = self.parse_poses(ids, poses)
        r = np.eye(dh.shape[0]) * R_UNCERTAINTY # measurement uncertainty
        s = dh @ self.uncertainty @ dh.T + r    # innovation covariance
        kalman_gain = self.uncertainty @ dh.T @ np.linalg.inv(s) # kalman gain
        innovation = kalman_gain @ (z - h)
        self.state += innovation

        # update uncertainty
        ident = np.eye(LM_DIMS * self.num_landmarks + CAM_DIMS)
        self.uncertainty = (ident - kalman_gain @ dh) @ self.uncertainty

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

        # perform update step
        z = None
        h = None
        dh = None
        for idx, pose in zip(ids, poses):
            index = self.landmarks[idx]

            jacobian_row = self.landmark_dh(index)

            cam_state = self.state[:CAM_DIMS] # x, y, z, roll, pitch, yaw
            state_index = LM_DIMS*index + CAM_DIMS
            landmark_state = self.state[state_index:state_index + LM_DIMS]

            h_row = self.h([*cam_state, *landmark_state]).squeeze()

            # add to the matrices
            if z is None:
                z = pose[:3] # only looking at translation
                h = h_row
                dh = jacobian_row
            else:
                z = np.hstack((z, pose[:3]))
                h = np.hstack((h, h_row))
                dh = np.vstack((dh, jacobian_row))

        return z, h, dh

    def landmark_dh(
            self,
            index: int
            ) -> np.ndarray:
        """
        Computes the jacobian for a specific landmark. Places it in the correct
        position in the full jacobian matrix

        params:
        - index: the index of the landmark
        - pose: the pose of the landmark

        returns:
        - dh: the partial jacobian
        """

        cam_state = self.state[:CAM_DIMS] # x, y, z, roll, pitch, yaw

        beginning_index = LM_DIMS*index + CAM_DIMS

        # x, y, z
        landmark_state = self.state[beginning_index:beginning_index+LM_DIMS]

        # get the partial jacobian
        jacobian = self.partial_jacobian([*cam_state, *landmark_state])

        camera_jacobian = jacobian[:, :CAM_DIMS]
        landmark_jacobian = jacobian[:, CAM_DIMS:]

        # get the H matrix
        dh = np.zeros((LM_DIMS, LM_DIMS*self.num_landmarks + CAM_DIMS))
        dh[:, :CAM_DIMS] = camera_jacobian

        index = LM_DIMS * index + CAM_DIMS
        dh[:, index:index+LM_DIMS] = landmark_jacobian
        return dh


    def add_marker(self, idx, pose):
        """
        Adds a new marker to the state
        """

        self.landmarks[idx] = self.num_landmarks
        self.num_landmarks += 1

        camera_pose = self.state[:CAM_DIMS]
        camera_translation = camera_pose[:3]
        camera_rotation = camera_pose[3:]

        # update the state
        rot_cm = Rotation.from_euler('xyz', camera_rotation).as_matrix()
        rot_mc = np.linalg.inv(rot_cm)

        # put the landmark's pose in map frame
        t_ml = rot_mc @ pose[:3] + camera_translation

        self.state = np.hstack((self.state, t_ml))

        # expand the uncertainity matricies
        n_dims = LM_DIMS * self.num_landmarks + CAM_DIMS
        new_uncertainty = np.eye(n_dims) * INITIAL_LANDMARK_UNCERTAINTY
        new_uncertainty[:n_dims - LM_DIMS, :n_dims - LM_DIMS] = self.uncertainty
        self.uncertainty = new_uncertainty

    def initialize_h(self):
        """
        Initializes the H and h functions by using scipy's symbolic math to
        generate lambdas

        returns:
        - h: the h function, which maps the state to the observed values
        - dh: the jacobian of the h function
        """

        # Define translation and rotation variables
        x_mc, y_mc, z_mc = sp.symbols('x_r^m y_r^m z_r^m')
        phi_mc, theta_mc, psi_mc = sp.symbols('phi_r^m theta_r^m psi_r^m')

        x_ml, y_ml, z_ml = sp.symbols('x_l^m y_l^m z_l^m')

        # using angle from camera to map, not map to camera
        forward = -1

        # Rotation matrices
        rot_cm_x = sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(forward * phi_mc), -sp.sin(forward * phi_mc)],
            [0, sp.sin(forward * phi_mc), sp.cos(forward * phi_mc)]
        ])

        rot_cm_y = sp.Matrix([
            [sp.cos(forward * theta_mc), 0, sp.sin(forward * theta_mc)],
            [0, 1, 0],
            [-sp.sin(forward * theta_mc), 0, sp.cos(forward * theta_mc)]
        ])

        rot_cm_z = sp.Matrix([
            [sp.cos(forward * psi_mc), -sp.sin(forward * psi_mc), 0],
            [sp.sin(forward * psi_mc), sp.cos(forward * psi_mc), 0],
            [0, 0, 1]
        ])

        # Combine the rotation matrices
        rot_cm = rot_cm_x @ rot_cm_y @ rot_cm_z

        # Define state vectors
        x_m_r = sp.Matrix([x_mc, y_mc, z_mc,])
        x_m_l = sp.Matrix([x_ml, y_ml, z_ml])

        # Define the function to compute landmark position in camera frame
        function = rot_cm @ (x_m_l - x_m_r)

        variables = sp.Matrix([
            x_mc, y_mc, z_mc, phi_mc, theta_mc, psi_mc,
            x_ml, y_ml, z_ml
        ])
        h = sp.lambdify([variables], function, modules=['numpy'])

        # Compute the Jacobian
        jacobian = function.jacobian(variables)

        # Lambdify the Jacobian
        dh = sp.lambdify([variables], jacobian, modules=['numpy'])

        return h , dh
    