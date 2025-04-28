"""Defines an EKF class."""

# save to ignore the warning about the sparse matrix format
import time
import warnings

import dill
import numpy as np
import sympy as sp
from scipy import sparse
from scipy.spatial.transform import Rotation

from filters.base_filter import BaseFilter
# from filters.twist_ekf import TwistMotionModel

warnings.filterwarnings(
    "ignore",
    message="splu converted its input to CSC format",
)

MOVING_AVG_WINDOW = 10

INITIAL_CAMERA_UNCERTAINTY = 0.1
INITIAL_LANDMARK_UNCERTAINTY = 0.7

R_UNCERTAINTY = 0.9
Q_UNCERTAINTY_CAM = 0.3
Q_ERROR_UNCERTAINTY_CAM = 0.5
Q_UNCERTAINTY_LM = 0.01

CAM_DIMS = 10  # x, y, z, qw, qx, qy, qz, ex, ey, ez
XYZ_DIMS = slice(0, 3)  # x, y, z
QUAT_DIMS = slice(3, 7)  # qw, qx, qy, qz
ERROR_DIMS = slice(7, 10)  # ex, ey, ez

LM_DIMS = 10  # x, y, z

QUAT_THRESHOLD = 1.2


class EKF_Rotations(BaseFilter):
    """Object for tracking the positions of the cameras and landmarks."""

    def __init__(self, initial_camera_pose: np.ndarray) -> None:
        """Initialize filter."""
        # TODO(ssilver): add map loading functionality # noqa: TD003 FIX002
        super().__init__(initial_camera_pose, None)

        # 3n + 10, where n is the number of landmarks
        self.state = np.array(initial_camera_pose)

        self.uncertainty = np.eye(CAM_DIMS) * INITIAL_CAMERA_UNCERTAINTY

        self.num_landmarks = 0
        self.landmarks = {}

        # self.twist_model = TwistMotionModel()

        # initialize the H and h functions
        h, dh = self.initialize_h()
        self.h = h
        self.partial_jacobian = dh

    def observe(
        self,
        ids: list[int],
        poses: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Take in a set of observations and updates the state of the filter.

        Arguments:
            ids: list of ids of the markers
            poses: list of poses of the markers

        Returns:
            camera_pose: the updated camera pose
            marker_poses: the updated marker poses in the map frame

        """
        # add any new markers
        for idx, pose in zip(ids, poses):
            if idx in self.landmarks:
                continue
            self.add_marker(idx, pose)

        # perform prediction, update steps
        self.predict()
        self.update(ids, poses)

    def get_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current camera and marker poses."""
        camera_pose = self.state[:CAM_DIMS]
        marker_poses = self.state[CAM_DIMS:].reshape(-1, LM_DIMS)

        return camera_pose, marker_poses

    def get_lm_uncertainties(self) -> np.ndarray:
        """Return the uncertainties of the landmarks."""
        return np.diagonal(self.uncertainty)[CAM_DIMS:].reshape(-1, LM_DIMS)

    def predict(self) -> None:
        """Predict the next state of the system."""
        # # cam_movement avg
        # dt = 1.0 / 30.0  # assuming 30 FPS
        # twist = self.twist_model.predict(dt)
        # self.state[XYZ_DIMS] += twist[:3]  # update translation

        # w = twist[3:6]  # angular velocity
        # # update the quaternion using small angle approximation
        # dq = [1, *w * dt / 2]  # small angle approximation
        # q = self.state[QUAT_DIMS]
        # q = Rotation.from_quat(q, scalar_first=True)
        # dq = Rotation.from_quat(dq, scalar_first=True)
        # q = dq * q  # quaternion multiplication
        # self.state[QUAT_DIMS] = q.as_quat(scalar_first=True)

        # update the uncertainity matrix for camera motion
        q_dims = LM_DIMS * self.num_landmarks + CAM_DIMS
        q = np.zeros((q_dims, q_dims))
        q[XYZ_DIMS, XYZ_DIMS] = np.eye(XYZ_DIMS.stop) * Q_UNCERTAINTY_CAM
        q[ERROR_DIMS, ERROR_DIMS] = np.eye(3) * Q_ERROR_UNCERTAINTY_CAM
        q[CAM_DIMS:, CAM_DIMS:] = (
            np.eye(LM_DIMS * self.num_landmarks) * Q_UNCERTAINTY_LM
        )
        self.uncertainty += q

    def update(
        self,
        ids: list[int],
        poses: list[np.ndarray],
    ) -> None:
        """Update the state of the filter.

        Arguments:
            ids: list of ids of the markers
            poses: list of poses of the markers

        """
        # perform update step
        z, h, dh = self.parse_poses(ids, poses)
        dh = sparse.csr_matrix(dh)
        uncertainty = sparse.csr_matrix(self.uncertainty)

        r = (
            sparse.eye(dh.shape[0], format="csc") * R_UNCERTAINTY
        )  # measurement uncertainty
        s = dh @ uncertainty @ dh.T + r  # innovation covariance

        s_inv = sparse.linalg.spsolve(s, sparse.eye(s.shape[0], format="csc"))
        kalman_gain = uncertainty @ dh.T @ s_inv

        innovation = kalman_gain @ (z - h)

        # additive update for linear components of the state
        self.state[XYZ_DIMS] += innovation[XYZ_DIMS]

        # quaternion multiplication for rotational components
        q = self.state[QUAT_DIMS]
        dq = [1, *innovation[ERROR_DIMS] / 2]  # small angle approximation

        # apply the correction to the accumulative quaternion
        q = Rotation.from_quat(q, scalar_first=True)
        dq = Rotation.from_quat(dq, scalar_first=True)
        q = dq * q

        self.state[QUAT_DIMS] = q.as_quat(scalar_first=True)

        # explicit reset, but not necessarily needed
        self.state[ERROR_DIMS] = [0, 0, 0]

        # # update for state for camera + detected landmarks
        for idx in range(self.num_landmarks):
            lm_i = LM_DIMS * idx + CAM_DIMS

            xyz_dims = slice(lm_i, lm_i + 3)  # x, y, z
            quat_dims = slice(lm_i + 3, lm_i + 7)
            error_dims = slice(lm_i + 7, lm_i + 10)  # ex, ey, ez

            # update the translation
            self.state[xyz_dims] += innovation[xyz_dims]

            # update the quaternion
            lm_q = self.state[quat_dims]
            lm_dq = [1, *innovation[error_dims] / 2]
            # small angle approximation
            lm_q = Rotation.from_quat(lm_q, scalar_first=True)
            lm_dq = Rotation.from_quat(lm_dq, scalar_first=True)
            lm_q = lm_dq * lm_q
            self.state[quat_dims] = lm_q.as_quat(scalar_first=True)

        # update uncertainty
        ident = np.eye(LM_DIMS * self.num_landmarks + CAM_DIMS)
        self.uncertainty = (ident - kalman_gain @ dh) @ self.uncertainty

    def parse_poses(
        self,
        ids: list[int],
        poses: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse the observations into observed, predicted, and jacobian values.

        Arguments:
            ids: list of ids of the markers
            poses: list of poses of the markers

        Returns:
            z: the observed values
            h: the predicted values
            dh: the jacobian matrix

        """
        # perform update step
        z = None
        h = None
        dh = None
        for idx, pose in zip(ids, poses):
            index = self.landmarks[idx]

            jacobian_row = self.landmark_dh(index)

            cam_state = self.state[:CAM_DIMS]  # x, y, z, roll, pitch, yaw
            state_index = LM_DIMS * index + CAM_DIMS
            landmark_state = self.state[state_index : state_index + LM_DIMS]

            h_row = self.h([*cam_state, *landmark_state]).squeeze()

            orientation = Rotation.from_euler(
                "xyz",
                pose[3:],
            ).as_quat(scalar_first=True)

            # get the current orientation
            lm_i = LM_DIMS * index + CAM_DIMS
            state_quat = self.state[lm_i + 3 : lm_i + 7]  # qw, qx, qy, qz

            norm = np.linalg.norm(
                orientation - state_quat,
            )

            # check if the orientation is too different
            if norm > QUAT_THRESHOLD:
                # if so, skip this observation
                print(
                    f"Skipping landmark {idx} "
                    f"due to large orientation difference ({norm:.3f})",
                )
                continue
            # convert to quaternion
            pose_quat = np.hstack(
                (pose[XYZ_DIMS], orientation),
            )

            # add to the matrices
            if z is None:
                z = pose_quat  # x, y, z, qw, qx, qy, qz
                h = h_row
                dh = jacobian_row
            else:
                z = np.hstack((z, pose_quat))
                h = np.hstack((h, h_row))
                dh = np.vstack((dh, jacobian_row))

        return z, h, dh

    def landmark_dh(
        self,
        index: int,
    ) -> np.ndarray:
        """Compute the jacobian for a specific landmark.

        Places it in the correct position in the full jacobian matrix

        Arguments:
            index: the index of the landmark
            pose: the pose of the landmark

        Returns:
            dh: the partial jacobian

        """
        cam_state = self.state[:CAM_DIMS]  # x, y, z, qw, qx, qy, qz, ex, ey, ez

        beginning_index = LM_DIMS * index + CAM_DIMS

        # x, y, z
        landmark_state = self.state[beginning_index : beginning_index + LM_DIMS]

        # get the partial jacobian
        jacobian = self.partial_jacobian([*cam_state, *landmark_state])

        camera_jacobian = jacobian[:, :CAM_DIMS]
        landmark_jacobian = jacobian[:, CAM_DIMS:]

        # get the H matrix
        dh = np.zeros((7, LM_DIMS * self.num_landmarks + CAM_DIMS))
        dh[:, :CAM_DIMS] = camera_jacobian

        index = LM_DIMS * index + CAM_DIMS
        dh[:, index : index + LM_DIMS] = landmark_jacobian
        return dh

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
        self.landmarks[idx] = self.num_landmarks
        self.num_landmarks += 1

        camera_pose = self.state[:CAM_DIMS]
        camera_translation = camera_pose[XYZ_DIMS]
        camera_rotation = camera_pose[QUAT_DIMS]

        # TODO(ssilver): this may be wrong. Since all # noqa: TD003 FIX002
        # markers are added when the camera has nearly zero rotation,
        # a new demo will be needed to test this update the state

        rot_mc = Rotation.from_quat(
            camera_rotation,
            scalar_first=True,
        ).as_matrix()
        rot_cm = np.linalg.inv(rot_mc)

        xyz_cl = pose[XYZ_DIMS]  # landmark position in camera frame
        rot_cl = Rotation.from_euler(
            "xyz",
            pose[QUAT_DIMS],
        ).as_matrix()

        # put the landmark's pose in map frame
        t_ml = rot_cm @ xyz_cl + camera_translation
        q_ml = rot_cm @ rot_cl
        q_ml = Rotation.from_matrix(q_ml).as_quat(scalar_first=True)
        t_ml = np.hstack((t_ml, q_ml))
        t_ml = np.hstack((t_ml, np.zeros(3)))  # add error correction

        self.state = np.hstack((self.state, t_ml))

        # expand the uncertainity matricies
        n_dims = LM_DIMS * self.num_landmarks + CAM_DIMS

        new_uncertainty = np.eye(n_dims)
        if uncertainity is not None:
            new_uncertainty[-LM_DIMS:, -LM_DIMS:] *= uncertainity
        else:
            new_uncertainty[-LM_DIMS:, -LM_DIMS:] *= (
                INITIAL_LANDMARK_UNCERTAINTY
            )

        new_uncertainty[: n_dims - LM_DIMS, : n_dims - LM_DIMS] = (
            self.uncertainty
        )
        self.uncertainty = new_uncertainty

    def initialize_h(self) -> None:
        """Initialize the H and h functions/lambdas.

        Uses scipy's symbolic math to generate lambdas.

        Returns:
            h: the h function, which maps the state to the observed values
            dh: the jacobian of the h function

        """
        cache_file = "/tmp/ekf_h_dh_cache2.dill"
        try:
            with open(cache_file, "rb") as f:
                h, dh = dill.load(f)
        except FileNotFoundError:
            # if True:
            # Define translation and rotation variables
            x_mc, y_mc, z_mc = sp.symbols("x_r^m y_r^m z_r^m")
            qx_mc, qy_mc, qz_mc, qw_mc = sp.symbols("qx_mc qy_mc qz_mc qw_mc")
            ex_mc, ey_mc, ez_mc = sp.symbols("ex_mc ey_mc ez_mc")

            x_ml, y_ml, z_ml = sp.symbols("x_l^m y_l^m z_l^m")
            qx_ml, qy_ml, qz_ml, qw_ml = sp.symbols("qx_ml qy_ml qz_ml qw_ml")
            ex_ml, ey_ml, ez_ml = sp.symbols("ex_ml ey_ml ez_ml")

            # error corrected rotations
            dq_mc = sp.Quaternion(1, ex_mc, ey_mc, ez_mc)
            q_mc = sp.Quaternion(qw_mc, qx_mc, qy_mc, qz_mc)

            dq_ml = sp.Quaternion(1, ex_ml, ey_ml, ez_ml)
            q_ml = sp.Quaternion(qw_ml, qx_ml, qy_ml, qz_ml)

            # hamilton product
            ecq_mc = dq_mc * q_mc
            ecq_ml = dq_ml * q_ml

            # Rotation matrices
            rot_mc = ecq_mc.to_rotation_matrix()
            rot_cm = rot_mc.inv()

            # Define state vectors
            xyz_mc = sp.Matrix([x_mc, y_mc, z_mc])
            xyz_ml = sp.Matrix([x_ml, y_ml, z_ml])

            # Define the function to compute landmark position in camera frame
            xyz_cl = rot_cm @ (xyz_ml - xyz_mc)

            ecq_cm = ecq_mc.inverse()
            q_cl = ecq_cm.mul(ecq_ml)

            function = sp.Matrix.vstack(
                xyz_cl,  # landmark position in camera frame
                q_cl.to_Matrix(),
            )

            variables = sp.Matrix(
                [
                    x_mc,  # cam state
                    y_mc,
                    z_mc,
                    qw_mc,
                    qx_mc,
                    qy_mc,
                    qz_mc,
                    ex_mc,
                    ey_mc,
                    ez_mc,
                    x_ml,  # landmark state
                    y_ml,
                    z_ml,
                    qw_ml,
                    qx_ml,
                    qy_ml,
                    qz_ml,
                    ex_ml,
                    ey_ml,
                    ez_ml,
                ],
            )

            # Lambdify the function
            h = sp.lambdify([variables], function, modules=["numpy"])
            # Compute the Jacobian
            jacobian = function.jacobian(variables)
            # Lambdify the Jacobian
            dh = sp.lambdify([variables], jacobian, modules=["numpy"])
            with open(cache_file, "wb") as f:
                dill.dump((h, dh), f)

        # return the lambdas
        return h, dh

    def get_lm_estimates(self) -> np.ndarray:
        """Return the estimates of the landmarks."""
        return self.landmarks.items()
