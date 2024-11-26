import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation

INITIAL_CAMERA_UNCERTAINTY = 0.8
INITIAL_LANDMARK_UNCERTAINTY = 0.3

CAMERA_DIMS = 6 # x, y, z, roll, pitch, yaw
LANDMARK_DIMS = 6 # x, y, z, roll, pitch, yaw

def get_rotation_matrix(r):
    return Rotation.from_euler('xyz', r).as_matrix()


class KF():
    """
    Implements a Kalman Filter that tracks the positions of the cameras and 
    the aruco landmarks
    """

    def __init__(self, initial_camera_pose):
        """
        initializes filter
        """
        self.measurement_dims = 6 # x, y, z, roll, pitch, yaw

        self.position = initial_camera_pose[:3]
        self.rotation = initial_camera_pose[3:]

        # 6n + 6, where n is the number of landmarks
        self.state = np.array(initial_camera_pose)

        # system model
        self.F = np.eye(self.measurement_dims)
        self.G = np.eye(self.measurement_dims)

        # uncertainity matrices (system, prediction)
        self.Q = np.eye(self.measurement_dims) * INITIAL_CAMERA_UNCERTAINTY
        self.R = np.eye(self.measurement_dims) * INITIAL_CAMERA_UNCERTAINTY

        self.Sigma_p = np.eye(self.measurement_dims) * INITIAL_CAMERA_UNCERTAINTY

        self.num_landmarks = 0
        self.landmarks = {}

        self.initialize_partial_jacobian()

    def observe(self, ids, poses):
        for idx, pose in zip(ids, poses):
            if idx in self.landmarks:
                continue
            else:
                self.add_marker(idx, pose)
            
        # perform update step
        stack = None
        for idx, pose in zip(ids, poses):
            index = self.landmarks[idx]

            prediction = self.H(index, pose)
            if stack is None:
                stack = prediction
            else:
                stack = np.vstack((stack, prediction))

        rows, cols = stack.shape
        assert rows == len(ids) * LANDMARK_DIMS, \
            "Incorrect number of rows in jacobian."
        assert cols == 6*self.num_landmarks + 6, \
            "Incorrect number of columns in jacobian."
        
        H = stack
        S = self.S(H)
        K = self.K(H, S)
        Sigma = self.Sigma(H, K)

        z = np.hstack(poses)

        innovation = K @ (z - H @ self.state)

        self.state += innovation
        self.Sigma_p = Sigma

        # print(self.state[:3])

    def K(self, H, S):
        return self.Sigma_p @ H.T @ np.linalg.inv(S)

    def S(self, H):
        # construct sensor noise
        R = np.eye(H.shape[0]) * INITIAL_LANDMARK_UNCERTAINTY

        return H @ self.Sigma_p @ H.T + R

    def Sigma(self, H, K):
        I = np.eye(6*self.num_landmarks + 6)
        return (I - K @ H) @ self.Sigma_p

    def H(self, index, pose):
        """
        """

        # get the camera pose in map space
        camera_rotation = self.state[3:6]
        camera_translation = self.state[:3]

        m_x_c, m_y_c, m_z_c = camera_translation
        m_phi_c, m_theta_c, m_psi_c = camera_rotation

        beginning_index = 6*index + 6
        state = self.state[beginning_index:beginning_index+6]
        translation = state[:3]
        rotation = state[3:]

        m_x_l, m_y_l, m_z_l = translation
        m_phi_l, m_theta_l, m_psi_l = rotation

        # get the partial jacobian
        jacobian = self.partial_jacobian(
            [m_x_c, m_y_c, m_z_c, m_phi_c, m_theta_c, m_psi_c,
             m_x_l, m_y_l, m_z_l, m_phi_l, m_theta_l, m_psi_l]
        )

        camera_jacobian = jacobian[:, :6]
        landmark_jacobian = jacobian[:, 6:]

        # get the H matrix
        H = np.zeros((6, 6*self.num_landmarks + 6))
        H[:, :6] = camera_jacobian
        H[:, 6*index+6:6*index+12] = landmark_jacobian
        return H

        
    def add_marker(self, idx, pose):
        self.landmarks[idx] = self.num_landmarks
        self.num_landmarks += 1

        camera_pose = self.state[:6]
        camera_translation = camera_pose[:3]
        camera_rotation = camera_pose[3:]

        # update the state
        R = get_rotation_matrix(camera_rotation)
        R = np.linalg.inv(R)
        
        # put pose in map frame
        t = R @ pose[:3] + camera_translation
        r = R @ pose[3:]

        self.state = np.hstack((self.state, t, r))

        # expand the uncertainity matricies
        n = self.num_landmarks
        new_Q = np.eye(6*n + 6) * INITIAL_LANDMARK_UNCERTAINTY
        new_Q[:6*n, :6*n] = self.Q
        self.R = self.Q
        self.Q = new_Q
        self.Sigma_p = new_Q

        # expand the jacobian matrix
        new_F = np.zeros((6*n + 6, 6*n + 6))
        new_F[:6*n, :6*n] = self.F
        self.F = new_F
        self.G = new_F

    def initialize_partial_jacobian(self):
        # Define translation and rotation variables
        m_x_r, m_y_r, m_z_r = sp.symbols('x_r^m y_r^m z_r^m')
        m_x_l, m_y_l, m_z_l = sp.symbols('x_l^m y_l^m z_l^m')
        m_phi_r, m_theta_r, m_psi_r = sp.symbols('phi_r^m theta_r^m psi_r^m')
        m_phi_l, m_theta_l, m_psi_l = sp.symbols('phi_l^m theta_l^m psi_l^m')

        # Define the forward parameter
        forward = -1  # Assuming forward = 1 for simplicity

        # Rotation matrices
        r_R_m_x = sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(forward * m_phi_r), -sp.sin(forward * m_phi_r)],
            [0, sp.sin(forward * m_phi_r), sp.cos(forward * m_phi_r)]
        ])

        r_R_m_y = sp.Matrix([
            [sp.cos(forward * m_theta_r), 0, sp.sin(forward * m_theta_r)],
            [0, 1, 0],
            [-sp.sin(forward * m_theta_r), 0, sp.cos(forward * m_theta_r)]
        ])

        r_R_m_z = sp.Matrix([
            [sp.cos(forward * m_psi_r), -sp.sin(forward * m_psi_r), 0],
            [sp.sin(forward * m_psi_r), sp.cos(forward * m_psi_r), 0],
            [0, 0, 1]
        ])

        m_R_r = r_R_m_x * r_R_m_y * r_R_m_z

        # Convert to 6x6 matrix
        m_R_r = m_R_r.row_join(sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        m_R_r = m_R_r.col_join(sp.Matrix(
            [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]))

        # Define state vectors
        m_X_r = sp.Matrix([m_x_r, m_y_r, m_z_r, m_phi_r, m_theta_r, m_psi_r])
        m_X_l = sp.Matrix([m_x_l, m_y_l, m_z_l, m_phi_l, m_theta_l, m_psi_l])

        # Define the function
        sub = m_X_l - m_X_r
        function = m_R_r * sub

        # Define all variables
        variables = sp.Matrix([
            m_x_r, m_y_r, m_z_r, m_phi_r, m_theta_r, m_psi_r, 
            m_x_l, m_y_l, m_z_l, m_phi_l, m_theta_l, m_psi_l
        ])

        # Compute the Jacobian
        jacobian = function.jacobian(variables)

        # Lambdify the Jacobian
        jacobian_func = sp.lambdify([variables], jacobian, modules=['numpy'])

        self.partial_jacobian = jacobian_func

def main():
    kf = KF([0, 0, 0, 0, 0, 0])

if __name__ == "__main__":
    main()