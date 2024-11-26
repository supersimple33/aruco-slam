import numpy as np
import cv2
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr

from kalman_filter import KF

class ArucoSlam():
    def __init__(self, initial_pose, calib_matrix, dist_coeffs):
        self.calib_matrix = calib_matrix
        self.dist_coeffs = dist_coeffs
        self.detector = self.init_aruco_detector()

        self.filter = KF(initial_pose)

        self.camera_pose = initial_pose

    def init_aruco_detector(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.cornerRefinementWinSize = 5
        aruco_params.cornerRefinementMaxIterations = 30
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 30
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        return detector
    
        
    def transform_from_rvec_tvec(self, rvec, tvec):
        # cv2.Rodrigues(rvec.squeeze())[0] == pr.matrix_from_compact_axis_angle(rvec.squeeze())
        return pt.transform_from(
            pr.matrix_from_compact_axis_angle(rvec), 
            tvec
        )

    def rvec_tvec_from_transform(self, transform):
        rvec = pr.compact_axis_angle_from_matrix(transform[:3,:3])
        tvec = transform[:3,3]
        return rvec, tvec

    def estimate_pose_of_markers(self, corners, marker_size):
        '''
        https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get
        '''

        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

        tvecs = []
        rvecs = []
        
        for c in corners:
            n, R, t = cv2.solvePnP(
                marker_points,
                c,
                self.calib_matrix,
                self.dist_coeffs,
                False, 
                cv2.SOLVEPNP_IPPE
            )

            tvecs.append(t.flatten())
            rvecs.append(R.flatten())

        tvecs = np.array(tvecs)
        rvecs = np.array(rvecs)

        poses = np.hstack((tvecs, rvecs))

        return poses

    def process_frame(self, frame):
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(frame)

        poses = []
        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            ids = ids.flatten()

            poses = self.estimate_pose_of_markers(corners, 0.05)

            self.filter.observe(ids, poses)

        return frame, self.filter.state, poses
    
    