import cv2

import numpy as np

AXIS_SIZE = 0.1

class Viewer2D():
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def view(self, frame, camera_position, points, points_detected):
        """
        """

        # draw the camera
        # frame = self.draw_camera(frame, camera_position)

        # draw the points
        for p in points_detected:
            frame = self.draw_axis(frame, p[3:6], p[:3])

        return frame


    def draw_axis(self, img, R, t):
        """
        https://stackoverflow.com/questions/30207467/how-to-draw-3d-coordinate-axes-with-opencv-for-face-pose-estimation
        """

        # unit is mm
        # rotV, _ = cv2.Rodrigues(R)
        rotV = R

        points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
        points *= AXIS_SIZE
        
        axisPoints, _ = cv2.projectPoints(points, rotV, t, self.camera_matrix, (0, 0, 0, 0))
       
        point0 = tuple(axisPoints[0].ravel())
        point1 = tuple(axisPoints[1].ravel())
        point2 = tuple(axisPoints[2].ravel())
        point3 = tuple(axisPoints[3].ravel())

        # convert to ints
        point0 = (int(point0[0]), int(point0[1]))
        point1 = (int(point1[0]), int(point1[1]))
        point2 = (int(point2[0]), int(point2[1]))
        point3 = (int(point3[0]), int(point3[1]))
    
        # draw the lines
        thickness = 8
        img = cv2.line(img, point3, point0, (255,0,0), thickness)
        img = cv2.line(img, point3, point1, (0,255,0), thickness)
        img = cv2.line(img, point3, point2, (0,0,255), thickness)
        
        return img