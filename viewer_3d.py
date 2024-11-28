import pangolin
import OpenGL.GL as gl

import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Array

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

LM_POINT_SIZE = 10
POSE_POINT_SIZE = 5

DISPLAY_SIZE = 960, 540

class Viewer3D():
    def __init__(self, image_size, export_video=False):
        width, height  = image_size

        self.q = Queue()
        self.scale = 1
        
        w, h = DISPLAY_SIZE
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        viewpoint_x =   0 * self.scale
        viewpoint_y = 1 * self.scale
        viewpoint_z = -10 * self.scale
        viewpoint_f = 1000
            
        self.proj = pangolin.ProjectionMatrix(w, h, viewpoint_f, viewpoint_f, w//2, h//2, 0.1, 5000)
        self.look_view = pangolin.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)
        self.scam = pangolin.OpenGlRenderState(self.proj, self.look_view)
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        handler = pangolin.Handler3D(self.scam)
        self.dCam = (
            pangolin.CreateDisplay()
            .SetBounds(
                pangolin.Attach(0),
                pangolin.Attach(1),
                pangolin.Attach.Pix(0),
                pangolin.Attach(1),
                -float(width) / float(height),
            )
            .SetHandler(handler)
        )

        self.poses = []

        self.T = pangolin.OpenGlMatrix()

        self.export_video = export_video
        if export_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
            self.video_writer = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
            pangolin.SaveWindowOnRender('pangolin')

    def view(self, camera_position, points, points_detected):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.dCam.Activate(self.scam)

        # Draw camera
        pose = np.eye(4)
        camera_rotation = R.from_euler('xyz', camera_position[3:6]) \
            .as_matrix() \
            .copy()
        camera_translation = camera_position[:3].copy()
        pose[:3, :3] =  camera_rotation.copy()
        pose[:3, 3] = camera_translation
        self.T.m = pose
        self.scam.Follow(self.T, True)
        gl.glColor3f(0.0, 0.0, 0.0)
        pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)

        # draw, record the translation
        gl.glPointSize(POSE_POINT_SIZE)
        gl.glColor3f(0.0, 0.0, 1.0)
        self.draw_poses(self.poses)
        self.poses.append(camera_translation)

        # draw state points
        if points.shape != (0,):
            gl.glPointSize(LM_POINT_SIZE)
            gl.glColor3f(1.0, 0.0, 0.0)
            self.draw_markers(points)


        # draw detected points
        if points_detected.shape != (0,):
            points_detected = points_detected[:, :3].copy()
            points_detected = points_detected @ np.linalg.inv(camera_rotation) + camera_translation
            gl.glPointSize(LM_POINT_SIZE//2)
            gl.glColor3f(0.0, 1.0, 0.0)
            self.draw_markers(points_detected)

        pangolin.FinishFrame()

        if self.export_video:
            pangolin.SaveWindowOnRender('pangolin')
            img = cv2.imread('pangolin.png')
            self.video_writer.write(img)

    def draw_poses(self, poses):
        if len(poses) > 0:
            for point in poses:
                pangolin.DrawPoints([point])
        
    # def draw_markers_boxes(self, markers):
    #     if len(markers) > 0:
    #         poses = []
    #         sizes = []
    #         for p in markers:
    #             pose = np.eye(4)
    #             pose[:3, :3] = R.from_euler('xyz', p[3:6]).as_matrix()
    #             pose[:3, 3] = p[:3]
    #             poses.append(pose)
    #             size = np.array([0.1, 0.1, 0.01])
    #             sizes.append(size)
    #         pangolin.DrawBoxes(poses, sizes)

    def draw_markers(self, markers):
        if len(markers) > 0:
            pangolin.DrawPoints(np.array(markers))

    def drawPlane(self, num_divs=200, div_size=10, scale=1.0):
        gl.glLineWidth(0.5)
        # Plane parallel to x-z at origin with normal -y
        div_size = scale*div_size
        minx = -num_divs*div_size
        minz = -num_divs*div_size
        maxx = num_divs*div_size
        maxz = num_divs*div_size
        #gl.glLineWidth(2)
        #gl.glColor3f(0.7,0.7,1.0)
        gl.glColor3f(0.7,0.7,0.7)
        gl.glBegin(gl.GL_LINES)
        for n in range(2*num_divs):
            gl.glVertex3f(minx+div_size*n,0,minz)
            gl.glVertex3f(minx+div_size*n,0,maxz)
            gl.glVertex3f(minx,0,minz+div_size*n)
            gl.glVertex3f(maxx,0,minz+div_size*n)
        gl.glEnd()
        gl.glLineWidth(1)