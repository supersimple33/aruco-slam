"""
Module for creating 3D visualization

IMPORTANT:
pangolin needs to have setup.py changed:
all instances of 'install_dirs' -> 'install_dir'
"""

from typing import Tuple, List
import pangolin
import OpenGL.GL as gl

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

LM_POINT_SIZE = 10
POSE_POINT_SIZE = 5

DISPLAY_SIZE = 960, 540

VIDEO_NAME = "output_3d.mp4"

class Viewer3D():
    """
    Wrapper class for visualizing 3D state with Pangolin
    """

    def __init__(
            self,
            image_size: Tuple[int, int],
            export_video:bool
            ) -> None:
        """
        Constructor

        params:
        - image_size: size of OpenCV image mat
        - export_video: whether or not to create video

        returns:
        - None
        """

        # get size
        width, height  = image_size

        self.scale = 1

        w, h = DISPLAY_SIZE
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        viewpoint_x =   0 * self.scale
        viewpoint_y = 3 * self.scale
        viewpoint_z = -10 * self.scale
        viewpoint_f = 1000

        self.proj = pangolin.ProjectionMatrix(
            w,
            h,
            viewpoint_f,
            viewpoint_f,
            w//2,
            h//2,
            0.1,
            5000
        )
        self.look_view = pangolin.ModelViewLookAt(
            viewpoint_x,
            viewpoint_y,
            viewpoint_z,
            0,
            0,
            0,
            0,
            -1,
            0
        )
        self.scam = pangolin.OpenGlRenderState(self.proj, self.look_view)
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        handler = pangolin.Handler3D(self.scam)
        self.d_cam = (
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

        self.transform = pangolin.OpenGlMatrix()

        self.export_video = export_video
        if export_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
            self.video_writer = cv2.VideoWriter(VIDEO_NAME, fourcc, 20.0, DISPLAY_SIZE)
            pangolin.SaveWindowOnRender('pangolin')

    def close(self):
        """
        Destroys the viewer object
        """

        self.video_writer.release()

    def view(
            self,
            camera_position: np.ndarray,
            points: List[np.ndarray],
            points_detected: List[np.ndarray]
            ) -> None:
        """
        Draws the 3D window of the state

        params:
        - camera_position: pose of camera
        - points: estimated pose of markers
        - points_detected: list of markers as they're seen.
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.d_cam.Activate(self.scam)

        # Draw camera
        pose = np.eye(4)
        camera_rotation = R.from_euler('xyz', camera_position[3:6]) \
            .as_matrix() \
            .copy()
        camera_translation = camera_position[:3].copy()
        pose[:3, :3] =  camera_rotation.copy()
        pose[:3, 3] = camera_translation
        self.transform.m = pose
        self.scam.Follow(self.transform, True)
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

    def draw_poses(
            self,
            poses: List[np.ndarray]
            ) -> None:
        """
        draws poses as points

        params:
        - poses: list of poses

        returns:
        - None: just draws
        """
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

    def draw_markers(
            self,
            markers: List[np.ndarray]
            ) -> None:
        """
        Draws markers as points in GL window.

        params:
        - markers: list of points, locations

        returns:
        - None: just draws
        """
        if len(markers) > 0:
            pangolin.DrawPoints(np.array(markers))

    def draw_plane(
            self,
            num_divs:int = 200,
            div_size:int = 10,
            scale:int = 1.0
            ) -> None:
        """
        Draws plane on OpenGL Window

        params:
        - num_divs: number of divs to draw on x, y
        - div_size: width (m) of grid cells
        - scale: possible scaling
        """

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
