"""Module for creating 3D visualization.

Important:
git clone https://github.com/uoip/pangolin.git
cd pangolin
mkdir build
cd build
cmake .. -DPython_EXECUTABLE=`which python3` -DBUILD_PANGOLIN_FFMPEG=OFF
make -j8
cd ..

pangolin needs to have setup.py changed:
all instances of 'install_dirs' -> 'install_dir'

python setup.py install

"""
from __future__ import annotations

import cv2
import numpy as np
try:
    import pangolin
except ImportError:
    pass
from OpenGL import GL
from scipy.spatial.transform import Rotation

LM_POINT_SIZE = 10
POSE_POINT_SIZE = 5

DISPLAY_SIZE = 960, 540

VIDEO_NAME = "outputs/images/output_3d.mp4"


class Viewer3D:
    """Wrapper class for visualizing 3D state with Pangolin."""

    def __init__(
        self,
        *,
        export_video: bool = False,
    ) -> None:
        """Construct.

        Arguments:
            image_size: size of OpenCV image mat
            export_video: whether or not to create video

        """
        # get size
        width, height = DISPLAY_SIZE

        self.scale = 1

        w, h = DISPLAY_SIZE
        pangolin.CreateWindowAndBind("Map Viewer", w, h)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_BLEND)

        viewpoint_x = 0 * self.scale
        viewpoint_y = 3 * self.scale
        viewpoint_z = -10 * self.scale
        viewpoint_f = 1000

        self.proj = pangolin.ProjectionMatrix(
            w,
            h,
            viewpoint_f,
            viewpoint_f,
            w // 2,
            h // 2,
            0.1,
            5000,
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
            0,
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
            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v",
            )  # Use 'mp4v' for MP4 format
            self.video_writer = cv2.VideoWriter(
                VIDEO_NAME,
                fourcc,
                20.0,
                DISPLAY_SIZE,
            )

    def close(self) -> None:
        """Destroy the viewer object."""
        if self.export_video:
            self.video_writer.release()

    def view(
        self,
        camera_pose: np.ndarray,
        points: list[np.ndarray],
        points_detected: list[np.ndarray],
    ) -> None:
        """Draw the 3D window of the state.

        Arugments:
            camera_position: pose of camera
            points: estimated pose of markers
            points_detected: list of markers as they're seen.
        """
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glClearColor(1.0, 1.0, 1.0, 0.0)

        self.d_cam.Activate(self.scam)

        # Draw camera
        pose = np.eye(4)
        camera_rotation = (
            Rotation.from_quat(camera_pose[3:7], scalar_first=True)
            .as_matrix()
            .copy()
        )
        camera_translation = camera_pose[:3].copy()
        pose[:3, :3] = camera_rotation.copy()
        pose[:3, 3] = camera_translation
        self.transform.m = pose
        self.scam.Follow(self.transform, True)  # noqa: FBT003
        GL.glColor4f(0.0, 0.0, 0.0, 1.0)
        pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)

        # draw, record the translation
        GL.glPointSize(POSE_POINT_SIZE)
        GL.glColor4f(0 / 255, 135 / 255, 68 / 255, 1.0)

        self.draw_poses(self.poses)
        self.poses.append(camera_translation)

        GL.glColor4f(0 / 255, 87 / 255, 231 / 255, 0.5)
        self.draw_markers(np.array([[1e9, 1e9, 1e9]]))

        # draw state points
        if points.shape != (0,):
            GL.glPointSize(LM_POINT_SIZE)

            GL.glColor4f(0 / 255, 87 / 255, 231 / 255, 0.7)
            self.draw_markers(points)

        GL.glFinish()

        GL.glColor4f(214 / 255, 45 / 255, 32 / 255, 0.5)
        self.draw_markers(np.array([[1e9, 1e9, 1e9]]))

        # draw detected points
        if points_detected.shape != (0,):
            points_detected = points_detected[:, :3].copy()
            points_detected = (
                points_detected @ np.linalg.inv(camera_rotation)
                + camera_translation
            )
            GL.glPointSize(LM_POINT_SIZE * 2)
            GL.glColor4f(214 / 255, 45 / 255, 32 / 255, 0.5)
            self.draw_markers(points_detected)

        pangolin.FinishFrame()

        if self.export_video:
            pangolin.SaveWindowOnRender("outputs/images/pangolin")
            img = cv2.imread("outputs/images/pangolin.png")
            self.video_writer.write(img)

    def draw_poses(self, poses: list[np.ndarray]) -> None:
        """Draws poses as points.

        Arguments:
            poses: list of poses

        """
        if len(poses) > 1:
            pangolin.DrawLine(poses, 3)

    def draw_markers(self, markers: list[np.ndarray]) -> None:
        """Draws markers as points in GL window.

        Arguments:
            markers: list of points, locations

        """
        if len(markers) > 0:
            pangolin.DrawPoints(np.array(markers))

    def draw_plane(
        self,
        num_divs: int = 200,
        div_size: int = 10,
        scale: int = 1.0,
    ) -> None:
        """Draws plane on OpenGL Window.

        Arguments:
            num_divs: number of divs to draw on x, y
            div_size: width (m) of grid cells
            scale: possible scaling

        """
        GL.glLineWidth(0.5)
        # Plane parallel to x-z at origin with normal -y
        div_size = scale * div_size
        minx = -num_divs * div_size
        minz = -num_divs * div_size
        maxx = num_divs * div_size
        maxz = num_divs * div_size

        GL.glColor3f(0.7, 0.7, 0.7)
        GL.glBegin(GL.GL_LINES)
        for n in range(2 * num_divs):
            GL.glVertex3f(minx + div_size * n, 0, minz)
            GL.glVertex3f(minx + div_size * n, 0, maxz)
            GL.glVertex3f(minx, 0, minz + div_size * n)
            GL.glVertex3f(maxx, 0, minz + div_size * n)
        GL.glEnd()
        GL.glLineWidth(1)
