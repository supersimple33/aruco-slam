import pangolin
import OpenGL.GL as gl

import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Array

import numpy as np
from scipy.spatial.transform import Rotation as R

kUiWidth = 180
kDefaultPointSize = 5
kViewportWidth = 1024
#kViewportHeight = 768
kViewportHeight = 550
kDrawCameraPrediction = False   
kDrawReferenceCamera = True   

kMinWeightForDrawingCovisibilityEdge=100

kAlignGroundTruthEveryNKeyframes = 10


class Viewer3D():
    def __init__(self, image_size):
        width, height  = image_size

        self.q = Queue()
        self.scale = 1
        
        w, h = kViewportWidth, kViewportHeight
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        viewpoint_x =   0 * self.scale
        viewpoint_y = -1 * self.scale
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

        self.T = pangolin.OpenGlMatrix()

    def view(self, camera_position, points, points_detected):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.dCam.Activate(self.scam)

        self.drawPlane(200, 10, self.scale)

        # Draw camera
        pose = np.eye(4)
        # pose[:3, :3] =  R.from_euler('xyz', camera_position[3:6]).as_matrix()
        # pose[:3, 3] = camera_position[:3]
        self.T.m = pose
        self.scam.Follow(self.T, True)
        pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)

        # draw state points
        gl.glPointSize(kDefaultPointSize)
        gl.glColor3f(1.0, 0.0, 0.0)
        self.draw_markers(points)

        # draw detected points
        gl.glPointSize(kDefaultPointSize)
        gl.glColor3f(0.0, 1.0, 0.0)
        self.draw_markers(points_detected)

        pangolin.FinishFrame()

    def draw_markers(self, markers):
        if len(markers) > 0:
            poses = []
            sizes = []
            for p in markers:
                pose = np.eye(4)
                pose[:3, :3] = R.from_euler('xyz', p[3:6]).as_matrix()
                pose[:3, 3] = p[:3]
                poses.append(pose)
                size = np.array([0.1, 0.1, 0.01])
                sizes.append(size)
            pangolin.DrawBoxes(poses, sizes)

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