import numpy as np
import cv2 as cv
import glob

# SHAPE = (3840, 2160)
SHAPE = 1920, 1080

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./images/*.jpg')
for fname in images:
    img = cv.imread(fname)

    h, w, c = img.shape
    
    if h > w:
    # if img_0.shape[1] > img_0.shape[0]:
        img = cv.transpose(img)
        img = cv.flip(img, 1)

    img = cv.resize(img, SHAPE)

    # print(img.shape)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (6,9), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(10)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx)

print(dist)

# TODO IMPORTANT: FIX THIS
mtx = np.array([[1.39126191e+03, 0.00000000e+00, 9.73151367e+02],
                [0.00000000e+00, 1.39443582e+03, 5.36453230e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],])

dist = np.array([ 0.03393005, -0.01507052, -0.00097307,  0.00057635, -0.04426605])


# save the camera matrix and distortion coefficients
np.save('calib_mtx.npy', mtx)
np.save('calib_dist.npy', dist)