# aruco-slam

This project aims to use ArUco markers to perform SLAM in a 3D environment using an Extended Kalman Filter.

![Aruco SLAM](output.gif)

Due to possible ambiguities in ArUco orientation estimation, only marker positions are used in the EKF (hence why they are represented as points). These orientation instabilities can be seen in the video (z axis flipping) and are discussed more thoroughly in [this OpenCV github issue](https://github.com/opencv/opencv/issues/8813). In the future, I may opt to use the stable dimensions (x and y) of the orientation estimation as part of the state. g