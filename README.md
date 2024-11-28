# aruco-slam

This project aims to use ArUco markers to perform SLAM in a 3D environment. Insipired by UcoSLAM.

Video:
![Aruco SLAM](pangolin.avi)


Due to possible ambiguities in ArUco orientation estimation, only marker positions are used in the EKF. These orientation instabilities can be seen in the video and is discussed more thoroughly in [this OpenCV github issue](https://github.com/opencv/opencv/issues/8813). In the future, I may opt to use the stable dimensions of the orientation estimation as part of the state. 