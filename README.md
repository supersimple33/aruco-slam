# ArUco SLAM

This project aims to use ArUco markers to perform SLAM in a 3D environment.

![Aruco SLAM](output.gif)

## Orientation Ambiguities

Due to possible ambiguities in ArUco orientation estimation, only marker positions are used (hence why they are represented as points). These orientation ambiguities can be seen in the video (`z` axis flipping) and are discussed more thoroughly in [this OpenCV github issue](https://github.com/opencv/opencv/issues/8813). In the future, I may opt to use the stable `x` and `y` dimensions as part of the state for better results.

## TODOs

- [x] EKF
- [x] ArUco Detection, Pose Estimation 
- [x] 3D Visualization
- [ ] Quaternions, Angle-Wrap Handling
- [ ] GTSAM, ISAM2
- [ ] Map Saving, Loading
- [ ] Trajectory Saving
- [ ] Ground Truth Comparison