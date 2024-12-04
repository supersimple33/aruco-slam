# ArUco SLAM

This project aims to use ArUco markers to perform SLAM with a handheld camera in a 3D environment. With no motion model, the predicted motion relies exclusively on ArUco marker measurements.

The gifs below may take a second to load.

## Extended Kalman Filter:

![Aruco SLAM](outputs/ekf.gif)

## Factor Graph:

![GTSAM Factor Graph](outputs/factorgraph.gif)

## Orientation Ambiguities

Due to possible ambiguities in ArUco orientation estimation, only marker positions are used (hence why they are represented as points). These orientation ambiguities can be seen in the video (`z` axis flipping) and are discussed more thoroughly in [this OpenCV github issue](https://github.com/opencv/opencv/issues/8813). In the future, I may opt to use the stable `x` and `y` dimensions as part of the state for better results.

## TODOs

- [x] ArUco Detection, Pose Estimation 
- [x] EKF
- [ ] Quaternions, Angle-Wrap Handling in EKF
- [ ] UKF
- [ ] Iteratative EKF
- [x] GTSAM, ISAM2 (Factor Graph)
- [ ] Particle Filter
- [x] 3D Visualization
- [ ] Map Saving, Loading
- [ ] Trajectory Saving
- [ ] Ground Truth Comparison