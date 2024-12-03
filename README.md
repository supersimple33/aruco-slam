# ArUco SLAM

This project aims to use ArUco markers to perform SLAM in a 3D environment.

![Aruco SLAM](outputs/ekf.gif)

## Orientation Ambiguities

Due to possible ambiguities in ArUco orientation estimation, only marker positions are used (hence why they are represented as points). These orientation ambiguities can be seen in the video (`z` axis flipping) and are discussed more thoroughly in [this OpenCV github issue](https://github.com/opencv/opencv/issues/8813). In the future, I may opt to use the stable `x` and `y` dimensions as part of the state for better results.

## GTSAM Progress:

While the factor graph looks functional, there are clearly significant issues. I believe that these are due to improper orientation constraints. Due to the large amount of noise, I am not marking GTSAM as complete.

<details>
<summary>Trajectory Visualization</summary>

![GTSAM Factor Graph](outputs/factorgraph.gif)

</details>

## TODOs

- [x] EKF
- [x] ArUco Detection, Pose Estimation 
- [x] 3D Visualization
- [ ] Quaternions, Angle-Wrap Handling
- [ ] GTSAM, ISAM2
- [ ] Map Saving, Loading
- [ ] Trajectory Saving
- [ ] Ground Truth Comparison