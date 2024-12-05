
<div align=center>
  <img src="outputs/icon.png" width="150" height="150"/>
</div>

  
<h1 align="center">ArUco SLAM</h1>


<div align=center>

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/) -->


  **Using various algorithms to perform SLAM with a monocular camera and visual markers.** 

![Aruco SLAM](outputs/factorgraph.gif)
</div>

## About

**ArUco** markers are a type of fiducial marker commonly used in computer vision applications, especially because they are built into OpenCV. Since their size is known, their corners can be used with the PnP algorithm to estimate their pose in the camera frame. Additionally, ArUco markers individually encode an ID. Together, these properties make them ideal for SLAM applications.

**SLAM** stands for Simultaneous Localization and Mapping. This means that the system can localize itself in an environment while simultaneously building out its understanding of that environment. This project detects the position and orientation of ArUco markers in a video feed, inserts those markers into a map, and then uses various methods to optimize the estimates for both the camera and marker positions. In the video above, all markers are placed randomly; before the first frame is processed, there is no information about their positions.

**Why is this important?** SLAM is a fundamental problem in robotics and computer vision. It is used in self-driving cars, drones, and augmented reality applications. Will this project be used in any of those applications? Probably not. But it is a fun and challenging problem to work on.

## Extended Kalman Filter:

Due to (non-linear) rotations of the camera, a Kalman Filter cannot be used. 

The key components of the Extended Kalman Filter are as follows:
- **State Vector**: the 3D pose of the camera, along with the 3D position of each ArUco marker (all in the map frame):
  - $x_{cam}, y_{cam}, z_{cam}, roll_{cam}, pitch_{cam}, yaw_{cam}, x_{m0}, y_{m0}, z_{lm0}, x_{m1}, y_{m1}, z_{m1}, ...$
  - There will be $3n + 6$ dimensions, for $n$ landmarks
- **Measurement** Vector: the 3D position of each ArUco marker in the camera frame:
  - ${}^{cam}x_{mi},{}^{cam}y_{mi},{}^{cam}z_{mi}$    
- **State Transition**: since there is no motion model, we only add uncertainty to the camera's state. Since the landmarks are static, we don't modify them at all.
  - $X_{k|k-1} = X_{k-1|k-1}$
- **Measurement Model**: in order to model what we will measure, we get the displacement between the landmark and the camera positions ($X$), and then rotate it to put it in the camera frame:
  - ${}^{cam}X_{marker} = {}^{cam}R_{map} \cdot ({}^{map}X_{marker} - {}^{map}X_{cam})$
 
There is an excellent explanation by Cyrill Stachniss for a similar, 2D example that can be found [here](https://www.youtube.com/watch?v=X30sEgIws0g) [1].

`python3 run_slam.py --kalman`
  

<details>
  <summary><strong>Visualization</strong></summary>
  
![Aruco SLAM](outputs/ekf.gif)
</details>


## Factor Graph:
<div align="center">
  <img src="https://gtsam.org/assets/fg-images/image1.png" width="450" height="300"/>
  <p><em>Image credit: GTSAM [2]</em></p>
</div>

A factor graph optimizes the joint probability:

<div align="center">

$P(X \mid Z) \propto \prod_{i} \phi_i(X_i, Z_i)$
</div>

Where:
- $X$: the camera poses and the landmark positions.
- $Z$: the measured landmark poses in the camera frame.
- $\phi_i$: the factor that relates one camera pose to the next as well as each 
camera pose to the landmark that were seen at that time.

In other words, we can estimate the camera and landmark positions by optimizing 
the posterior probability.

It should also be noted that, similar to the EKF, the factor graph does not 
have a motion model. Therefore, the factors connecting sequential camera poses 
are zero change with high uncertainty, thus weighting the measurements more 
heavily.

My implementation leverages GTSAM with the ISAM2 solver, following the 
postulation above. It reconstructs the graph at each timestep, maintaining 
conciseness while also accounting for both the local environment and historical
constraints.

`python3 run_slam.py --factorgraph`

<details>
  <summary><strong>Visualization</strong></summary>

This is the same as the gif shown at the top of the README.
  
![GTSAM Factor Graph](outputs/factorgraph.gif)
</details>


## Orientation Ambiguities

Due to possible ambiguities in ArUco orientation estimation, only marker 
positions are used (hence why they are represented as points). These 
orientation ambiguities can be seen in the video (`z` axis flipping) and are 
discussed more thoroughly in 
[this OpenCV github issue](https://github.com/opencv/opencv/issues/8813). In 
the future, I may opt to use the stable `x` and `y` dimensions as part of the 
state for better results.

## TODOs

- [x] ArUco Detection, Pose Estimation 
- [x] EKF
- [ ] Quaternions, Angle-Wrap Handling in EKF
- [ ] UKF
- [ ] Iteratative EKF
- [x] Factor Graph
- [ ] Particle Filter
- [x] 3D Visualization
- [ ] Map Saving, Loading
- [ ] Trajectory Saving
- [ ] Ground Truth Comparison
- [ ] Duplicate Marker ID Handling
- [ ] Non-Static Landmark Tracking

## References

[1] Cyrill Stachniss. (2020, October 2). EKF-SLAM (Cyrill Stachniss). YouTube. https://www.youtube.com/watch?v=X30sEgIws0g

[2] Dellaert, F., & GTSAM Contributors. (2022, May). *borglab/gtsam* (Version 4.2a8) [Software]. Georgia Tech Borg Lab. https://doi.org/10.5281/zenodo.5794541