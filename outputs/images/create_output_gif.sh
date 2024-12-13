#!/bin/bash

ffmpeg -i output_3d.mp4 -i output_2d.mp4 -filter_complex "[0:v][1:v]vstack=inputs=2" -pix_fmt yuv420p combined.mp4 -y
ffmpeg -i combined.mp4 -vf "fps=15,scale=iw/2:ih/2:flags=lanczos" output.gif -y
