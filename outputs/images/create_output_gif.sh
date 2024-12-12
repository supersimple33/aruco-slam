#!/bin/bash

ffmpeg -i outputs/images/output_3d.mp4 -i outputs/images/output_2d.mp4 -filter_complex "[0:v][1:v]vstack=inputs=2" -pix_fmt yuv420p outputs/images/combined.mp4 -y
ffmpeg -i outputs/images/combined.mp4 -vf "fps=15,scale=iw/2:ih/2:flags=lanczos" outputs/images/output.gif -y
