# Stereo Camera System - 3D Reconstruction  

## Overview  
This project is a stereo vision system that captures images from two USB cameras, processes them, and generates a 3D point cloud. It is designed to be a portable, home-made software solution for 3D geometry reconstruction.  

## Features  
- **Camera Calibration** – Compute intrinsic and extrinsic parameters.  
- **Stereo Rectification** – Align images to simplify depth estimation.  
- **Feature Detection & Matching** – Identify and match key points between images.  
- **Stereo Geometry Estimation** – Compute the essential matrix and recover camera poses.  
- **Triangulation & 3D Reconstruction** – Generate a sparse 3D point cloud.  
- **Point Cloud Post-Processing** – Filter and refine the point cloud for improved accuracy.  

## Technologies Used  
- **Programming Language:** Python
- **Libraries:** OpenCV, NumPy, Open3D (for visualization)    

## Installation  

