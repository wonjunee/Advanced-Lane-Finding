# Udacity Self-Driving Car Engineer Nanodegree
# Computer Vision Project - Advanced Lane Finding
## By Wonjun Lee

## Overview

This is the **Advanced Lane Finding** project from Udacity Self-Driving Car Engineer Nanodegree.

The purpose of the project is to detect the lanes from the video stream and shade the areas between the identified lanes.

Below is the list of techniques used to detect lanes.

- Camera Calibration: Transformation between 2D image points to 3D object points.
- Distortion Correction: Consistent representation of the geometrical shape of objects.
- Perpective Transform: Warping images to effectively view them from a different angle or direction.
- Edge Detection: Sobel Operator, Magnitude Gradient, Directional Gradient, and HLS Color Space with Color thresholding
- Sanity Check: Used Radius of Curvature and Coefficients of Polynomial fittings.

You can see the detail of the project through *Advanced-Lane-Finding-Submission.ipynb*.

*Advanced-Lane-Finding-Submission-Multiple-Vidoes.ipynb* is almost identical to the above one except it will generate the video of three different images.