# FacePose_pytorch

The pytorch implement of the head pose estimation(yaw,roll,pitch) with SOTA performance in real time.Easy to deploy, easy to use, and high accuracy.


## Demo

    # install requirements
    pip install numpy opencv-python tensorboard tensorboardX
    pip install torch==1.4.0
    pip install torchvision==0.5.0
	
	
    # run the simple inference script
	Record a head rotation video MP4 and put it in the "video" folder
    python video.py --image_name ./video/your_video_name.mp4

## Training

There is no need to model train

## Introduction

1. Firstly, the Retinaface is used to advance the face frame, and then PFLD is used to identify the key points of the face. Finally, the key points are followed up to estimate the face pose. It is very easy to deploy and use, with high precision and fast speed.
2. We collected our own facial angle conversion data from hundreds of colleagues and fit a simple linear model through the rotation key points of hundreds of people's faces.Experiments show that the simple mathematical linear point model is more efficient and accurate.

## Performance

    # speed
    type	Nvidia-V100
    fps/s	90
	
	# Angle error(yaw,roll,pitch)
	-3°~+3°

## FAQ


**Q1. Why implement this while there are several FacePose projects already.**

A1: Because the existing open source project identification error is large。

**Q2: What exactly is the difference among this repository and the others?**

A2: For example, these two are the most popular FacePose-pytorch,

<https://github.com/>

<https://github.com/>

Here is the issues and why these are difficult to achieve good effect

The first one:

1. Dlib is not accurate for face key points recognition, and the error is large when the face is rotated or occluded.
2. it is very inaccurate to compare the recognition of key points with a conventional 3D model.

The second one:

1. 

2. 



**Q3: What should I do when I find a bug?**

A3: Check out the update log if it's been fixed, then pull the latest code to try again. If it doesn't help, create a new issue and describe it in detail.
