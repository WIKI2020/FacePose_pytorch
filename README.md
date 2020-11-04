# FacePose_pytorch

The pytorch implement of the head pose estimation(yaw,roll,pitch) with SOTA performance in real time.Easy to deploy, easy to use, and high accuracy.

## Update Log

[2020/09/14]congratulate! The algorithm has been applied to the following two products: online education for children, which is used to identify whether children listen carefully; on-site meeting or school classroom, to judge the quality of speech.  
We will release the ultra-high precision model in October or November. If you need, please add a github star and leave email, I will send it to you separately.  
![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/y1.jpg)
![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/y2.jpg)


## Demo

    # install requirements
    First install Anaconda3, python 3.7,and then:
    pip install numpy opencv-python 
    pip install torch==1.4.0
    pip install torchvision==0.5.0
	
	
    # run the simple inference script
    Take a video of face rotation with a computer camera,and put it into video file
    CUDA_VISIBLE_DEVICE=0 python video.py --video_name ./video/your_video_name.mp4
    (tips:You can modify video.py file to infer pictures)

## Training

There is no need to model train(Using the open source model is enough)

## Introduction

1. Firstly, the Retinaface is used to extract the face frame, and then PFLD is used to identify the key points of the face. Finally, the key points are followed up to estimate the face pose. It is very easy to deploy and use, with high precision and fast speed.
2. We collected our own facial angle conversion data from hundreds of people and fit a simple linear model through the rotation key points of hundreds of people's faces.Experiments show that the simple mathematical linear point model is more efficient and accurate.
3. our program is capable of real-time performance and is able to run from a simple webcam without any specialist hardware.

## Performance
	
    GPU type     | fps/s  | Angle error(yaw,roll,pitch)
    Nvidia-V100  | 90     | -3°~+3° 

## Example
   ![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/0.jpg)
   ![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/1.jpg)
   ![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/2.jpg)

## Other Project
The following is a list of some classic face pose detection items, you can compare the effect by yourself  
OpenFace:[https://github.com/TadasBaltrusaitis/OpenFace]  
Dlib:[https://github.com/KeeganRen/FaceReconstruction]  
3DDFA_V2:[https://github.com/cleardusk/3DDFA_V2]  

## TODO
- [ ] Training details
- [ ] estimate details

## FAQ


**Q1. Why implement this while there are several FacePose projects already.**

A1: Because the existing open source project identification error is big。

**Q2: What exactly is the difference among this repository and the others?**

A2: For example, Here are some of the common methods used by other open source projects:
1. Dlib:It is not accurate for face key points recognition, and the error is large when the face is rotated or roll.
2. Virtual 3D model:it is very inaccurate to compare the recognition of key points with a "2D to 3D Virtual fix model",because everyone has a different face.
3. 3DDFA:The effect is poor,I don't know why there are so many "github stars".
4. Through the convolution network statistics face key point bitmap, the angle is also very inaccurate.


**Q3: What should I do when I find a bug?**

A3: Check out the update log if it's been fixed, then pull the latest code to try again. If it doesn't help, create a new issue and describe it in detail.


## Contribution
Many thanks to ElonMusk,RobertDowney,huangwj and wangz for contributing and revising to the code and English documentation.

