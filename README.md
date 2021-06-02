# FacePose_pytorch

The pytorch implement of the head pose estimation(yaw,roll,pitch) and emotion detection with SOTA performance in real time.Easy to deploy, easy to use, and high accuracy.Solve all problems of face detection at one time.(极简，极快，高效是我们的宗旨)

## Update Log
[2020/12]We found a better algorithm for face key points(estimates 468 3D face landmarks in real-time even on CPU or mobile devices).  
#First install Anaconda3, python 3.7  
pip install mediapipe   
python newdectect.y  (run on cpu)  
You can replace the pfld algorithm(this GitHub) by yourself.  
![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/face.gif) 
 
[2020/11]congratulate! The algorithm has been applied to the following two products: online education for children, which is used to identify whether children listen carefully; on-site meeting or school classroom, to judge the quality of speech.  

The head angle, especially the accuracy of facial emotion recognition, has reached the world's top level. However, for some reasons, we only open source prediction code.  
 
We will release the ultra-high precision model in future(Including angles and emotion). If you need, please add a github star and leave email, I will send it to you separately.  
![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/y1.jpg)
![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/y2.jpg)


## Demo

    # install requirements
    First install Anaconda3, python 3.7,and then:
    pip install numpy opencv-python 
    pip install torch==1.4.0
    pip install torchvision==0.5.0
    Download the emoticon model (the angle model is already in the code):[https://pan.baidu.com/s/1oxznkRcP5w8lzYMAjj87-w],accesscode:WIKI
	
	
    # run the simple inference script(angel)
    Take a video of face rotation with a computer camera,and put it into video file
    CUDA_VISIBLE_DEVICE=0 python video.py --video_name ./video/your_video_name.mp4
    (tips:You can modify video.py file to infer pictures)

    # run the simple inference script(emotion)
    Download the emoticon model into checkpoint file,and(If you use your own photo, you need to cut out the face from the picture or use Retinaface to detection the face first. You can look at it in video.py)
    CUDA_VISIBLE_DEVICE=0 python emotion.py --img ./img/surprise.jpg
    
    At present, only one face is supported. You can try to modify the code to support the angle and expression recognition of multiple faces. It may be a bit complicated.

## Training

There is no need to model train(Using my model is enough)

## Introduction

1. Firstly, the Retinaface is used to extract the face frame, and then PFLD is used to identify the key points of the face. Finally, the key points are followed up to estimate the face pose. It is very easy to deploy and use, with high precision and fast speed.
2. We collected our own facial angle conversion data from hundreds of people and fit a simple linear model through the rotation key points of hundreds of people's faces.Experiments show that the simple mathematical linear point model is more efficient and accurate(You can also use GBDT or other algorithms to regression).
3. At the same time, referring to several papers published in the United States summit in 2020, we developed a highly accurate emotion recognition model. Results show that the methods with 95% on raf-db, 80% on affectnet, and 98% on ferplus.At present, we predict seven kinds of expressions: "surprise", "fear", "strange", "happiness", "Sadness", "anger" and "neutral".


## Performance
	
    GPU type     | fps/s  | Angle error(yaw,roll,pitch)| emotion
    Nvidia-V100  | 90     | -3°~+3°                    | 95% on average

## Example
   ![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/emo.jpg)
   ![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/1.jpg)
   ![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/0.jpg)
   ![image](https://github.com/WIKI2020/FacePose_pytorch/blob/master/img/2.jpg)

## Other Project
The following is a list of some classic face pose detection items, you can compare the effect by yourself  
OpenFace:[https://github.com/TadasBaltrusaitis/OpenFace]  
Dlib:[https://github.com/KeeganRen/FaceReconstruction]  
3DDFA_V2:[https://github.com/cleardusk/3DDFA_V2]  


## FAQ


**Q1. Why implement this while there are several FacePose projects already.**

A1: Because the existing open source project identification error is big.And this is a Key point detection and expression detection integration project.

**Q2: What exactly is the difference among this repository and the others?**

A2: For example, Here are some of the common methods used by other open source projects:
1. Dlib:It is not accurate for face key points recognition, and the error is large when the face is rotated or roll.
2. Virtual 3D model:it is very inaccurate to compare the recognition of key points with a "2D to 3D Virtual fix model",because everyone has a different face.
3. Through the convolution network statistics face key point bitmap, the angle is also very inaccurate.


**Q3: What should I do when I find a bug?**

A3: Check out the update log if it's been fixed, then pull the latest code to try again. If it doesn't help, create a new issue and describe it in detail.


## Contribution
Many thanks to ElonMusk,huangwj and wangz for contributing and revising to the code and English documentation.

