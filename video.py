import argparse
import time
import warnings
import numpy as np
import torch
import math
import torchvision
from torchvision import transforms
import cv2
from dectect import AntiSpoofPredict

from pfld.pfld import PFLDInference, AuxiliaryNet
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num(point_dict,name,axis):
    num = point_dict.get(f'{name}')[axis]
    num = float(num)
    return num

def cross_point(line1, line2):  
    x1 = line1[0]  
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1) 
    b1 = y1 * 1.0 - x1 * k1 * 1.0  
    if (x4 - x3) == 0: 
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def point_line(point,line):
    x1 = line[0]  
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1) 
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def point_point(point_1,point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return distance

def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    plfd_backbone = PFLDInference().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    videoCapture = cv2.VideoCapture(args.video_name)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps:",fps,"size:",size)
    videoWriter = cv2.VideoWriter("./video/result.avi",cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)

    success,img = videoCapture.read()
    cv2.imwrite("1.jpg",img)
    while success:
        height, width = img.shape[:2]
        model_test = AntiSpoofPredict(args.device_id)
        image_bbox = model_test.get_bbox(img)
        x1 = image_bbox[0]
        y1 = image_bbox[1]
        x2 = image_bbox[0] + image_bbox[2]
        y2 = image_bbox[1] + image_bbox[3]
        w = x2 - x1
        h = y2 - y1

        size = int(max([w, h]))
        cx = x1 + w/2
        cy = y1 + h/2
        x1 = cx - size/2
        x2 = x1 + size
        y1 = cy - size/2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped = img[int(y1):int(y2), int(x1):int(x2)]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        cropped = cv2.resize(cropped, (112, 112))

        input = cv2.resize(cropped, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = transform(input).unsqueeze(0).to(device)
        _, landmarks = plfd_backbone(input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
        point_dict = {}
        i = 0
        for (x,y) in pre_landmark.astype(np.float32):
            point_dict[f'{i}'] = [x,y]
            i += 1

        #yaw
        point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
        point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
        point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
        crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
        yaw_mean = point_point(point1, point31) / 2
        yaw_right = point_point(point1, crossover51)
        yaw = (yaw_mean - yaw_right) / yaw_mean
        yaw = int(yaw * 71.58 + 0.7037)

        #pitch
        pitch_dis = point_point(point51, crossover51)
        if point51[1] < crossover51[1]:
            pitch_dis = -pitch_dis
        pitch = int(1.497 * pitch_dis + 18.97)

        #roll
        roll_tan = abs(get_num(point_dict,60,1) - get_num(point_dict,72,1)) / abs(get_num(point_dict,60,0) - get_num(point_dict,72,0))
        roll = math.atan(roll_tan)
        roll = math.degrees(roll)
        if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
            roll = -roll
        roll = int(roll)
        cv2.putText(img,f"Head_Yaw(degree): {yaw}",(30,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        cv2.putText(img,f"Head_Pitch(degree): {pitch}",(30,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        cv2.putText(img,f"Head_Roll(degree): {roll}",(30,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        videoWriter.write(img)
        success, img = videoCapture.read()

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument(
        '--model_path',
        default="./checkpoint/snapshot/checkpoint.pth.tar",
        type=str)
    parser.add_argument(
        '--video_name',
        type=str,
        default="./video/1.mp4")
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
