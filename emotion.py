import cv2
import torch
from torchvision import transforms
import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import time
import argparse

result = ["Surprise","Fear","Disgust","Happiness","Sadness","Anger","Neutral"]

class Res18Feature(nn.Module):
  def __init__(self, pretrained, num_classes = 7):
    super(Res18Feature, self).__init__()
    resnet  = models.resnet18(pretrained)
    self.features = nn.Sequential(*list(resnet.children())[:-1]) 
    fc_in_dim = list(resnet.children())[-1].in_features 
    self.fc = nn.Linear(fc_in_dim, num_classes) 
    self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    attention_weights = self.alpha(x)
    out = attention_weights * self.fc(x)
    return attention_weights, out

model_save_path = "./checkpoint/wiki2020.pth" #mode path

def main(args):
  preprocess_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
  res18 = Res18Feature(pretrained = False)
  checkpoint = torch.load(model_save_path)
  res18.load_state_dict(checkpoint['model_state_dict'])
  res18.cuda()
  res18.eval()

  for i in [0]:
    time1=time.time()
    
    image = cv2.imread(args.img)
    image = image[:, :, ::-1]
    image_tensor = preprocess_transform(image)
    tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False)
    tensor=tensor.cuda()
    time2=time.time()
    _, outputs = res18(tensor)
    _, predicts = torch.max(outputs, 1)
    print(result[int(predicts.cpu().data)])

def parse_args():
  parser = argparse.ArgumentParser(description='Testing')
  parser.add_argument('--img',default="./img/suripse.jpg",type=str)
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  main(args)
