import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data

cvt2pil = transforms.ToPILImage()

class Spacenet_Dataset(data.Dataset):
    def __init__(self, annotationsFile,train = True, transform=None, target_transform = None):
        lines = open(annotationsFile,'r').readlines()
        self.transform=transform
        self.target_transform=target_transform
        self.data=[]
        for line in lines:
            ip,out = line.split(' ')
            out = out[:-5]+'segobj.png'
            item = [ip,out]
            self.data.append(item)
        inp_img_paths = self.data[:][0]
        out_img_paths = self.data[:][1]
              
    def getImage(self,path):
        pil_image = Image.open(path)
        pil_image = pil_image.convert('RGB')
        return pil_image

    def __getitem__(self, index):
        cur_data = self.data[index]
        ip_path= cur_data[0]
        out_path = cur_data[1]
        in_image = self.getImage(ip_path)
        out_image = self.getImage(out_path)
        
        if self.transform is not None:
            in_image = self.transform(in_image)
            out_image = self.transform(out_image)
        return in_image, out_image

imgTransform = transforms.Compose([transforms.Scale((400, 400)),
                                   transforms.ToTensor()])

trainData = Spacenet_Dataset("/home/dipshil/Spacenet_things/annotations/trainval.txt",transform=imgTransform)
inp,out = trainData[200]
