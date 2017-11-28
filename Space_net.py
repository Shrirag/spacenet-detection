
# coding: utf-8

# In[70]:

import torch, lab_utils, random
from torchvision.datasets import CIFAR10 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn 
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import json, string
import torch.nn.functional as F
get_ipython().magic('matplotlib inline')


# In[ ]:

"""Information about images"""
def get_num_pixels(filepath):
    width, height = Image.open(open(filepath)).size
    return width, height
print get_num_pixels("/Users/rajveernehra/Desktop/Machine_Learning!/Computer_Vision_UVa_Fall_2017/Space_net_project/Spacenet/annotations/annotations/RGB-PanSharpen__2.18641139997_49.0494609.jpg")
print get_num_pixels("/Users/rajveernehra/Desktop/Machine_Learning!/Computer_Vision_UVa_Fall_2017/Space_net_project/Spacenet/annotations/annotations/RGB-PanSharpen__2.18641139997_49.0494609segcls.png")


# In[35]:

'''
resnet = models.VGG19(pretrained  = True)
modules = list(resnet.children())[:-1]
List = list(resnet.children())
resnet2 = nn.Sequential(*modules) 
List.append(nn.Linear(1000,2))
resnet2 = nn.Sequential(*List)'''



# In[132]:

model = models.vgg19(pretrained  = True)
modules = list(model.children())[:-1]
model1 = nn.Sequential(*modules)


# In[93]:

import torch.utils.data as data
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

trainData = Spacenet_Dataset("/Users/rajveernehra/Desktop/Machine_Learning!/Computer_Vision_UVa_Fall_2017/Space_net_project/Spacenet/annotations/trainval.txt",transform=imgTransform)
valData = Spacenet_Dataset("/Users/rajveernehra/Desktop/Machine_Learning!/Computer_Vision_UVa_Fall_2017/Space_net_project/Spacenet/annotations/test.txt",transform=imgTransform)
inp,out = trainData[200]

def get_sample():
    inp,out = trainData[random.randint(0,3673)] # change it to Valdata if necessary, dont forget to change value inside rand()
    inp = inp.view(-1,3,400,400)
    model = MyNet().cuda()
    sample_ip = torch.autograd.Variable(inp).cuda()
    sample_out = model(sample_ip)
    sample_out = sample_out.view(-1,400,400)
    sample_out = cvt2pil(sample_out.data.cpu())
    sample_inp = cvt2pil(inp.view(-1,400,400)) 
    plt.imshow(sample_inp)
    plt.show()
    plt.imshow(sample_out)
    plt.show()


# In[129]:

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        #Pretrained VGG without classifier 
        self.model = models.vgg19(pretrained  = True).features
         
        """We wanted to play with the first and last layers of VGG but realized that it won't work 8 
        and 11 band images"""
        """self.List = [self.model[i] for i in range(len(self.model))]
        self.List[0] = nn.Conv2d(8,64, kernel_size = 3, stride= 1, padding = 1)
        self.List[]
        self.model = nn.Sequential(*self.List)""" 
        
        # FCC Convolutional layers network.
    
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3) 
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.ConvTranspose2d(32, 32, 3) 
        self.conv5 = nn.Conv2d(32, 16, 3)
        self.conv6 = nn.ConvTranspose2d(16, 16, 3) 
        self.conv7 = nn.ConvTranspose2d(16,1,3)

    def forward(self, x):
        
        m = nn.Dropout2d(p=0.1)
        
        Norm_1 = nn.BatchNorm2d(16)
        Norm_2 = nn.BatchNorm2d(32)
        Norm_3 = nn.BatchNorm2d(32) 
        Norm_4 = nn.BatchNorm2d(32)
        Norm_5 = nn.BatchNorm2d(16)
        Norm_6 = nn.BatchNorm2d(16)
        
        out = self.model(x)
        
        out = F.relu(self.conv1(x))
        out = Norm_1(out)
        
        out = F.relu(self.conv2(out))
        out = Norm_2(out)
        out = F.max_pool2d(out, 2)
        out = m(out)
        
        out = F.relu(self.conv3(out))
        out = Norm_3(out)
        
        out = F.relu(self.conv4(out))
        out = Norm_4(out)
        out = F.max_pool2d(out, 2)
        out = m(out)
        
        out = F.relu(self.conv5(out))
        out = Norm_5(out)
             
        out = F.relu(self.conv6(out))
        out = Norm_6(out)
        
        out = self.conv7(out)
        
        return out


# In[130]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



