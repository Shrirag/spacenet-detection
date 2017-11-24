import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # Convolutional layers.
        self.conv1 = nn.Conv2d(3, 64, 8, stride = 2)
        self.conv2 = nn.Conv2d(64, 128, 6, stride = 2)
        self.conv3 = nn.Conv2d(128, 256, 7,stride = 3)
        self.conv4 = nn.Conv2d(256, 512, 3,stride = 2, padding =2)
        
        self.deconv1 = nn.ConvTranspose2d(512,256,3,stride=2,padding=1)
        self.deconv2 = nn.ConvTranspose2d(256,128,7,stride=3)
        self.deconv3 = nn.ConvTranspose2d(128,64,6,stride=2)
        self.deconv4 = nn.ConvTranspose2d(64,3,8,stride=2,padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        
        return out
