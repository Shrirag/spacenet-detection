import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm
from torch.autograd import Variable
import numpy as np

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

train_loss=[]
test_loss=[]

def train_rnn_model(model, criterion, optimizer, trainLoader, valLoader, n_epochs = 10, use_gpu = False):      
    
    if use_gpu:
        print('using GPU!')
        model = model.cuda()
        criterion = criterion.cuda()
        
    # Training loop.
    for epoch in range(0, n_epochs):
        cum_loss = 0.0
        accuracy = 0
        
        # Make a pass over the training data.
        t = tqdm(trainData, desc = 'Training epoch %d' % epoch)
        model.train()  # This is important to call before training!
        for (ip_image, out_image) in enumerate(t): 

            input_image = torch.autograd.Variable(ip_image)
            target = torch.autograd.Variable(out_image)

            if use_gpu:
                input_image = input_image.cuda()
                target = target.cuda()
                
            y_ = model(input_image)

            loss = criterion(y_, target)
            cum_loss += loss.data[0]

            optimizer.zero_grad()

            loss.backward()
                    # Weight and bias updates.
            optimizer.step()                

            # logging information.
            t.set_postfix(loss = cum_loss / (1 + i)) 

        train_loss.append(cum_loss/(i+1))


        # Make a pass over the validation data.
        cum_loss = 0.0
        accuracy = 0
        
        t = tqdm(valData, desc = 'Validation epoch %d' % epoch)
        model.eval()  # This is important to call before evaluating!
        for (ip_image,out_image) in enumerate(t): 
        
            input_image = torch.autograd.Variable(ip_image)
            target = torch.autograd.Variable(out_image)

            if use_gpu:
                input_image = input_image.cuda()
                target = target.cuda()
                
            y_ = model(input_image)

            loss = criterion(y_, target)
            cum_loss += loss.data[0]
            
        test_loss.append(cum_loss/(i+1))
    return model

model = MyNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

# Train the previously defined model.
trained_model = train_rnn_model(model, criterion, optimizer, trainLoader, valLoader, n_epochs = 10, use_gpu = True)
