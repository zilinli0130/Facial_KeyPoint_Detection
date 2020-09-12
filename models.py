import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## It's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #Image size = [1, 224, 224]
        self.conv1 = nn.Conv2d(1, 32, 5)
        #Image size = [32, 220, 220]
        #Image size = [32, 110, 110] after 1st pooling
        self.conv2 = nn.Conv2d(32, 64, 4)
        #Image size = [64, 107, 107]
        #Image size = [64, 53, 53] after 2nd pooling
        self.conv3 = nn.Conv2d(64, 128, 3)
        #Image size = [128, 51, 51]
        #Image size = [128, 25, 25] after 3rd pooling
        self.conv4 = nn.Conv2d(128, 256, 2)
        #Image size = [256, 24, 24]
        #Image size = [256, 12, 12] after 4th pooling
       
        #1st full-connected layer
        self.fc1 = nn.Linear(256*12*12, 1000)
        
        #2st full-connected layer, create 2*68 output channels
        self.fc2 = nn.Linear(1000, 2*68)
        
        #Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
            
        #Dropout with p=0.4
        self.drop = nn.Dropout(p=0.4)

    #Define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop(x)

        #Prep for linear layer
        #This line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        #Two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        #Final output
        return x

#Instantiate and print your Net
net = Net()
print(net)