import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.nn import init


class AudioClassifier (nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=8)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        
        self.dropout = nn.Dropout(p=0.3)
 
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)
        x = self.dropout(x)

        # Final output
        return x



class Res2DBlock(nn.Module):
    expansion = 1 #we don't use the block.expansion here

    def __init__(self, inplanes, planes, stride=1,padding = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride=stride,
                     padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride=1,
                     padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
    
class nnet1(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, input_size), stride=1, padding=2, bias=True),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=1, padding=2, bias=True),
            nn.ReLU()
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size=(26, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=(26, 1))
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6656, 300)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(300, 150 )
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(150, num_classes)
        
        
     #in this method we tell pytorch how to pass data from layer to another layer   
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        #x = self.dropout2(x)
        x = self.conv3(x)
        y = self.maxpool(x)
        z = self.avgpool(x)
        x = torch.cat((y, z), dim=1)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_size, F=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels = 1, out_channels = F, kernel_size = (4, input_size), bias = False),
                        nn.BatchNorm2d(F),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels = F, out_channels = F, kernel_size = (4, 1), bias = False),
                        nn.BatchNorm2d(F),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels = F, out_channels = F, kernel_size = (4, 1),  padding=(3,0), bias = False),
                        nn.BatchNorm2d(F))
        self.relu = nn.ReLU()
        

        
    def forward(self, x):
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = self.conv3(out)
        out = self.relu(out + residual)
        return out       


class nnet2(nn.Module):
    def __init__(self, input_size, num_classes=8, F=256):
        super(nnet2, self).__init__()
        self.block = ResidualBlock(input_size, F=F)
        self.Max = nn.MaxPool2d(kernel_size = (125, 1))
        self.Avg = nn.AvgPool2d(kernel_size = (125, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(F*2, F)
        self.linear2 = nn.Linear(F,  F//2)
        self.linear3 = nn.Linear(F//2, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = self.block(x)
        #print(x.size())
        y = self.Max(x)
        z = self.Avg(x)
        #print(y.size(), z.size())
        x = torch.cat((y, z), dim=1)
        #print(x.size())
        x = self.flatten(x)
        #print(x.size())
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.dropout(x)
        return x
    

#ResNet34

class Res2DBlock(nn.Module):
    expansion = 1 #we don't use the block.expansion here

    def __init__(self, inplanes, planes, stride=1,padding = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride=stride,
                     padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride=1,
                     padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
    
class ResNet34(nn.Module):

    def __init__(self, FN=16, num_classes=8, p_dropout=None):
        super().__init__()
        
        self.FN = FN
        if FN == 128:
            self.name = 'ResNet34-XL' 
        elif FN == 64:
            self.name = 'ResNet34-L'
        elif FN == 32:
            self.name = 'ResNet34-M'
        elif FN == 16:
            self.name = 'ResNet34-S'
        else:
            self.name ='ResNet34'
        layers = [3, 4, 6, 3]
        self.c1 = nn.Conv2d(1, FN, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(FN)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(FN, FN, layers[0])
        self.layer2 = self._make_layer(FN, FN*2, layers[1], stride=2)
        self.layer3 = self._make_layer(FN*2, FN*2, layers[2], stride=2)
        self.layer4 = self._make_layer(FN*2, FN*4, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Linear(FN * 196 , num_classes)
        self.p_dropout = p_dropout
        if p_dropout:
            self.dropout = nn.Dropout(p=p_dropout)


    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(Res2DBlock(inplanes, planes, stride))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(Res2DBlock(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.c1(x)           
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         

        x = self.layer1(x)          
        x = self.layer2(x)          
        x = self.layer3(x)          
        x = self.layer4(x)          

        x = self.avgpool(x)         
        x = torch.flatten(x, 1)     
        x = self.fc(x)
        if self.p_dropout:
            x = self.dropout(x)

        return x
    
    
