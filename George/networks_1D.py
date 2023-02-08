import torch
import torch.nn as nn
import torch.nn.functional as F  


# Dense Block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels+in_channels, kernel_size, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        final = out_channels + 2 * in_channels
        self.conv_trans = nn.Conv1d(final, final//2, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = torch.cat([out, x], dim=1)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = torch.cat([out, x], dim=1)
        out = self.conv_trans(out)
        out = self.pool(out)
        return out
    
    
# Residual layer class and ResNets
class Res1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Res1DLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # projection shortcut
        self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # add the projection shortcut
        identity = self.projection(identity)
        out += identity
        out = F.leaky_relu(out)
        return out
    
    
class ResNet1D(nn.Module):
    """original paper: https://arxiv.org/abs/2105.07302"""
    def __init__(self, F=128, num_classes=10, p_dropout=None):
        super(ResNet1D, self).__init__()
        if F == 128:
            self.name = 'ResNet1D' # as implemented in the original paper
        elif F == 64:
            self.name = 'ResNet1D-M'
        elif F == 32:
            self.name = 'ResNet1D-S'
        else:
            self.name = 'ResNet1D-Custom'
        self.p_dropout = p_dropout
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=F, kernel_size=3, stride=3, padding=3, bias=False)
     
        self.layer1 = nn.Sequential(
            Res1DLayer(F, F, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=9, stride=9, padding=1),
            Res1DLayer(F, F, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=9, stride=9, padding=1)
        )
        
        self.layer2 = nn.Sequential(
            Res1DLayer(F, F*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=9, stride=9, padding=1),
            Res1DLayer(F*2, F*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=9, stride=9, padding=1)
        )
        
        self.layer3 = nn.Sequential(
            Res1DLayer(F*2, F*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            Res1DLayer(F*2, F*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            Res1DLayer(F*2, F*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            Res1DLayer(F*2, F*2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        )
        
        self.layer4 = nn.Sequential(
            Res1DLayer(F*2, F*4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            nn.Conv1d(F*4, F*4, kernel_size=1, stride=1, padding=0)
        )
        
        self.fc = nn.Linear(F*4, num_classes)
        if p_dropout:
            self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc(x)
        if self.p_dropout:
            x = self.dropout(x)
        return x


# SoundNet - 5 layer version
class SoundNet1D(nn.Module):
    def __init__(self, input_size, num_classes=8, p_dropout=None):
        super(SoundNet1D, self).__init__()
        self.name = 'SoundNet1D'
        self.p_dropout = p_dropout
        self.group1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=64, stride=2, padding=32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0)
        )
    
        self.group2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=0),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=2, padding=4),
            nn.Conv1d(in_channels=256, out_channels=1401, kernel_size=16, stride=12, padding=4)

        )
        
        self.fc = nn.Linear(2802, num_classes)
        if p_dropout:
            self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc(x)
        if self.p_dropout:
            x = self.dropout(x)
        return x


# Deeper SoundNet - 8 layer version
class SoundNet1D_L(nn.Module):
    def __init__(self, input_size, num_classes=8):
        super(SoundNet1DLarge, self).__init__()
        self.name = 'SoundNet1D_L'
        self.p_dropout = p_dropout
        self.group1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=8, stride=1, padding=0),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=8, stride=1, padding=0)
        )
    
        self.group2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4, stride=1, padding=0),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.conv81 = nn.Conv1d(in_channels=1024, out_channels=1000, kernel_size=8, stride=2, padding=0)
        self.conv82 = nn.Conv1d(in_channels=1024, out_channels=401, kernel_size=8, stride=2, padding=0)
        self.fc = nn.Linear(1401, num_classes)
        if self.p_dropout:
            self.dropout = nn.Dropout(p=p_dropout)

        
    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x1 = self.conv81(x)
        x2 = self.conv82(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc(x)
        if self.p_dropout:
            x = self.dropout(x)
        return x


    