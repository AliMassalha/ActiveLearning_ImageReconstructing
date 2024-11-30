import torch
import torch.nn as nn
import torch.nn.functional as F

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()

        # Feature Extraction Branch 1
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)

        # Feature Extraction Branch 2
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)

        # Fusion Branch
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Feature Enhancement Branch
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Feature Refinement Branch
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=1)

        # Channel Attention Branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature Extraction
        conv1 = self.relu(self.conv1(x))
        conv1 = self.relu(self.conv2(conv1))

        conv2 = self.relu(self.conv3(x))
        conv2 = self.relu(self.conv4(conv2))

        # Feature Fusion
        concat = torch.cat([conv1, conv2], dim=1)
        conv3 = self.relu(self.conv5(concat))

        # First Residual Connection
        add1 = x + conv3

        # Feature Enhancement
        conv4 = self.relu(self.conv6(add1))
        conv4 = self.conv7(conv4)
        add2 = self.relu(add1 + conv4)

        # Feature Refinement
        conv5 = self.relu(self.conv8(add2))
        conv5 = self.relu(self.conv9(conv5))
        conv5 = self.conv10(conv5)
        add3 = self.relu(add2 + conv5)

        # Channel Attention
        gap = self.gap(add3)
        conv6 = self.relu(self.conv11(gap))
        conv6 = self.sigmoid(self.conv12(conv6))

        # Final Feature Fusion
        mul = conv6 * add3
        out = x + mul

        return out

class RIDNet(nn.Module):
    def __init__(self):
        super(RIDNet, self).__init__()

        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Enhanced Attention Modules
        self.eam1 = EAM()
        self.eam2 = EAM()
        self.eam3 = EAM()
        self.eam4 = EAM()

        # Final reconstruction
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        conv1 = self.conv1(x)

        eam1 = self.eam1(conv1)
        eam2 = self.eam2(eam1)
        eam3 = self.eam3(eam2)
        eam4 = self.eam4(eam3)

        conv2 = self.conv2(eam4)
        out = conv2 + x

        return out



class LightEAM(nn.Module):
    def __init__(self):
        super(LightEAM, self).__init__()

        # Reduced number of channels from 64 to 32
        self.channels = 32

        # Feature Extraction Branch (simplified to one branch)
        self.conv1 = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=2, dilation=2)

        # Fusion Branch (removed second branch to reduce parameters)
        self.conv5 = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)

        # Feature Enhancement Branch (reduced to one conv layer)
        self.conv6 = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)

        # Channel Attention Branch (simplified)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels // 4, self.channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Feature Extraction
        conv1 = self.relu(self.conv1(x))
        conv1 = self.relu(self.conv2(conv1))

        # Feature Enhancement
        conv3 = self.relu(self.conv5(conv1))
        add1 = x + conv3

        # Simplified Enhancement
        conv4 = self.relu(self.conv6(add1))
        add2 = self.relu(add1 + conv4)

        # Channel Attention
        att = self.attention(self.gap(add2))
        out = x + (att * add2)

        return out

class LightRIDNet(nn.Module):
    def __init__(self):
        super(LightRIDNet, self).__init__()

        self.channels = 32  # Reduced from 64 to 32

        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=3, padding=1)

        # Reduced number of EAM modules from 4 to 2
        self.eam1 = LightEAM()
        self.eam2 = LightEAM()

        # Final reconstruction
        self.conv2 = nn.Conv2d(self.channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        conv1 = self.conv1(x)

        eam1 = self.eam1(conv1)
        eam2 = self.eam2(eam1)

        conv2 = self.conv2(eam2)
        out = conv2 + x

        return out

    def extract_features(self, x, extract_type='flatten'):
        # Pass through encoder layers
        encoding = self.forward(x)-x

        if extract_type=='flatten':
          # Flatten the spatial dimensions of encoding
          features = encoding.view(encoding.size(0), -1)  # Shape: [batch_size, 3 * 256 * 256]
        elif extract_type=='mean':
          # Calculate the mean along the spatial dimensions
          features = torch.mean(encoding, dim=[2, 3])  # Shape: [batch_size, 3]

        return features