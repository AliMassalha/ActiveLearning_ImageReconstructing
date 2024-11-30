import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        pool = self.pool(x)
        return pool, x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DenoisingUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # Encoder
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Bridge
        self.bridge_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bridge_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bridge_bn1 = nn.BatchNorm2d(1024)
        self.bridge_bn2 = nn.BatchNorm2d(1024)

        # Decoder
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        # Output
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        p1, s1 = self.down1(x)
        p2, s2 = self.down2(p1)
        p3, s3 = self.down3(p2)
        p4, s4 = self.down4(p3)

        # Bridge
        bridge = self.relu(self.bridge_bn1(self.bridge_conv1(p4)))
        bridge = self.relu(self.bridge_bn2(self.bridge_conv2(bridge)))

        # Decoder
        d1 = self.up1(bridge, s4)
        d2 = self.up2(d1, s3)
        d3 = self.up3(d2, s2)
        d4 = self.up4(d3, s1)

        # Output
        out = self.final_conv(d4)



        return out




class LightDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Single convolution instead of double
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        pool = self.pool(x)
        return pool, x

class LightUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Lighter than ConvTranspose2d
        self.conv = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        return x

class LightDenoisingUNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()


        base_channels = 32

        # Encoder with fewer channels
        self.down1 = LightDownBlock(in_channels, base_channels)
        self.down2 = LightDownBlock(base_channels, base_channels * 2)
        self.down3 = LightDownBlock(base_channels * 2, base_channels * 4)

        # Removed one down block to make the model lighter

        # Simplified Bridge
        self.bridge_conv = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.bridge_bn = nn.BatchNorm2d(base_channels * 8)

        # Decoder with fewer channels
        self.up1 = LightUpBlock(base_channels * 8, base_channels * 4)
        self.up2 = LightUpBlock(base_channels * 4, base_channels * 2)
        self.up3 = LightUpBlock(base_channels * 2, base_channels)

        # Output
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # Encoder
        p1, s1 = self.down1(x)
        p2, s2 = self.down2(p1)
        p3, s3 = self.down3(p2)

        # Bridge
        bridge = self.relu(self.bridge_bn(self.bridge_conv(p3)))

        # Decoder
        d1 = self.up1(bridge, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)

        # Output
        out = self.final_conv(d3)
        return out

    def extract_features(self, x, extract_type='flatten'):
        # Pass through encoder layers
        p1, _ = self.down1(x)
        p2, _ = self.down2(p1)
        p3, _ = self.down3(p2)
        if extract_type=='flatten':
          # Flatten the spatial dimensions of p3
          features = p3.view(p3.size(0), -1)  # Shape: [batch_size, 128 * 32 * 32]
        elif extract_type=='mean':
          # Calculate the mean along the spatial dimensions
          features = torch.mean(p3, dim=[2, 3])  # Shape: [batch_size, 128]

        return features