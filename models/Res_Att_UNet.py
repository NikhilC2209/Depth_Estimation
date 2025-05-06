import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv1 = nn.Sequential (
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),

                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    )

    def forward(self, x):
        out = self.conv1(x)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UpSample_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample_Block, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x, output_size):
        out = self.up(x, output_size=output_size)
        return out

class ResNet34_UNet_Attention(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, weights=ResNet34_Weights.DEFAULT):
        super(ResNet34_UNet_Attention, self).__init__()

        base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        self.input = ConvBlock(in_channels=3, out_channels=64)
        
        self.input_layer = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
        )
        self.maxpool = base_model.maxpool
        self.encoder1 = base_model.layer1  # 64
        self.encoder2 = base_model.layer2  # 128
        self.encoder3 = base_model.layer3  # 256
        self.encoder4 = base_model.layer4  # 512

        # Attention blocks
        #self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        #self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        #self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        #self.att1 = AttentionBlock(F_g=64,  F_l=64,  F_int=32)

        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=64)
        self.att1 = AttentionBlock(F_g=64,  F_l=64,  F_int=32)

        # Convolutional layers
        self.conv1 = ConvBlock(in_channels=512, out_channels=256)
        self.conv2 = ConvBlock(in_channels=256, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=64)
        self.conv4 = ConvBlock(in_channels=128, out_channels=64)
        
        # Decoder upsampling
        self.up4 = self.upsample(512, 256)
        self.up3 = self.upsample(256, 128)
        self.up2 = self.upsample(128, 64)
        self.up1 = self.upsample(64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)

    def upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):

        x0 = self.input(x)
        
        #x1 = self.input_layer(x)    # [B, 64, H/2, W/2]
        
        x1 = self.maxpool(x0)       # [B, 64, H/2, W/2]
        x2 = self.encoder1(x1)      # [B, 64, H/4, W/4]
        x3 = self.encoder2(x2)      # [B, 128, H/8, W/8]
        x4 = self.encoder3(x3)      # [B, 256, H/16, W/16]
        x5 = self.encoder4(x4)      # [B, 512, H/32, W/32]

        # Decoder
        d4 = self.up4(x5)
        x4 = self.att4(g=d4, x=x4)
        d4_concat = torch.cat((d4, x4), dim=1)
        d4 = self.conv1(d4_concat)

        
        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=x3)
        d3_concat = torch.cat((d3, x3), dim=1)
        d3 = self.conv2(d3_concat)
        
        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=x2)
        d2_concat = torch.cat((d2, x2), dim=1)
        d2 = self.conv3(d2_concat)
        
        d1 = self.up1(d2)
        x0 = self.att1(g=d1, x=x0)
        d1_concat = torch.cat((d1, x0), dim=1)
        d1 = self.conv4(d1_concat)
        
        out = self.conv_last(d1)

        return out

if __name__ == "__main__":
	
	model = ResNet34_UNet_Attention(n_classes=1)
	model = model.cuda()

	x = torch.randn(2, 3, 480, 640).cuda()
	out = model(x)

	print(out.shape)  # Should be [2, 1, 256, 256]
