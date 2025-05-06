import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

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

class DownSample_Block(nn.Module):
    def __init__(self, window_size, stride):
        super(DownSample_Block, self).__init__()

        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        out = self.pool(x)
        return out

class UpSample_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample_Block, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x, output_size):
        out = self.up(x, output_size=output_size)
        return out

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_channels=3, out_channels=64)
        self.conv2 = DoubleConv(in_channels=64, out_channels=128)
        self.conv3 = DoubleConv(in_channels=128, out_channels=256)
        self.conv4 = DoubleConv(in_channels=256, out_channels=512)
        self.conv5 = DoubleConv(in_channels=512, out_channels=1024)

        self.conv6 = DoubleConv(in_channels=1024, out_channels=512)
        self.conv7 = DoubleConv(in_channels=512, out_channels=256)
        self.conv8 = DoubleConv(in_channels=256, out_channels=128)
        self.conv9 = DoubleConv(in_channels=128, out_channels=64)
        
        self.down = DownSample_Block(window_size=2, stride=2)
        
        self.up1 = UpSample_Block(in_channels=1024, out_channels=512)
        self.up2 = UpSample_Block(in_channels=512, out_channels=256)
        self.up3 = UpSample_Block(in_channels=256, out_channels=128)
        self.up4 = UpSample_Block(in_channels=128, out_channels=64)
        
        self.last = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)


    def forward(self, x):
        c1 = self.conv1(x)                        			# in_channels=3, out_channels=64 
        down1 = self.down(c1)                       		# in_channels=64, out_channels=64
            
        c2 = self.conv2(down1)                      		# in_channels=64, out_channels=128
        down2 = self.down(c2)                       		# in_channels=128, out_channels=128

        c3 = self.conv3(down2)                      		# in_channels=128, out_channels=256
        down3 = self.down(c3)                       		# in_channels=256, out_channels=256

        c4 = self.conv4(down3)                    			# in_channels=256, out_channels=512
        down4 = self.down(c4)                     			# in_channels=512, out_channels=512
        
        bridge = self.conv5(down4)                   		# in_channels=512, out_channels=1024

        u1 = self.up1(bridge, output_size=c4.size())        # in_channels=1024, out_channels=512 & image dimesions get doubled
        u1_concat = torch.cat([u1, c4], dim=1)              
        c6 = self.conv6(u1_concat)                  

        u2 = self.up2(c6, output_size=c3.size())            # in_channels=512, out_channels=256 & image dimesions get doubled
        u2_concat = torch.cat([u2, c3], dim=1)              
        c7 = self.conv7(u2_concat)                  

        u3 = self.up3(c7, output_size=c2.size())            # in_channels=256, out_channels=128 & image dimesions get doubled
        u3_concat = torch.cat([u3, c2], dim=1)              
        c8 = self.conv8(u3_concat)                  

        u4 = self.up4(c8, output_size=c1.size())            # in_channels=128, out_channels=64 & image dimesions get doubled
        u4_concat = torch.cat([u4, c1], dim=1)              
        c9 = self.conv9(u4_concat)                  

        out = self.last(c9)

        return out

if __name__ == "__main__":

	model = UNet(n_channels=3, n_classes=1)

	summary(model, (3, 480, 640))
	#summary(model, (3, 572, 572))
