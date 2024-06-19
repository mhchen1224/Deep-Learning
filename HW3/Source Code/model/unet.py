# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1
  
class Encoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Encoder, self).__init__()
    self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
  def forward(self, x):
    x = self.block1(x)
    return x

class UNet(nn.Module):
    def __init__(self,):
        super(UNet, self).__init__()

        self.block1 = Encoder(3,64)
        self.block2 = Encoder(64,128)
        self.block3 = Encoder(128,256)
        self.block4 = Encoder(256,512)
        
        self.block5 = Decoder(512,256)
        self.block6 = Decoder(256,128)
        self.block7 = Decoder(128,64)
        self.output  = nn.Conv2d(64, 1, kernel_size=1)

        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2) 

    def forward(self,x):
        x1 = self.block1(x)
        x1_maxpool = self.MaxPool2d(x1)
        x2 = self.block2(x1_maxpool)
        x2_maxpool = self.MaxPool2d(x2)
        x3 = self.block3(x2_maxpool)
        x3_maxpool = self.MaxPool2d(x3)
        x4 = self.block4(x3_maxpool)
        x5 = self.block5(x4,x3)
        x6 = self.block6(x5,x2)
        x7 = self.block7(x6,x1)
        output = self.output(x7)
        output = F.sigmoid(output)
        return output

if __name__ == '__main__':
    model = UNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
