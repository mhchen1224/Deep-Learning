# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Decoder, self).__init__()
    self.up_conv_relu = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )
  def forward(self, x):
    x = self.up_conv_relu(x)
    return x
  
class Bottleneck_WDiffSize(nn.Module):
    def __init__(self,
                 cov1_inC, conv1_outC, conv1_ker, conv1_stride, conv1_padding,
                 cov2_inC, conv2_outC, conv2_ker, conv2_stride, conv2_padding,
                 downSample_inC, downSample_outC, downSample_ker, downSample_stride, downSample_padding):
        super(Bottleneck_WDiffSize, self).__init__()
        self.path1  = nn.Sequential(
            nn.Conv2d(in_channels=cov1_inC, out_channels=conv1_outC, kernel_size=conv1_ker, stride=conv1_stride,padding=conv1_padding),
            nn.BatchNorm2d(conv1_outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=cov2_inC, out_channels=conv2_outC, kernel_size=conv2_ker, stride=conv2_stride,padding=conv2_padding),
            nn.BatchNorm2d(conv2_outC)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=downSample_inC, out_channels=downSample_outC, kernel_size=downSample_ker, stride=downSample_stride, padding=downSample_padding),
            nn.BatchNorm2d(downSample_outC)
        )

    def forward(self,input):
        out1 = self.path1(input)
        out2 = self.path2(input)
        output = F.relu(out1+out2,inplace=True)
        return output
  

class Bottleneck(nn.Module):
    def __init__(self,channels):
        super(Bottleneck, self).__init__()
        self.layer  = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(channels),
        )

    def forward(self,input):
        out1 = self.layer(input)
        output = F.relu(input+out1,inplace=True)
        return output
    

class ResNet34_UNet(nn.Module):
    def __init__(self,):
        super(ResNet34_UNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block2 = nn.Sequential(
            Bottleneck(channels=64),
            Bottleneck(channels=64),
            Bottleneck(channels=64)
        )
        self.block3 = nn.Sequential(
            Bottleneck_WDiffSize(
                cov1_inC=64, conv1_outC=128, conv1_ker=(3,3), conv1_stride=(2,2), conv1_padding=(1,1),
                cov2_inC=128, conv2_outC=128, conv2_ker=(3,3), conv2_stride=(1,1), conv2_padding=(1,1),
                downSample_inC=64, downSample_outC=128, downSample_ker=(1,1), downSample_stride=(2,2), downSample_padding=(0,0)
            ),
            Bottleneck(channels=128),
            Bottleneck(channels=128),
            Bottleneck(channels=128),
        )
        self.block4 = nn.Sequential(
            Bottleneck_WDiffSize(
                cov1_inC=128, conv1_outC=256, conv1_ker=(3,3), conv1_stride=(2,2), conv1_padding=(1,1),
                cov2_inC=256, conv2_outC=256, conv2_ker=(3,3), conv2_stride=(1,1), conv2_padding=(1,1),
                downSample_inC=128, downSample_outC=256, downSample_ker=(1,1), downSample_stride=(2,2), downSample_padding=(0,0)
            ),
            Bottleneck(channels=256),
            Bottleneck(channels=256),
            Bottleneck(channels=256),
            Bottleneck(channels=256),
            Bottleneck(channels=256),
        )       
        self.block5 = nn.Sequential(
            Bottleneck_WDiffSize(
                cov1_inC=256, conv1_outC=512, conv1_ker=(3,3), conv1_stride=(1,1), conv1_padding=(1,1),
                cov2_inC=512, conv2_outC=512, conv2_ker=(3,3), conv2_stride=(2,2), conv2_padding=(1,1),
                downSample_inC=256, downSample_outC=512, downSample_ker=(1,1), downSample_stride=(2,2), downSample_padding=(0,0)
            ),
            Bottleneck(channels=512),
            Bottleneck(channels=512),
        ) #output [512,8,8]
        self.block6 = nn.Sequential(
            Bottleneck_WDiffSize(
                cov1_inC=512, conv1_outC=256, conv1_ker=(1,1), conv1_stride=(1,1), conv1_padding=(0,0),
                cov2_inC=256, conv2_outC=256, conv2_ker=(1,1), conv2_stride=(1,1), conv2_padding=(0,0),
                downSample_inC=512, downSample_outC=256, downSample_ker=(1,1), downSample_stride=(1,1), downSample_padding=(0,0)
            ),
            Bottleneck(channels=256),
            Bottleneck(channels=256),
        )

        self.block7 = Decoder(256+512,32)
        self.block8 = Decoder(32+256,32)
        self.block9 = Decoder(32+128,32)
        self.block10 = Decoder(32+64,32)
        self.block11 = Decoder(32,32)
        self.output  = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), stride=(1,1),padding=(1,1))

    def forward(self,x):
        x1 = self.block1(x)
        x2 = self.block2(x1)  #get last
        x3 = self.block3(x2)  #get last2
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(torch.cat((x6, x5), dim=1))
        x8 = self.block8(torch.cat((x7, x4), dim=1))
        x9 = self.block9(torch.cat((x8, x3), dim=1))
        x10 = self.block10(torch.cat((x9, x2), dim=1))
        x11 = self.block11(x10)
        output = self.output(x11)
        output = F.sigmoid(output)
        return output

if __name__ == '__main__':
    model = ResNet34_UNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
