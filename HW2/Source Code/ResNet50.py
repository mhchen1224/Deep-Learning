import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class Bottleneck_WDiffSize(nn.Module):
    def __init__(self,
                 cov1_inC, conv1_outC, conv1_ker, conv1_stride, conv1_padding,
                 cov2_inC, conv2_outC, conv2_ker, conv2_stride, conv2_padding,
                 cov3_inC, conv3_outC, conv3_ker, conv3_stride, conv3_padding,
                 downSample_inC, downSample_outC, downSample_ker, downSample_stride, downSample_padding):
        super(Bottleneck_WDiffSize, self).__init__()
        self.path1  = nn.Sequential(
            nn.Conv2d(in_channels=cov1_inC, out_channels=conv1_outC, kernel_size=conv1_ker, stride=conv1_stride,padding=conv1_padding),
            nn.BatchNorm2d(conv1_outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=cov2_inC, out_channels=conv2_outC, kernel_size=conv2_ker, stride=conv2_stride,padding=conv2_padding),
            nn.BatchNorm2d(conv2_outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=cov3_inC, out_channels=conv3_outC, kernel_size=conv3_ker, stride=conv3_stride,padding=conv3_padding),
            nn.BatchNorm2d(conv3_outC),
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
        midC = channels//4
        self.layer  = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=midC, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(midC),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=midC, out_channels=midC, kernel_size=(3,3), padding=(1,1), stride=(1,1)),
            nn.BatchNorm2d(midC),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=midC, out_channels=channels, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(channels),
        )

    def forward(self,input):
        out1 = self.layer(input)
        output = F.relu(input+out1,inplace=True)
        return output
        

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            Bottleneck_WDiffSize(
                cov1_inC=64, conv1_outC=64, conv1_ker=(1,1), conv1_stride=(1,1), conv1_padding=(0,0),
                cov2_inC=64, conv2_outC=64, conv2_ker=(3,3), conv2_stride=(1,1), conv2_padding=(1,1),
                cov3_inC=64, conv3_outC=256, conv3_ker=(1,1), conv3_stride=(1,1), conv3_padding=(0,0),
                downSample_inC=64, downSample_outC=256, downSample_ker=(1,1), downSample_stride=(1,1), downSample_padding=(0,0)
            ),
            Bottleneck(channels=256),
            Bottleneck(channels=256)
        )
        self.block3 = nn.Sequential(
            Bottleneck_WDiffSize(
                cov1_inC=256, conv1_outC=128, conv1_ker=(1,1), conv1_stride=(1,1), conv1_padding=(0,0),
                cov2_inC=128, conv2_outC=128, conv2_ker=(3,3), conv2_stride=(2,2), conv2_padding=(1,1),
                cov3_inC=128, conv3_outC=512, conv3_ker=(1,1), conv3_stride=(1,1), conv3_padding=(0,0),
                downSample_inC=256, downSample_outC=512, downSample_ker=(1,1), downSample_stride=(2,2), downSample_padding=(0,0)
            ),
            Bottleneck(channels=512),
            Bottleneck(channels=512),
            Bottleneck(channels=512)
        )
        self.block4 = nn.Sequential(
            Bottleneck_WDiffSize(
                cov1_inC=512, conv1_outC=256, conv1_ker=(1,1), conv1_stride=(1,1), conv1_padding=(0,0),
                cov2_inC=256, conv2_outC=256, conv2_ker=(3,3), conv2_stride=(2,2), conv2_padding=(1,1),
                cov3_inC=256, conv3_outC=1024, conv3_ker=(1,1), conv3_stride=(1,1), conv3_padding=(0,0),
                downSample_inC=512, downSample_outC=1024, downSample_ker=(1,1), downSample_stride=(2,2), downSample_padding=(0,0)
            ),
            Bottleneck(channels=1024),
            Bottleneck(channels=1024),
            Bottleneck(channels=1024),
            Bottleneck(channels=1024),
            Bottleneck(channels=1024)
        )
        self.block5 = nn.Sequential(
            Bottleneck_WDiffSize(
                cov1_inC=1024, conv1_outC=512, conv1_ker=(1,1), conv1_stride=(1,1), conv1_padding=(0,0),
                cov2_inC=512, conv2_outC=512, conv2_ker=(3,3), conv2_stride=(2,2), conv2_padding=(1,1),
                cov3_inC=512, conv3_outC=2048, conv3_ker=(1,1), conv3_stride=(1,1), conv3_padding=(0,0),
                downSample_inC=1024, downSample_outC=2048, downSample_ker=(1,1), downSample_stride=(2,2), downSample_padding=(0,0)
            ),
            Bottleneck(channels=2048),
            Bottleneck(channels=2048)
        )

        self.avgpool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.block6 = nn.Sequential(
            nn.Dropout(0.6, inplace=False),
            nn.Linear(2048,100)
        )

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool2d(x)
        x = x.reshape(x.size(0), -1)
        x = self.block6(x)
        return x
        