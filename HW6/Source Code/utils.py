import torch.nn as nn
import os
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision
from torch.nn import functional as F
from diffusers import UNet2DModel

def get_data():
    with open('train.json', "r") as f:
        data_json = json.load(f)

    with open("objects.json", "r") as f:
        object_json = json.load(f)
    
    img_path_list, img_label_list = [], []

    for img_path, img_label in data_json.items():
        img_path_list.append(os.path.join("iclevr", img_path))
        label = np.zeros(len(object_json),dtype=np.float32)
        for l in img_label:
            label[object_json[l]] = 1
        img_label_list.append(label)

    return img_path_list, img_label_list

class CLEVRLoader(torch.utils.data.Dataset):
    def __init__(self, transform_list):
        self.transform = transforms.Compose(transform_list)
        self.img_path_list, self.label_list = get_data()

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.img_path_list[index]).convert("RGB"))
        label = self.label_list[index]

        return img, label
    
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__()
        
        # The embedding layer will map the class label to a vector of size class_emb_size
        #self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.class_emb = nn.Linear(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=64,           # the target image resolution
            in_channels=3 + class_emb_size, # Additional input channels for class cond.
            out_channels=3,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            ),
            up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            )

        )

  # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
    # Shape of x:
        bs, ch, w, h = x.shape

        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels) # Map to embedding dimension
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)

        net_input = torch.cat((x, class_cond), 1)
        return self.model(net_input, t).sample


if __name__ == '__main__':
    transformsList = [transforms.Resize((64, 64)), 
                      transforms.ToTensor(), 
                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                      ]
    
    dataset = CLEVRLoader(transformsList)
    img, label = dataset.__getitem__(3)
    print(img)
    print(label)
