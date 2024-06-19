import pandas as pd
from PIL import Image, ImageFilter
from torch.utils import data
import numpy as np
import torch
import random

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('dataset/train.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'valid':
        df = pd.read_csv('dataset/valid.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    else:
        df = pd.read_csv('dataset/test.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index,task):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        if task == 'train':
            imgs = np.zeros((len(index),3,224,224))
            label = np.zeros((len(index)))

            for i in range(len(index)):
                path_img = self.root + "/" + self.img_name[index[i]]

                img = Image.open(path_img)
                if random.randint(0, 1):
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                # transform = [ImageFilter.BLUR,ImageFilter.CONTOUR,ImageFilter.FIND_EDGES,ImageFilter.EMBOSS,ImageFilter.SMOOTH]
                # rand = random.randint(0, 5)
                # if rand:
                #     img = img.filter(transform[rand-1])
                
                img = img.rotate(random.uniform(0, 360)) 

                img = np.array(img).reshape(3,224,224)/255.0
                imgs[i] = img.copy()
                label[i] = self.label[index[i]]

            imgs = torch.tensor(imgs).to(torch.float32)
            label = torch.tensor(label).to(torch.long)
            return imgs, label
        else:
            imgs = np.zeros((len(index),3,224,224))
            label = np.zeros((len(index)))
            for i in range(len(index)):
                path_img = self.root + "/" + self.img_name[index[i]]
                img = Image.open(path_img)
                img = np.array(img).reshape(3,224,224)/255.0
                imgs[i] = img.copy()
                label[i] = self.label[index[i]]
            imgs = torch.tensor(imgs).to(torch.float32)
            label = torch.tensor(label).to(torch.long)
            return imgs, label
