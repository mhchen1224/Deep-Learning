import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import discriminator, generator, CLEVRLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torchvision
import json

class CGAN(object):
    def __init__(self):
        # parameters
        self.epoch = 50
        self.batch_size = 256
        self.input_size = 64
        self.z_dim = 64
        self.class_num = 24
        lr = 3e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transformsList = [transforms.Resize((64, 64)), 
                      transforms.ToTensor(), 
                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                      ]

        # load dataset
        dataset = CLEVRLoader(transformsList)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=3, input_size=self.input_size, class_num=self.class_num).to(self.device)
        self.D = discriminator(input_dim=3, output_dim=1, input_size=self.input_size, class_num=self.class_num).to(self.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.BCE_loss = nn.BCELoss()
        self.Gscheduler = optim.lr_scheduler.MultiStepLR(self.G_optimizer, [80,120], gamma=0.5, last_epoch=-1)
    
    def save_models(self,path):
        G_path = path + 'G.pth'
        D_path = path + 'D.pth'
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)


    def load_model(self,path):
        self.G.load_state_dict(torch.load(f'{path}/G.pth'))
        self.D.load_state_dict(torch.load(f'{path}/D.pth'))

    def train(self,last_epoch):
        if last_epoch > 0 : #load model
            path = f'/home/mhchen/Code/DLP/HW6/Results/Epoch_{last_epoch}'
            self.load_model(path)

        for epoch in tqdm(range(last_epoch +1 , last_epoch + 1 + self.epoch)):
            epoch_GLoss = 0
            epoch_DLoss = 0

            for iter, (real_img, labels) in (enumerate(self.data_loader)):
                self.D_optimizer.zero_grad()
                real_img = real_img.to(self.device)
                labels = labels.to(self.device)
                latentZ = torch.rand((labels.shape[0], self.z_dim)).to(self.device)
                labels_expand = labels.unsqueeze(2).unsqueeze(3).expand(labels.shape[0], labels.shape[1], self.input_size, self.input_size)
                
                D_real = self.D(real_img, labels_expand)
                D_real_loss = self.BCE_loss(D_real, torch.ones((D_real.shape[0], 1)).to(self.device))
                fake_img = self.G(latentZ, labels)
                D_fake = self.D(fake_img.detach(), labels_expand)
                D_fake_loss = self.BCE_loss(D_fake, torch.zeros((D_fake.shape[0], 1)).to(self.device))
            
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                self.D_optimizer.step()

                self.G_optimizer.zero_grad()
                latentZ = torch.rand((labels.shape[0], self.z_dim)).to(self.device)
                fake_img = self.G(latentZ, labels)
                D_fake = self.D(fake_img, labels_expand)
                G_loss = self.BCE_loss(D_fake, torch.ones((D_fake.shape[0], 1)).to(self.device))

                G_loss.backward()
                self.G_optimizer.step()
        
                epoch_GLoss += G_loss.item()
                epoch_DLoss += D_loss.item()
            
            print("DLoss: {:.2f}, GLoss:{:.2f}".format(epoch_DLoss,epoch_GLoss))
            self.Gscheduler.step()

            if epoch_DLoss < 10:
                for _ in range(5):
                    for iter, (real_img, labels) in (enumerate(self.data_loader)):
                        real_img = real_img.to(self.device)
                        labels = labels.to(self.device)
                        latentZ = torch.rand((labels.shape[0], self.z_dim)).to(self.device)
                        labels_expand = labels.unsqueeze(2).unsqueeze(3).expand(labels.shape[0], labels.shape[1], self.input_size, self.input_size)

                        self.G_optimizer.zero_grad()
                        latentZ = torch.rand((labels.shape[0], self.z_dim)).to(self.device)
                        fake_img = self.G(latentZ, labels)
                        D_fake = self.D(fake_img, labels_expand)

                        G_loss = self.BCE_loss(D_fake, torch.ones((D_fake.shape[0], 1)).to(self.device))
                        G_loss.backward()
                        self.G_optimizer.step()

            if not os.path.exists(f'/home/mhchen/Code/DLP/HW6/Results/Epoch_{epoch}/'):
                os.makedirs(f'/home/mhchen/Code/DLP/HW6/Results/Epoch_{epoch}/')

            model.save_models(f'/home/mhchen/Code/DLP/HW6/Results/Epoch_{epoch}/')
            
    def test(self, load_path=None):
        if load_path is not None:
            self.load_model(load_path)

        from evaluator import evaluation_model
        eval = evaluation_model()

        with open("test.json", "r") as f:
            test = json.load(f)
    
        with open("new_test.json", "r") as f:
            new_test = json.load(f)
        
        with open("objects.json", "r") as f:
            objects = json.load(f)

        test_label = []
        new_test_label = []

        for img_label in test:
            label = np.zeros(len(objects),dtype=np.float32)
            for l in img_label:
                label[objects[l]] = 1
            test_label.append(label)
        
        for img_label in new_test:
            label = np.zeros(len(objects),dtype=np.float32)
            for l in img_label:
                label[objects[l]] = 1
            new_test_label.append(label)

        test_label = torch.tensor(test_label).to(self.device)
        new_test_label = torch.tensor(new_test_label).to(self.device)

        latentZ = torch.rand((len(test), self.z_dim)).to(self.device)
        img_test = self.G(latentZ,test_label)
        img_test = (img_test+1)/2

        transform = [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        img_test_n = transforms.Compose(transform)(img_test).to(torch.float32)
        result_Test = eval.eval(img_test_n,test_label)

        latentZ = torch.rand((len(new_test), self.z_dim)).to(self.device)
        img_new_Test = self.G(latentZ,new_test_label)
        img_new_Test = (img_new_Test+1)/2

        img_new_Test_n = transforms.Compose(transform)(img_new_Test).to(torch.float32)
        result_new_Test = eval.eval(img_new_Test_n, new_test_label)

        print("-"*10 + " Testing Score " + "-"*10)
        print("Test Score: ", result_Test)
        print("new_Test Score: ", result_new_Test)

        img_test = torchvision.utils.make_grid(img_test,nrow=8)
        torchvision.utils.save_image(img_test, f'Results/Test/GAN.png')
        img_new_Test = torchvision.utils.make_grid(img_new_Test,nrow=8)
        torchvision.utils.save_image(img_new_Test, f'Results/New_Test/GAN.png')


if __name__=="__main__":
    model = CGAN()
    model.train(last_epoch=150)
    path = f'/home/mhchen/Code/DLP/HW6/Results/Epoch_{200}'
    model.test(load_path = path)


                 
