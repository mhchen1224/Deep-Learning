from utils import CLEVRLoader, ClassConditionedUnet
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
import torch.nn as nn
from tqdm import tqdm
import json
import numpy as np
import torchvision

class DDPM(object):
    def __init__(self,**kwargs):
        # Init hyperparameters for DDPM
        self.__dict__.update(kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformsList = [transforms.Resize((64, 64)), 
                        transforms.ToTensor(), 
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                        ]

        # Create dataset
        self.data_loader = DataLoader(CLEVRLoader(transformsList), batch_size=batch_size, shuffle=True)
        # Create a scheduler
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.denoise_timesteps, beta_schedule='squaredcos_cap_v2')
        self.Unet = ClassConditionedUnet(num_classes=num_classes, class_emb_size=class_emb_size).to(self.device)

        # Our loss function
        self.loss_fn = nn.MSELoss()

        # The optimizer
        self.opt = torch.optim.Adam(self.Unet.parameters(), lr=lr) 
    
    def train(self):
        for epoch in range(self.n_epochs):
            Total_Loss = 0
            for x, y in tqdm(self.data_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                noise = torch.randn_like(x)
                timesteps = torch.randint(0, self.denoise_timesteps, (x.shape[0],)).long().to(self.device)
                noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
                # Get the model prediction
                pred = self.Unet(noisy_x, timesteps, y) # Note that we pass in the labels y
                # Calculate the loss
                loss = self.loss_fn(pred, noise) # How close is the output to the noise
                # Backprop and update the params:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                Total_Loss += loss.item()

            print(f'Finished epoch {epoch}.Total Loss: {Total_Loss:05f}')

            torch.save(self.Unet.class_emb.state_dict(),f"/home/mhchen/Code/DLP/HW6/DDPM_Unet_class_emb_{epoch}.pth")
            torch.save(self.Unet.model.state_dict(),f"/home/mhchen/Code/DLP/HW6/DDPM_Unet_model_{epoch}.pth")
    
    def load(self):
        self.Unet.model.load_state_dict(torch.load(f'/home/mhchen/Code/DLP/HW6/DDPM_Unet_model.pth'))
        self.Unet.class_emb.load_state_dict(torch.load(f'/home/mhchen/Code/DLP/HW6/DDPM_Unet_class_emb.pth'))
    
    def test(self):
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

        x = torch.randn(len(test), 3, self.input_size, self.input_size).to(self.device)
        for t in tqdm(range(self.denoise_timesteps)):
            with torch.no_grad():
                residual = self.Unet(x, self.denoise_timesteps-t-1, test_label)

            # Update sample with step
            x = self.noise_scheduler.step(residual, self.denoise_timesteps-t-1, x).prev_sample

        result_Test = eval.eval(x,test_label)
        img_Test = torchvision.utils.make_grid(x,nrow=8)

        x = torch.randn(len(test), 3, self.input_size, self.input_size).to(self.device)
        for t in tqdm(range(self.denoise_timesteps)):
            with torch.no_grad():
                residual = self.Unet(x, self.denoise_timesteps-t-1, new_test_label)  

            # Update sample with step
            x = self.noise_scheduler.step(residual, self.denoise_timesteps-t-1, x).prev_sample
        result_new_Test = eval.eval(x, new_test_label)
        img_new_Test = torchvision.utils.make_grid(x,nrow=8)
        
        print("-"*10 + " Testing Score " + "-"*10)
        print("Test Score: ",result_Test)
        print("new_Test Score: ",result_new_Test)
        torchvision.utils.save_image(img_Test, f'Results/Test/DDPM.png', normalize=True)
        torchvision.utils.save_image(img_new_Test, f'Results/New_Test/DDPM.png',normalize=True)
    
    def denoise_process(self):
        label = np.zeros((1,24),dtype=np.float32)
        label[0][7] = 1
        label[0][9] = 1
        label[0][22] = 1
        label = torch.tensor(label).to(self.device)
        process = torch.tensor([]).to(self.device)
        x = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        for t in tqdm(range(self.denoise_timesteps)):
            with torch.no_grad():
                residual = self.Unet(x, self.denoise_timesteps-t-1, label)
            x = self.noise_scheduler.step(residual, self.denoise_timesteps-t-1, x).prev_sample
            if (t+1)%100 ==0:
                process = torch.cat([process,x],dim=0)

        process = (process+1)/2
        torchvision.utils.save_image(process, f'Results/DDPM_process.png', nrow=10)


if __name__=='__main__':
    batch_size = 64
    n_epochs = 50
    lr = 3e-4
    num_classes = 24
    class_emb_size = 16
    input_size = 64
    denoise_timesteps = 1000
    model = DDPM(batch_size=batch_size,n_epochs=n_epochs,lr=lr,num_classes=num_classes,
                 class_emb_size=class_emb_size,input_size=input_size,denoise_timesteps=denoise_timesteps)
    model.train()
    # model.load()
    # model.denoise_process()
    # model.test()



