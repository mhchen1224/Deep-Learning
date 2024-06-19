import argparse
from models import unet, resnet34_unet
from torch.utils.data import DataLoader
import torch
from evaluate import evaluate
from oxford_pet import load_dataset
import random
import matplotlib.pyplot as plt
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--Network', '-net', type=str, default='Unet', help='Network')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    #Construct the dataloader and load the trained model paprameter
    dataset = load_dataset(data_path=args.data_path, mode='test', transform=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = unet.UNet().to(device) if args.Network=='Unet' else resnet34_unet.ResNet34_UNet().to(device)
    model.load_state_dict(torch.load(args.model))

    # Test the performance on testing dataset
    Total = 0
    for idx, data in enumerate(loader):
        BatchDiceScore = evaluate(model,data,device)
        Total += BatchDiceScore
        print(f"Testing Batch {idx}/{loader.__len__()}, Batch Average Dice Score: {BatchDiceScore}")

    print(f"Average Testing Dice Score : {Total/loader.__len__()}")

    # Random inference 5 testing image and show the results 
    indices = random.sample(range(dataset.__len__()), 5)
    fig, axs = plt.subplots(3, 5, figsize=(12, 6))
    for i,idx in enumerate(indices):
        data = dataset.__getitem__(idx)
        image = torch.tensor(data['image']).to(torch.float32).to(device).unsqueeze(dim=0)
        with torch.no_grad():
            pred = model(image)

        axs[0,i].imshow(np.moveaxis(data['image'], 0, -1))
        axs[0,i].set_title(f'Figure {i+1}')
        axs[1,i].imshow(np.squeeze(data['mask'], axis=0))
        axs[1,i].set_title(f'Ground Truth {i+1}')
        axs[2,i].imshow(np.where(np.moveaxis(pred.cpu().squeeze(dim=0).numpy(), 0, -1) > 0.5, 1, 0)) 
        axs[2,i].set_title(f'Model Predict {i+1}') 
    plt.tight_layout()
    plt.show()

