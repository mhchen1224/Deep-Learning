import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from models import unet, resnet34_unet
from oxford_pet import load_dataset
from evaluate import evaluate
from utils import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90degree, Compose

def train(args):
    #Define device, network model and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = unet.UNet().to(device) if args.Network=='Unet' else resnet34_unet.ResNet34_UNet().to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.learning_rate)

    #Load dataset and define dataloader
    train_dataset = load_dataset(data_path=args.data_path, mode='train', transform=Compose([RandomHorizontalFlip(p=0.5),
                                                                                            RandomVerticalFlip(p=0.5),
                                                                                            RandomRotation90degree()]))
    EvalTrainDataSet = load_dataset(data_path=args.data_path, mode='train', transform=None)
    EvalValidDataSet = load_dataset(data_path=args.data_path, mode='valid', transform=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    EvalTrain_loader = DataLoader(EvalTrainDataSet, batch_size=args.batch_size, num_workers=16)
    EvalValid_loader = DataLoader(EvalValidDataSet, batch_size=args.batch_size, num_workers=16)

    #training
    LossSet = []
    train_dice = []
    valid_dice = []
    for epoch in range(args.epochs):
        Total_Loss = 0
        DiceScore_train = 0
        DiceScore_valid = 0
        for data in (train_loader):
            model.train()
            image = data['image'].to(torch.float32).to(device)
            mask = data['mask'].to(torch.float32).to(device)
            pred = model(image)

            loss = nn.BCELoss()(pred,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Total_Loss += loss.item()

        
        for data in EvalTrain_loader:
            DiceScore_train += evaluate(model, data, device)
        for data in EvalValid_loader:
            DiceScore_valid += evaluate(model, data, device)

        DiceScore_train /= EvalTrain_loader.__len__()
        DiceScore_valid /= EvalValid_loader.__len__()
        LossSet.append(Total_Loss)
        train_dice.append(DiceScore_train.item())
        valid_dice.append(DiceScore_valid.item())

        print(f"{epoch:3d}/{args.epochs}, Loss:{Total_Loss:.2f}, DiceScore_train: {DiceScore_train:.2f}, DiceScore_valid: {DiceScore_valid:.2f}")

    if args.Network=='Unet':
        torch.save(model.state_dict(), "/home/mhchen/Code/DLP/Lab3-Binary_Semantic_Segmentation/saved_models/DL_Lab3_UNet_312554002_陳明宏.pth")
    else:
        torch.save(model.state_dict(), "/home/mhchen/Code/DLP/Lab3-Binary_Semantic_Segmentation/saved_models/DL_Lab3_ResNet34_UNet_312554002_陳明宏.pth")
    
    np.save(f'Loss_{args.Network}.npy', LossSet)
    np.save(f'TrainDice_{args.Network}.npy', train_dice)
    np.save(f'ValidDice_{args.Network}.npy', valid_dice)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--Network', '-net', type=str, default='Unet', help='Network')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)