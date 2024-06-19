import random
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from dataloader import BufferflyMothLoader
import  VGG19
import  ResNet50
import numpy as np
from torch.optim import lr_scheduler
import argparse

class network(object):
    def __init__(self, args):
        self.dataset_train = BufferflyMothLoader(root='dataset', mode='train')
        self.dataset_valid = BufferflyMothLoader(root='dataset', mode='valid')
        self.dataset_test = BufferflyMothLoader(root='dataset', mode='test')
        self.args = args
        self.batchSize = args.BatchSize

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ",self.device)
        
        if args.Network == 'ResNet50':
            self.network = ResNet50.ResNet50().to(self.device)
        elif args.Network == 'VGG19':
            self.network = VGG19.VGGNet().to(self.device)
        else:
            raise ValueError("Arguments Network wrong!")
        
        if args.Optimizer == 'RAdam':
            self.network_optimizer = torch.optim.RAdam(self.network.parameters(), lr=args.lr)
        elif args.Optimizer == 'SGD':
            self.network_optimizer = torch.optim.SGD(self.network.parameters(), lr=args.lr)
        else:
            raise ValueError("Arguments Optimizer wrong!")

        #self.scheduler = lr_scheduler.StepLR(self.network_optimizer, step_size=1, gamma=0.98)

        self.trainingTotalBatch = self.dataset_train.__len__()//self.batchSize + (1 if self.dataset_train.__len__()%self.batchSize!=0 else 0)

    def evaluate(self, mode):
        self.network.eval()
        with torch.no_grad():
            if mode == 'train':
                count = 0
                indexs = list(range(0, self.dataset_train.__len__()))
                TotalBatch = self.dataset_train.__len__()//self.batchSize + (1 if self.dataset_train.__len__()%self.batchSize!=0 else 0)

                for batch in range(TotalBatch):
                    start_index = batch*self.batchSize
                    end_index = (batch+1)*self.batchSize if batch!=TotalBatch-1 else -1
                    index = indexs[start_index:end_index]
                    imgs, labels = self.dataset_train.__getitem__(index, task='eval')
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    pred = self.network(imgs)
                    pred = torch.argmax(pred, dim=1)
                    count += (pred == labels).sum().item()
                return count/self.dataset_train.__len__()
            else:
                count = 0
                indexs = list(range(0, self.dataset_valid.__len__()))
                TotalBatch = self.dataset_valid.__len__()//self.batchSize + (1 if self.dataset_valid.__len__()%self.batchSize!=0 else 0)
                for batch in range(TotalBatch):
                    start_index = batch*self.batchSize
                    end_index = (batch+1)*self.batchSize if batch!=TotalBatch-1 else -1
                    index = indexs[start_index:end_index]
                    imgs, labels = self.dataset_valid.__getitem__(index, task='eval')
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    pred = self.network(imgs)
                    pred = torch.argmax(pred, dim=1)
                    count += (pred == labels).sum().item()
                return count/self.dataset_valid.__len__()

    def test(self):
        self.network.eval()
        count = 0
        indexs = list(range(0, self.dataset_test.__len__()))
        TotalBatch = self.dataset_test.__len__()//self.batchSize + (1 if self.dataset_test.__len__()%self.batchSize!=0 else 0)
        for batch in range(TotalBatch):
            start_index = batch*self.batchSize
            end_index = (batch+1)*self.batchSize if batch!=TotalBatch-1 else self.dataset_test.__len__()
            index = indexs[start_index:end_index]
            imgs, labels = self.dataset_test.__getitem__(index, task='eval')
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            pred = self.network(imgs)
            pred = torch.argmax(pred, dim=1)
            batch_count = (pred == labels).sum().item()
            count += batch_count
            print("Test batch {:2d}, batch predict accuracy: {:2d}/{:2d}({:.2f})".format(batch,batch_count,end_index-start_index,batch_count/(end_index-start_index)))
        
        print("Total predict accuracy: {:3d}/{:3d}({:2.2f})".format(count,self.dataset_test.__len__(),count/self.dataset_test.__len__()))

    def train(self):
        self.network.train()
        training_indexs = list(range(0, self.dataset_train.__len__()))
        random.shuffle(training_indexs)
        TotalLoss = 0
        for batch in (range(self.trainingTotalBatch)):
            start_index = batch*self.batchSize
            end_index = (batch+1)*self.batchSize if batch!=self.trainingTotalBatch-1 else -1
            index = training_indexs[start_index:end_index]
            imgs, labels = self.dataset_train.__getitem__(index, task='train')
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            pred = self.network(imgs)
            Loss = CrossEntropyLoss()(pred,labels)

            self.network_optimizer.zero_grad()
            Loss.backward()
            self.network_optimizer.step()
            TotalLoss += Loss.item()
        
        #self.scheduler.step()
        return TotalLoss

    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))
        
    def save_weights(self):
        torch.save(self.network.state_dict(), f"{args.Network}_{args.Optimizer}_{args.BatchSize}_best.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Epoch", type=int, default=100,
        help="Total training epoch")
    parser.add_argument("--BatchSize", type=int, default=32,
        help="Training Batch Size")
    parser.add_argument("--Optimizer", type=str, default="",
        help="Training Optimizer RAdam or SGD")
    parser.add_argument("--Network", type=str, default="",
        help="Network")
    parser.add_argument("--lr", type=float, default=1e-2,
        help="learning rate")
    args = parser.parse_args()

    # print("-"*10)
    # print("args.Epoch: ", args.Epoch)
    # print("args.BatchSize: ", args.BatchSize)
    # print("args.Optimizer: ", args.Optimizer)
    # print("args.Network: ", args.Network)
    # print("args.lr: ", args.lr)
    # print("-"*10)

    max_epoch = args.Epoch
    lr = args.lr
    batchSize = args.BatchSize

    EpochSet = []
    LossSet = []
    TrainEvalSet = []
    ValidEvalSet = []

    # Construct model with Classifier(either VGG19 or ResNet50) and define train, eval and test function
    model = network(args=args)

    for epoch in range(max_epoch):
        #------------- training phase -------------#
        Loss = model.train()

        #------------- evaluate phase -------------#
        evalTrain = model.evaluate(mode='train')
        evalvalid = model.evaluate(mode='valid')

        #------------- print result -------------#
        LossSet.append(Loss)
        TrainEvalSet.append(evalTrain)
        ValidEvalSet.append(evalvalid)
        print("epoch: {:2d}, Total Loss: {:8.2f}, TrainAcc: {:5.2f}, ValidAcc: {:5.2f}".format(epoch,Loss,evalTrain,evalvalid))

    #------------- testing phase -------------#
    model.test()
    #------------- save result -------------#
    model.save_weights()
    np.save(f'LossSet_{args.Network}_{args.Optimizer}_{args.BatchSize}_best.npy', LossSet)
    np.save(f'TrainEvalSet_{args.Network}_{args.Optimizer}_{args.BatchSize}_best.npy', TrainEvalSet)
    np.save(f'ValidEvalSet_{args.Network}_{args.Optimizer}_{args.BatchSize}_best.npy', ValidEvalSet)

    # Load model and test
    # model = network(lr=lr, batchSize=batchSize,args=args)
    # if args.Network=="VGG19":
    #     model.load_model('BestResult/VGG19_SGD_50.pth')
    # else:
    #     model.load_model('BestResult/ResNet50_RAdam_80.pth')
    # model.test()