import numpy as np
import matplotlib.pyplot as plt

class Network(object):
    def __init__(self,input_size = 2, hidden_size=32, output_size=1, lr=5e-2, actFuncs=['sigmoid','sigmoid','sigmoid']):

        #Initialize the network parameter
        self.W1 = np.random.normal(0, 1, (input_size, hidden_size)) 
        self.W2 = np.random.normal(0, 1, (hidden_size, hidden_size))
        self.W3 = np.random.normal(0, 1, (hidden_size, output_size))
        self.lr = lr
        self.actFuncs= actFuncs

        #Record forwardPass
        self.Z1 = 0
        self.Z1Act = 0
        self.Z2 = 0
        self.Z2Act = 0
        self.Z3 = 0
        self.Z3Act = 0

        #Record GradLoss
        self.GradLoss = 0
        # Record Momentum
        self.MomW3 = 0
        self.MomW2 = 0
        self.MomW1 = 0


    def activateFunv(self, input, func='sigmoid'):
        if func == 'sigmoid':
            output = 1/(1+np.exp(-input))
        elif func == 'ReLU':
            output = np.maximum(0.0, input)
        return output
    
    def derivative_activateFunv(self, input, func='sigmoid'):
        if func == 'sigmoid':
            x = self.activateFunv(input, func='sigmoid')
            output = np.multiply(x , 1.0 - x)
        elif func == 'ReLU':
            output = np.where(input > 0, 1, 0)
        return output
    
    def computeMSELoss(self, pred, label):
        Loss = (pred - label)**2
        self.GradLoss = 2 * (pred - label)
        return np.sum(Loss)/pred.shape[0]

    def forwardPass(self, x):
        self.Z1 = x @ self.W1
        self.Z1Act = self.activateFunv(self.Z1, func=self.actFuncs[0])
        self.Z2 = self.Z1Act @ self.W2
        self.Z2Act = self.activateFunv(self.Z2, func=self.actFuncs[1])
        self.Z3 = self.Z2Act @ self.W3
        self.Z3Act = self.activateFunv(self.Z3, func=self.actFuncs[2])
        return self.Z3Act
    
    def Backpropagation(self):
        grad_LossAct = self.derivative_activateFunv(self.Z3, func=self.actFuncs[2]) * self.GradLoss
        grad_W3 = self.Z2Act.T @ grad_LossAct
        grad_Z2Act = grad_LossAct @ self.W3.T
        grad_Z2 = self.derivative_activateFunv(self.Z2, func=self.actFuncs[1]) * grad_Z2Act
        grad_W2 = self.Z1Act.T @ grad_Z2
        grad_Z1Act = grad_Z2 @ self.W2.T
        grad_Z1 = self.derivative_activateFunv(self.Z1, func=self.actFuncs[0]) * grad_Z1Act
        grad_W1 = x.T @ grad_Z1

        self.W3 -= self.lr * grad_W3
        self.W2 -= self.lr * grad_W2
        self.W1 -= self.lr * grad_W1
    
    def Backpropagation_wMomentum(self, epoch):
        grad_LossAct = self.derivative_activateFunv(self.Z3, func=self.actFuncs[2]) * self.GradLoss
        grad_W3 = self.Z2Act.T @ grad_LossAct
        grad_Z2Act = grad_LossAct @ self.W3.T
        grad_Z2 = self.derivative_activateFunv(self.Z2, func=self.actFuncs[1]) * grad_Z2Act
        grad_W2 = self.Z1Act.T @ grad_Z2
        grad_Z1Act = grad_Z2 @ self.W2.T
        grad_Z1 = self.derivative_activateFunv(self.Z1, func=self.actFuncs[0]) * grad_Z1Act
        grad_W1 = x.T @ grad_Z1

        if epoch == 0:
            self.MomW3 = grad_W3 
            self.MomW2 = grad_W2 
            self.MomW1 = grad_W1 
        else:
            self.MomW3 = 0.1 * grad_W3 + 0.9 * self.MomW3
            self.MomW2 = 0.1 * grad_W2 + 0.9 * self.MomW2
            self.MomW1 = 0.1 * grad_W1 + 0.9 * self.MomW1

        self.W3 -= self.lr * self.MomW3
        self.W2 -= self.lr * self.MomW2
        self.W1 -= self.lr * self.MomW1      

    def testing(self,x,y):
        Z1 = x @ self.W1
        Z1Act = self.activateFunv(Z1, func=self.actFuncs[0])
        Z2 = Z1Act @ self.W2
        Z2Act = self.activateFunv(Z2, func=self.actFuncs[1])
        Z3 = Z2Act @ self.W3
        pred = self.activateFunv(Z3, func=self.actFuncs[2])
        Loss = np.sum(abs(pred-y))/pred.shape[0]
        for i in range(len(y)):
            print("Iter{:3d} |\tGround Truth: {:d}|\tPrediction: {:.5f} |".format(i+1,y[i][0],pred[i][0]))
        
        pred = (pred > 0.5).astype(int)
        Acc = (1 - np.sum(abs(pred-y))/pred.shape[0])*100
        print("Loss = {:.5f}, accuracy = {:.1f}%".format(Loss,Acc))
    
class Network_woAct(object):
    def __init__(self,input_size = 2, hidden_size=16, output_size=1, lr=5e-6):

        #Initialize the network parameter
        self.W1 = np.random.normal(0, 1, (input_size, hidden_size)) 
        self.W2 = np.random.normal(0, 1, (hidden_size, hidden_size))
        self.W3 = np.random.normal(0, 1, (hidden_size, output_size))
        self.lr = lr

        #Record forwardPass
        self.Z1 = 0
        self.Z2 = 0
        self.Z3 = 0

        #Record GradLoss
        self.GradLoss = 0
    
    def computeMSELoss(self, pred, label):
        Loss = (pred - label)**2
        self.GradLoss = 2 * (pred - label)
        return np.sum(Loss)/pred.shape[0]

    def forwardPass(self, x):
        self.Z1 = x @ self.W1
        self.Z2 = self.Z1 @ self.W2
        self.Z3 = self.Z2 @ self.W3
        return self.Z3
    
    def Backpropagation(self):
        grad_W3 = self.Z2.T @ self.GradLoss
        grad_Z2 = self.GradLoss @ self.W3.T
        grad_W2 = self.Z1.T @ grad_Z2
        grad_Z1 = grad_Z2 @ self.W2.T
        grad_W1 = x.T @ grad_Z1

        self.W3 -= self.lr * grad_W3
        self.W2 -= self.lr * grad_W2
        self.W1 -= self.lr * grad_W1

    def testing(self,x,y):
        Z1 = x @ self.W1
        Z2 = Z1 @ self.W2
        pred = Z2 @ self.W3
        Loss = np.sum(abs(pred-y))/pred.shape[0]
        for i in range(len(y)):
            print("Iter{:3d} |\tGround Truth: {:d}|\tPrediction: {:.5f} |".format(i+1,y[i][0],pred[i][0]))
        
        pred = (pred > 0.5).astype(int)
        Acc = (1 - np.sum(abs(pred-y))/pred.shape[0])*100
        print("Loss = {:.5f}, accuracy = {:.1f}%".format(Loss,Acc))
    
def show_result(x, y, pred_y,filename):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth", fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1, 2, 2)
    plt.title("Predict Result", fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    #plt.show()
    plt.savefig(filename)

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21,1)

if __name__ == "__main__":
    x,y = generate_XOR_easy()
    network = Network()
    LossList = []
    for i in range(10000):
        pred = network.forwardPass(x)
        Loss = network.computeMSELoss(pred, y)
        network.Backpropagation()
        LossList.append(Loss)
        if (i+1) % 500 == 0:
            print("epoch: {:5d} Loss: {:.5f}".format(i+1,Loss))
    
    #if value > 0.5, set predict class=1, otherwise set predict class=0
    network.testing(x,y)
    pred = network.forwardPass(x)
    pred = (pred > 0.5).astype(int)
    show_result(x,y,pred,f'XOR')

    plt.clf()
    epo = [i+1 for i in range(len(LossList))]
    plt.plot(epo,LossList)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve')
    plt.savefig(f'Loss')
