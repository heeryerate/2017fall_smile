############

#   @File name: emtion_recognition.py
#   @Author: Zhenyu Li
#   @Email: chunjukli@gmail.com

#   @Create date:   2017-09-07 12:37:02

#   @Description: read the local data files and train the data. Output will be the loss function and
#   the accuracy
#   @Example:


############

#import pylab
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import utils
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from PIL import Image
import pickle


with open('data.pickle', 'rb') as f:
	data  = pickle.load(f)



X =np.array( data['XTr']/255.0, dtype='f') #normalize 
y = data['yTr']                            #training images


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input is 49x64
        # padding=2 
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # self.conv1Bias = nn.Linear(1,1,1,32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 96, 5, padding=2)
        
        self.fc1 = nn.Linear(4608, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        #return x
        x = x.view(-1, 96*6*8)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

model = Model()

n = len(y)
print ('ylength', n)
batch_size = 4
model.train()
optimizer = optim.Adam(model.parameters())


train_loss = []
train_accu = []
j = 0h
#train the model
for epoch in range(200):
    for i in range(n//batch_size):
        optimizer.zero_grad()
        yD  =  y[i*batch_size:(i+1)*batch_size]
        #print (yD)
        yy = torch.from_numpy(yD)
        train_label = Variable(yy)
        #print (train_label)
        xD  =  X[i*batch_size:(i+1)*batch_size,:,:,:]
        xx = torch.from_numpy(xD) # creates a tensor 
        xOut = model.forward(Variable(xx))
#print (xOut.size())

        loss = F.nll_loss(xOut, train_label)
        loss.backward()    # calc gradients
        train_loss.append(loss.data[0])
        optimizer.step()   # update gradients
        #print (xOut)
        prediction = xOut.data.max(1)[1]   
        #print ("pre", xOut.data.max(1)[1])
        #print (xOut.data)
        accuracy = prediction.eq(train_label.data).sum()/batch_size*100
        train_accu.append(accuracy)
        if j % 10 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(j, loss.data[0], accuracy))
        j += 1

        #print (xOut)
 
# test on testing data 
model.eval()
yTe = data['yTe']
xTe =np.array( data['XTe']/255.0, dtype='f')
n_test = len(yTe)
correct = 0
for i in range(n_test//20): #batch size = 20 
    teX = xTe[i*20:(i+1)*20,:,:,:]
    testX = torch.from_numpy(teX)

    tey = yTe[i*20:(i+1)*20]
    testy = torch.from_numpy(tey)

    testingX, testing_label = Variable(testX, volatile = True), Variable(testy)
    testingOut = model(testingX)
    prediction = testingOut.data.max(1)[1]
    correct += prediction.eq(testing_label.data).sum()

print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / n_test))

plt.plot(np.arange(len(train_accu)), train_accu)
plt.show()


'''train_loader = torch.utils.data.DataLoader(
                                           datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                                           batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
                                          datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
                                          batch_size=1000)

for p in model.parameters():
    print(p.size())


'''






