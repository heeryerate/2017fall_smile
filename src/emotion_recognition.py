############

#   @File name: emtion_recognition.py
#   @Author: Zhenyu Li
#   @Email: chunjukli@gmail.com

#   @Create date:   2017-09-07 12:37:02

#   @Description: read the local data files and train the data. Output will be the loss function and
#   the accuracy
#   @Example:


############

import pylab
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import utils
from matplotlib.pyplot import *
from PIL import Image
import pickle


with open('data.pickle', 'rb') as f:
	data  = pickle.load(f)



X =np.array( data['XTr']/255.0, dtype='f')
y = data['yTr']


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(12288, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*16*12)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Model()

n = len(y)
BS = 10


for epoch in range(10):
    for i in range(n//BS):

        yD  =  y[i*BS:(i+1)*BS]
        xD  =  X[i*BS:(i+1)*BS,:,:,:]


        xx = torch.from_numpy(xD)

        xOut = model.forward(Variable(xx))
# 64   160  122
        print (xOut)




'''train_loader = torch.utils.data.DataLoader(
                                           datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                                           batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
                                          datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
                                          batch_size=1000)

for p in model.parameters():
    print(p.size())

optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.train()
train_loss = []
train_accu = []
i = 0
for epoch in range(15):
    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()    # calc gradients
        train_loss.append(loss.data[0])
        optimizer.step()   # update gradients
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        train_accu.append(accuracy)
        if i % 1000 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data[0], accuracy))
        i += 1

plt.plot(np.arange(len(train_loss)), train_loss)
plt.plot(np.arange(len(train_accu)), train_accu)
model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

'''






