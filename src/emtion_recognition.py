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



X = np.array( data['XTr']/255.0, dtype ='f') #normalize
y = data['yTr']                            #training images
yTe = data['yTe']
xTe =np.array( data['XTe']/255.0, dtype='f')
print (len(X), len(xTe))






class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input is 128x128
        # padding=2
        self.conv1 = nn.Conv2d(1, 96, 7, padding = 2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding = 2)
        self.conv3 = nn.Conv2d(256, 512, 3, padding = 2)
        self.conv4 = nn.Conv2d(512, 512, 3, padding = 2)
        self.conv5 = nn.Conv2d(512, 512, 3, padding = 2)

        self.fc1 = nn.Linear(512*16*16, 4048)
        self.fc2 = nn.Linear(4048, 4049) #1024
        self.fc3 = nn.Linear(4049, 7)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), 3)
        # return x
        x = x.view(-1, 512*16*16)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

class NaiveModel(nn.Module):
    def __init__(self):
        super(NaiveModel, self).__init__()
        # input is 128x128
        # padding=2
        self.conv1 = nn.Conv2d(1, 4, 11, padding = 2)
        # self.conv2 = nn.Conv2d(96, 256, 5, padding = 2)
        # self.conv3 = nn.Conv2d(256, 512, 3, padding = 2)
        # self.conv4 = nn.Conv2d(512, 512, 3, padding = 2)
        # self.conv5 = nn.Conv2d(512, 512, 3, padding = 2)

        self.fc1 = nn.Linear(4*83*83, 7)
        # self.fc2 = nn.Linear(4048, 4049) #1024
        # self.fc3 = nn.Linear(4049, 7)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.max_pool2d(F.relu(self.conv5(x)), 3)
        x = x.view(-1, 4*83*83)   # reshape Variable
        x = F.relu(self.fc1(x))
        return F.log_softmax(x)



# model = Model()
model = Model()

n = len(y)

#print ('ylength', n)
#print ('testLength', nTest)
batch_size = 20
model.train()
optimizer = optim.Adam(model.parameters())

train_loss = []
avg_train_accu = []
test_loss = []
avg_test_accu = []

def testing_acc():

    model.eval()

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
        test_accuracy = 100. * correct / n_test
        test_accu.append(test_accuracy)
    avg_test_accu.append(np.mean(test_accu))
    print('\nTest set: Accuracy: {:.3f}'.format(test_accuracy))

    #plt.plot(np.arange(len(train_accu)), train_accu)
    #plt.show()


j = 0
#train the model
for epoch in range(200):
    train_accu = []
    test_accu = []
    for i in range(n//batch_size):
        optimizer.zero_grad()
        yD  =  y[i*batch_size:(i+1)*batch_size]
        #print (yD)
        yy = torch.from_numpy(yD)
        #print ('label', Variable(yy))
        train_label = Variable(yy)
        #print (train_label)
        xD  =  X[i*batch_size:(i+1)*batch_size,:,:,:]
        xx = torch.from_numpy(xD) # creates a tensor
        xOut = model.forward(Variable(xx))
        #break
        loss = F.nll_loss(xOut, train_label)
        loss.backward()    # calc gradients
        train_loss.append(loss.data[0])
        optimizer.step()   # update gradients
        #print (xOut)
        Train_prediction = xOut.data.max(1)[1]
        #print ("pre", xOut.data.max(1)[1])
        #print (xOut.data)
        trainAccuracy = Train_prediction.eq(train_label.data).sum()/batch_size*100
        #print (trainAccuracy)
        train_accu.append(trainAccuracy)
        if j % 10 == 0:
            print('Train Step: {}\t\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(j, loss.data[0], trainAccuracy))
        #if j % 100 == 0:
            #pass
            #testing_acc()
        j += 1
    avg_train_accu.append(np.mean(train_accu))
    print ('epoch')
    testing_acc()

#print (train_accu)
#print (test_accu)
#x = np.arange(0, 100)
plt.plot(avg_train_accu, label = 'training')
plt.plot(avg_test_accu, label = 'testing')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))
