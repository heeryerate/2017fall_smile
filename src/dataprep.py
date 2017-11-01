

import numpy as np
import os
import cv2
import random
import pickle


folderPredix = '../images/'
cls  =['anger','disgust','fear','happy','neutral','sadness','surprise']
		# 0		# 1		   # 2	# 3		   #4		# 5			# 6


y = []	# labels
X = []	# images 
i = 0
for cl  in cls:
    foldLoc = folderPredix+cl+'/'
    for im in  os.listdir(foldLoc):
        f = foldLoc+im
        if im[0] != 'S':
            continue
        img = cv2.imread(f,-1)
        #resize by multiplying 0.1
        res = cv2.resize(img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)
        #new image size: 49x64
        imM = np.array(res)
        X.append(imM)
        y.append(i)

    i=i+1

n = len(y)
idx = [i for i in range(n)]

np.random.shuffle(idx)

XTr=[]
XTe=[]
yTr=[]
yTe=[]

for id in idx:
    xx = np.reshape(X[id],[1,49,64])
    
    if random.random()<0.7:	#70% training 
        XTr.append(xx)
        yTr.append(y[id])
    else:
        XTe.append(xx)
        yTe.append(y[id])

with open('data.pickle', 'wb') as f:
    data={}
    data['XTr']=np.array(XTr)
    data['yTr']=np.array(yTr)
    data['XTe']=np.array(XTe)
    data['yTe']=np.array(yTe)
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
