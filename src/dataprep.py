

import numpy as np
import os
import cv2
import random
import pickle


folderPredix = '../images/'
cls  =['Anger','Disgust','Fear','Happy','Neutral','Sadness','Surprise']
		# 0		# 1		   # 2	# 3		   #4		# 5			# 6
'''
img = cv2.imread("/Users/zhenyu_li/Desktop/Fall2017/2017fall_smile/images/Anger/S011.jpg")
s = img.shape
print (s)
cv2.imshow("Faces found" ,img)

#cv2.imwrite('./images/test_faces.png',image)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
y = []	# labels
X = []	# images 
i = 0
for cl  in cls:
    foldLoc = folderPredix+cl+'/'
    for im in  os.listdir(foldLoc):     
        f = foldLoc+im
        #print (im[0])
        if im[0] != '.':
            continue
        print (f)    
        img = cv2.imread(f,0) #greyscale
        print (img.size())

        #resize by multiplying 0.25
        res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        #new image size: 64x64
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
    xx = np.reshape(X[id],[1,128,128])
    
    if random.random()<0.85:	#85% training 
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


