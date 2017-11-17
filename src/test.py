import cv2
import numpy as np
img = cv2.imread('./images/123.jpg')
print(img.shape)
xx = np.reshape(img,[1,1080,1080])
print(xx)
#res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
