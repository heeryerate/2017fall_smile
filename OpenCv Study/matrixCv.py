import numpy as np
import cv2

image = cv2.imread('book.jpg')
cv2.imshow('raw image', image) 

#transpose(), 0&1
new_image = image.copy().transpose(1,0,2) 
cv2.imshow('transpose image', new_image)
cv2.waitKey()
cv2.destroyAllWindows()