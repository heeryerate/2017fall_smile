import numpy as np
import cv2

image = cv2.imread('book.jpg')
cv2.imshow('raw image', image)
rows = image.shape[0]
cols = image.shape[1] 

for r in range(rows): 
    for c in range(cols):
        image[r, c, 0] = 255-image[r, c, 0] #Pixelf = 255 - Pixeli
        image[r, c, 1] = 255-image[r, c, 1]
        image[r, c, 2] = 255-image[r, c, 2]

cv2.imshow('negative image', image)
cv2.waitKey()
cv2.destroyAllWindows()