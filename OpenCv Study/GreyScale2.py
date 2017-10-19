import numpy as np
import cv2

image = cv2.imread('book.jpg')
rows = image.shape[0]
cols = image.shape[1] 

for row in range(rows): 
    for col in range(cols):
        gray = 0.11*image[row,col,0]+0.59*image[row,col,1]+0.3*image[row,col,2]
        image[row, col, :] = gray
cv2.imshow('formula image', image)
cv2.waitKey()
cv2.destroyAllWindows()