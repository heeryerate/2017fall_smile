import numpy as np
import cv2

image = cv2.imread('book.jpg')
rows = image.shape[0]
cols = image.shape[1] 

for row in range(rows): 
    for col in range(cols):
        average = np.mean(image[row,col,:])
        image[row, col, :] = average
cv2.imshow('average image', image)
cv2.waitKey()
cv2.destroyAllWindows()