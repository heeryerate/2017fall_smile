import numpy as np
import cv2

raw_image = cv2.imread('book.jpg')
cv2.imshow('raw image',raw_image) 

#get the size of pic(row & column)
rows = raw_image.shape[0]
cols = raw_image.shape[1] 

#create a null array and be filled in with random number
image = np.zeros(shape=(rows,cols,3), dtype=np.uint8) 
for r in range(rows): 
    for c in range(cols):
        image[r, c, 0] = np.random.randint(0, 255)
        image[r, c, 1] = np.random.randint(0, 255)
        image[r, c, 2] = np.random.randint(0, 255)

cv2.imshow('random pixel image', image)
cv2.waitKey()
cv2.destroyAllWindows()
