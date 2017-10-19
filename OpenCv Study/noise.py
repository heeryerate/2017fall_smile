import numpy as np
import cv2

image = cv2.imread('book.jpg')
rows = image.shape[0]
cols = image.shape[1] 
#5000 noises
noises = 5000 
for i in range(noises): 
    row = np.random.randint(0, rows)
    col = np.random.randint(0, cols)
    image[row, col, :] = np.array([225,20,19])

cv2.imshow('noise image', image)
cv2.waitKey()
cv2.destroyAllWindows()