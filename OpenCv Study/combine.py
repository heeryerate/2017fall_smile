import numpy as np
import cv2

#A、B、C图的尺寸相同
A_img = cv2.imread('com1.jpg')
B_img = cv2.imread('com2.jpg')
cv2.imshow('A', A_img)
cv2.imshow('B', B_img)

rows = A_img.shape[0]
cols = A_img.shape[1]
C_img = np.zeros(shape=(rows, cols, 3), dtype=np.uint8) 

for r in range(rows): 
    for c in range(cols):
        C_img[r, c, 0] = (int( A_img[r, c, 0])+int (B_img[r, c, 0]))/2
        C_img[r, c, 1] = (int( A_img[r, c, 1])+int (B_img[r, c, 1]))/2
        C_img[r, c, 2] = (int( A_img[r, c, 2])+int (B_img[r, c, 2]))/2

cv2.imshow('C',C_img)
cv2.waitKey()
cv2.destroyAllWindows()