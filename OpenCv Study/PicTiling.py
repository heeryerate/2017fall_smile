import numpy as np
import cv2

image = cv2.imread('book.jpg') 
#get the size of pictures 
rows = image.shape[0]
cols = image.shape[1] 

#new size 2*m 3*n
new_rows = rows * 3
new_cols = cols * 4 

#null array for new picture 
new_image = np.zeros(shape=(new_rows, new_cols, 3), dtype=np.uint8) 

#copy from old pixel
row = 0
col = 0 

for R in range (3):
 for now_row in range(new_rows): 
    for now_col in range(new_cols):
        new_image[now_row, now_col, R] = image[row, col, R]
       # new_image[now_row, now_col, R] = image[row, col, R]
       ##After finishing a full image, start copying from [0,0,0] again new_image[now_row, now_col, R] = image[row, col, R]
        col+=1 
        #After finishing a full image, start copying from [0,0,0] again
        if col>=cols:
           col = 0




    row+=1 
   #After finishing a full image, start copying from [0,0,0] again
    if row>=rows:
        row=0

cv2.imshow('new image', new_image)
cv2.waitKey()
cv2.destroyAllWindows()