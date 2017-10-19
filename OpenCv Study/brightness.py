wimport cv2
import os
import numpy as np


img_path = 'book.jpg'
image = cv2.imread(filename=img_path)
cv2.imshow(winname='raw image',mat=image) 

row_num = image.shape[0]
column_num = image.shape[1] 

#same pic as before
#bright_image = image.copy() 
new_image = np.zeros(shape=(row_num,column_num,3), dtype = np.uint8)

#RGB * 1.5, make it brighter
for RGB in range (3):
	for row in range(row_num): 
	    for column in range(column_num):
	        #bright_image[row, column, 0] = bright_image[row, column, 0]* 1.5
	        #bright_image[row, column, 1] = bright_image[row, column, 1] * 1.5
	        new_image [row, column, RGB] = image [row, column, RGB] * 1.5

cv2.imshow('show bright image',new_image)

black_image = image.copy() 
#RGB/1.5, make it darker
for row in range(row_num): 
    for column in range(column_num):
        black_image[row, column, 0] = black_image[row, column, 0]* 0.5
        black_image[row, column, 1] = black_image[row, column, 1] * 0.5
        black_image[row, column, 2] = black_image[row, column, 2] * 0.5

cv2.imshow(winname='show black image',mat=black_image)
cv2.waitKey()
cv2.destroyAllWindows()
