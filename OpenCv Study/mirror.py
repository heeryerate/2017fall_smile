import numpy as np
import cv2

image = cv2.imread('wb.jpg')
rows = image.shape[0]
cols = image.shape[1]
#mirror_col = int(cols/2) 
new_cols = cols*2

# mirror_image = np.zeros(shape=(rows, new_cols, 3), dtype = np.uint8)

# print(image)
#image = np.concatenate((image, image[::-1]), axis=1)
#mirror_image = np.concatenate((image), axis=0)
image = cv2.flip( image, 1 )


# coli = 0
# rowa = 0
# for row in range(rows): 
#     for col in range(cols):
#         mirror_image[row, col, 0] = image[rowa, coli, 0]
#         mirror_image[row, col, 1] = image[rowa, coli, 1]
#         mirror_image[row, col, 2] = image[rowa, coli, 2]
#         coli += 1
#         if coli>=cols:
#             col=0

# rowa += 1
# if rowa>=rows:
#     rowa=0

        
# colj = cols
# rowj = 0
# for row in range(rows): 
#     for col in range(cols, new_cols):
#         mirror_image[row, col, 0] = image[rowj, colj, 0]
#         mirror_image[row, col, 1] = image[rowj, colj, 1]
#         mirror_image[row, col, 2] = image[rowj, colj, 2]
#         colj -= 1
# rowj +=1
# if rowj>=rows:
#     rowj=0
       
cv2.imshow('mirror image', image)

cv2.waitKey()
cv2.destroyAllWindows()