import numpy as np
import cv2

image = cv2.imread('book.jpg')
cvt_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('cvtColor image', cvt_image)
cv2.waitKey()
cv2.destroyAllWindows()