import cv2
import numpy as np
img = np.zeros(shape = (512,512,3), dtype = np.uint8)
cv2.line(img,(0,0),(511,511),(200,100,200),10)
cv2.imshow('gray',img)
cv2.waitKey()
cv2.destroyAllWindows()