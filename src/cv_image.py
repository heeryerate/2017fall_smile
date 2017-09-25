import cv2
import sys

image_path = "./images/test.jpg"

img = cv2.imread(image_path)

cv2.imshow("face:", img)
cv2.waitKey(0)
