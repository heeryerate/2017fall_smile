import cv2
img_path = 'book.jpg'
image = cv2.imread(filename = img_path)
cv2.imshow('image',image)
cv2.waitKey()
cv2.destroyAllWindows()