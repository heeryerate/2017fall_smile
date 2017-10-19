import cv2

img_path = 'book.jpg'
image = cv2.imread(filename=img_path)
cv2.imshow(winname='raw image',mat=image)


x, y = 100, 400

cv2.putText(img = image, text = 'Beautiful Scene!',
            org = (x, y), fontFace = cv2.FONT_HERSHEY_TRIPLEX,
            fontScale = 1, color = (255,255,255))

cv2.imshow('add text  on image',image)
cv2.waitKey()
cv2.destroyAllWindows()