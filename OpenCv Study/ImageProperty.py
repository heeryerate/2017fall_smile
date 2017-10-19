import cv2 
img_path = 'book.jpg'
image = cv2.imread(img_path)
print(type(image))
print(image.ndim)
print(image.shape)
print(image[4,5])
print(type(image[4,5]))
		