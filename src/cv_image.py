# @Author: Judy
# @Date:   2017-09-27T16:49:38-04:00
# @Email:
# @Filename: cv_image.py
# @Last modified by:   Heerye
# @Last modified time: 2017-09-27T17:10:29-04:00



import cv2
import sys

# imagePath = sys.argv[1]
# cascPath = sys.argv[2]

imagePath = "./images/test.jpg"
cascPath = "./haarcascade_frontalface_default.xml"

#face recognition
faceCascade = cv2.CascadeClassifier(cascPath)
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


#load the image
image = cv2.imread(imagePath)

#GreyScale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect face
#faces = faceCascade.detectMultiScale(gray, 1.2, 5)
faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(6, 6),
        
    )


   
print ("Yay~ Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 105, 200), 2)

cv2.imshow("Faces found" ,image)

#cv2.imwrite('./images/test_faces.png',image)

cv2.waitKey(0)
cv2.destroyAllWindows()


