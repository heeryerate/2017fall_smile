import glob
from shutil import copyfile # copy file from one to another one
from shutil import move
import cv2
import sys
import os
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
participants = glob.glob("./sorted_set/*") 
#print(participants)

for x in participants:
    #print("%s"%x)

    for sessions in glob.glob("%s/*"%x):

        part = sessions[-4:]
        #print(part)        
        
        image = cv2.imread(sessions)
        #print (sessions)
        print(image.shape)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        
        #for i, face in enumerate(image_faces):
        #   cv2.imwrite("face-" + str(i) + ".jpg", face)

        for (x,y,w,h) in faces:
            image = image[y: y+256 , x: x+256]
            cv2.imwrite(sessions+".jpg",image)
            os.remove(sessions)
            os.rename(sessions+".jpg",sessions)    
            #copyfile(image, "./sorted_set/%s/%s"%(emotions[emotion], part))    


           
          
		

