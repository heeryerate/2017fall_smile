import numpy as np
import cv2
import glob 
import os
cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

participants = glob.glob("./sorted_set/*") 

for x in participants: #x source_emotion/S005
	for sessions in glob.glob("%s/*" %x):
		image = cv2.imread(sessions)
		mirror_image = cv2.flip( image, 1)

		cv2.imwrite(sessions+"flip.jpg",mirror_image)
		os.remove(sessions)


