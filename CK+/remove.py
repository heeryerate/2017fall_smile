import glob 
import glob 
import glob 
import os
from shutil import copyfile # copy file from one to another one
import cv2
cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
participants = glob.glob("./sorted_set/*") 

for x in participants: #x source_emotion/S005
	for sessions in glob.glob("%s/*" %x):
		os.remove(sessions)


