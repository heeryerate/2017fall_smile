import glob
from shutil import copyfile # copy file from one to another one
from shutil import move
import os
emotions = ["Happy", "Sadness", "Surprise", "Anger", "Disgust", "Fear"]
participants = glob.glob("./JapanPic/*") 

with open('Japan60','r') as f:        
	row = f.read()
		
for sessions in participants:
	sss = sessions;
	ss = sessions.split(".")
	#print("ss[2] = %s"%(ss[3]))
	
	for data in row.split("\n"):
		Data = data.split(" ")
		
			
		
		if ss[3] == Data[0]:

			maxIndex = (Data[1:-1].index(max(Data[1:-1])))
			print (maxIndex)
			copyfile(sessions,"./%s/%s"%(emotions[maxIndex],ss[3]+'.jpg'))
			
		
