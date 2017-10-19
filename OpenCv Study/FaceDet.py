import cv2
import os


photopath = 'wb.jpg'
classifier = os.getcwd()+'/haarcascade_frontalface_default.xml'

#读取图片
image = cv2.imread(photopath)

#灰度转换
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#获取人脸识别训练数据
face_casacade = cv2.CascadeClassifier(classifier)

#探测人脸
faces = face_casacade.detectMultiScale(image)

# 方框的颜色和粗细
color = (0,0,255)
strokeWeight = 1
#弹出框名字
windowName = "Object Detection"

while True:  #为了防止
    #人脸个数
    print(len(faces))
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color, strokeWeight)

    #展示人脸识别效果
    cv2.imshow(windowName, image)

    #点击弹出的图片，按escape键，结束循环
    if cv2.waitKey(20) == 27:
        break

#循环结束后，退出程序。
exit()