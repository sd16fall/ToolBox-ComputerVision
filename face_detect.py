""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
img = cv2.imread("img.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.3,5)


kernel = np.ones((50,50),"uint8")

for (x,y,w,h) in faces:
    print x
    print y
    print w
    print h
    img[y:y+h,x:x+w] = cv2.dilate(img[y:y+h,x:x+w],kernel)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
    cv2.circle(img,(x+w/3,y+h/3),(w+h)/20,(0,0,255),-1,8)
    cv2.circle(img,(int(4*(x+w)/5),y+h/3),(w+h)/20,(0,0,255),-1,8)
    cv2.circle(img,((x+(x+w)/2),int(4*(y+h)/5)),(h+w)/15,(255,0,0),-1,8)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
