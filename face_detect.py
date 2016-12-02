""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml') #file that finds faces :)
kernel = np.ones((50,50),'uint8')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() #gets the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20)) ##creates rectangle paramaters
    for (x,y,w,h) in faces:
    	frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    	cv2.circle(frame,(x+60,y+60), 20, (0,0,255), -1) #red eye (filled in)
    	cv2.circle(frame,(x+120,y+60), 20, (0,0,255), -1) #red eye 2
    	cv2.circle(frame,(x+60,y+60), 5, (0,0,0), -1) #single pupil
    	cv2.rectangle(frame,(x+80,y+90),(x+100,y+110),(255,0,0),-1) #nose! :-D
    	cv2.ellipse(frame,(x+90,y+150),(30,50),0,0,180,255,-1) #mouth



    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
