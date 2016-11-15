""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((50,50),'uint8')


#keep running
while(True):
    # Capture each frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
    for (x,y,w,h) in faces:
        #Blur face
        frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
        #draw rectangle around face
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        #draw mouth
        cv2.ellipse(frame,(x+w/2,y+3*h/4),(w/5,h/5),0,0,180,(0,0,255),-1)
        #draw eyes
        cv2.circle(frame,(x+w/4,y+h/3),w/10,(0,0,255),-1)
        cv2.circle(frame,(x+3*w/4,y+h/3),w/10,(0,0,255),-1)
                 

    # Display each frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
