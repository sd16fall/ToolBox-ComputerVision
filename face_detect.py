""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np


cap = cv2.VideoCapture('wayne.MP4')

face_cascade = cv2.CascadeClassifier('/home/wayne/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
kernel = np.ones((40,40), 'uint8')


while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.2, minSize = (20, 20))

    for (x,y,w,h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x: x+w, :], kernel)
        cv2.circle(frame,(x+w/3, y+h/3), (w/10), (255,255,255), -1)
	cv2.circle(frame,(x+w/3,y+h/3), (w/40), (0,0,0), -1)
        cv2.circle(frame,(x+2*w/3,y+h/3), (w/10), (255,255,255), -1)
        cv2.circle(frame,(x+2*w/3,y+h/3), (w/40), (0,0,0), -1)
	cv2.ellipse(frame, (int(x + w/2), int(y + 2*h/3)), (int(w/4), int(h/5)), 0, 0, 150, (0, 0, 255), 10)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#when everything done, release the capture
cap.release()
cv2.destroyAllWindows()
