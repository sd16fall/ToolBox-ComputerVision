""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/seed/Toolbox/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
# Creates a numpy matrix to control the degree of blurring
kernel = np.ones((21,21),'uint8')
font = cv2.FONT_HERSHEY_DUPLEX

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Runs Viola-Jones face detection cascades to get a list of faces
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))

    for (x,y,w,h) in faces:
        frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
        #x_right_top = int((x+w)/4 + 15)
        #y_right_top = int((y+h)/4 - 15)
        #x_left_top = int((x+w/4) + 25)
        #y_left_top = int((y+h/4) - 15)

        # Draws eyeballz
        cv2.circle(frame[y:y+h,x:x+w,:],(55,70),15,(255,255,255),-1)
        cv2.circle(frame[y:y+h,x:x+w,:],(135,70),15,(255,255,255),-1)
        cv2.circle(frame[y:y+h,x:x+w,:],(55,72),7,(0,0,0),-1)
        cv2.circle(frame[y:y+h,x:x+w,:],(135,72),7,(0,0,0),-1)

        # Draw sad face
        cv2.ellipse(frame[y:y+h,x:x+w,:],(95,140),(50,25),0,180,360,(255,0,255),3)

        # Special function
        cv2.putText(frame[y:y+h,x:x+w,:],'#2020',(55,35),font,1,(255,255,255),1)


    # Display the frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Outside of the loop
cap.release()
cv2.destroyAllWindows()
