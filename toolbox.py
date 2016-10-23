import numpy as np
import cv2

cap = cv2.VideoCapture('vtest2.MOV')
face_cascade = cv2.CascadeClassifier('/home/jonathan/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')


while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	kernel = np.ones((21,21),'uint8')

	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))

	for (x,y,w,h) in faces:
		cv2.ellipse(frame, (x+w/2,y+h/2), (int(.25*w), int(.25*h)), 0, 0, 180, (0,0,0), 2)


	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

