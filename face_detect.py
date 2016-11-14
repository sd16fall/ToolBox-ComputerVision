import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('/home/andrewholmes/haarcascade_frontalface_alt.xml')

kernel = np.ones((41,41),'uint8')

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
		cv2.circle(frame, (x+w/3,y+h/3), 30, (255,255,255), -1)
		cv2.circle(frame, (x+2*w/3,y+h/3), 30, (255,255,255), -1)
		cv2.circle(frame, (x+w/3,y+h/3+15), 15, (0,0,0), -1)
		cv2.circle(frame, (x+2*w/3,y+h/3-8), 15, (0,0,0), -1)
		cv2.ellipse(frame, (x+w/2,y+2*h/3,), (50,70),0,0,180,(0,0,0),-1)
		cv2.ellipse(frame, (x+w/2,y+2*h/3,), (50,70),0,0,180,(0,0,0),-1)
		cv2.ellipse(frame, (x+w/2,y+2*h/3,), (25,70),0,0,180,(180,105,255),-1)
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))

	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

