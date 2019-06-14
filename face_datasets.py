import cv2
import sys
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

#load the Haar face detection algorithm
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#cap = cv2.VideoCapture(0)
print("Starting camera..")
camera = PiCamera()
camera.resolution = (384, 240)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(384, 240))
#time.sleep(0.5)
print("camera started")

# For each person, one face id
face_id = 1
count = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image_frame = frame.array
	gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

	# Draw a rectangle around the faces
	rawCapture.truncate(0)
	for (x, y, w, h) in faces:
		cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
		if len(faces) > 0 :
			print("found %d face at x:%d and y:%d, frame: %d size %d x %d" % (len(faces), x, y, count,  w, h))
	count += 1
	# Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray)
	# Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
	elif count>200:
        	break

#cap.release()
cv2.destroyAllWindows()

def detect_face(image, cascade):
	image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(image_grayscale, scaleFactor=1.2, minNeighbors=5)
	return faces
