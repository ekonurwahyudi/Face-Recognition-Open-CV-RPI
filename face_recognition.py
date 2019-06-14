# Import OpenCV2 for image processing
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import sys
import time
import numpy as np
import pygame

pygame.init()
eko = pygame.mixer.Sound("suara/eko.wav")
google = pygame.mixer.Sound("suara/google.wav")

# Import numpy for matrices calculations


# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_alt.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

#cap = cv2.VideoCapture(0)
print("Starting camera..")
camera = PiCamera()
camera.resolution = (384, 240)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(384, 240))
#time.sleep(0.1)
print("camera started")

# Loop
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	#capture image and convert to grayscale
	#print("processing frame :%d" % frame_count)
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

	# Draw a rectangle around the faces
	for (x,y,w,h) in faces:
		cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
	        # Recognize the face belongs to which ID
	        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
	        # Check the ID if exist 
	        if (id == 1):
		    print(id)
		    print(eko.play())
	            id = "Eko"
	        elif (id == 2):
		    print(id)
		    print(google.play())
	            id = "Itok"
	        #If not exist, then it is Unknown
	        else:
	            print(id)
		    id = "Ngak Kenal"
	
	        # Put text describe who is in the picture
        	cv2.rectangle(image, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        	cv2.putText(image, str(id), (x,y-40), font, 2, (255,255,255), 3)

   	cv2.imshow("Faces found", image)
    	key = cv2.waitKey(1) & 0xFF
	rawCapture.truncate(0)	
	if key == ord("q"):
       		break

#cap.release()
cv2.destroyAllWindows()
