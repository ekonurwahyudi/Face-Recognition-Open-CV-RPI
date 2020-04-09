# Import OpenCV2 for image processing
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import sys
import time
import numpy as np
# Import Sound
import pygame

pygame.init()
face_name = {
	1: "Munzir",
	2: "Muhajirin",
	3: "Andi",
	4: "Efpri",
	5: "Eko",
}

face_sound = {
	1: "suara/eko.wav",
	2: "suara/eko.wav",
	3: "suara/eko.wav",
	4: "suara/eko.wav",
	5: "suara/eko.wav",
}

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Load the trained mode
recognizer.read('trainer/trainer.yml')
# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"
# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);
# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX
#cap = cv2.VideoCapture(0)
print("Starting camera..")
camera = PiCamera()
camera.resolution = (512, 512)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(512, 512))
print("camera started")
unknown = 0
known = 0
count = 0
# Loop
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	image = cv2.flip(image, +1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
	# Draw a rectangle around the faces 
	for (x,y,w,h) in faces:
		cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
	        # Recognize the face belongs to which ID
		name = ""
	        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
	        # Check the ID if exist
		start = time.time()
	        if id in face_name and conf<60:
		    name = face_name[id]
		    conf = "{0}%".format(round(100-conf))
		    print "Nama :", name
		    pygame.mixer.Sound(face_sound[id]).play()
		    print "Putar suara", name
		    #time.sleep(1)
		    known += 1
	        #If not exist, then it is Unknown
	        else:
		    name = "Orang Tidak Dikenal"
		    conf = " {0}%".format(round(100-conf))
	            print(name)
	 	    unknown += 1
        	cv2.putText(image, str(name), (x+5,y-5), font, 1, (255,255,255),2)
	        cv2.putText(image, str(conf), (x+5,y+h-5), font, 1, (255,255,255),2)
		end = time.time()
		print("waktu : {0} Seconds".format(end-start))
		print("confidence : {0}".format(conf))
		count += 1
		print "Frame:", count,"\n"	
   	cv2.imshow("Faces", image)
    	#key = cv2.waitKey(1) & 0xFF
	rawCapture.truncate(0)	
	#if key == ord("q"):
       		#break
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
	elif count>=100:
        	break
print "-------------------------------------------"
print "Known :", known,"peoples"
print "Unknown :", unknown,"peoples"
print ("Accuracy : {0}%".format(known))
cv2.destroyAllWindows()
