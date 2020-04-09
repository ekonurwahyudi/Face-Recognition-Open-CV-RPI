import cv2
import sys
import time
# import camera buffer class
from picamera.array import PiRGBArray
# import camera class
from picamera import PiCamera

start =time.time()
#load the Haar face detection algorithm
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

print("Starting camera..")
camera = PiCamera()
camera.resolution = (512, 512)
camera.framerate = 24 # fps kenpa 24 karana mata manusia ngak bisa di atas 24
rawCapture = PiRGBArray(camera, size=(512, 512))
print("camera started")

# For each person, one face id
face_id = sys.argv[1]
count = 0
#proses code dibawah di setiap FPSNya
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image_frame = frame.array
	image_frame = cv2.flip(image_frame, +1) #balik kamera video jadi normal
	#ubah warna jadi grayscale
	gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

	# Draw a rectangle around the faces
	rawCapture.truncate(0)
	for (x, y, w, h) in faces:
		cv2.rectangle(image_frame, (x,y), (x+w,y+h), (0,255,0), 2)
		if len(faces) > 0 :
			count += 1
			print("found %d face at x:%d and y:%d, frame: %d size %d x %d" % (len(faces), x, y, count,  w, h))
		cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
	#rawCapture.truncate(0)
	# Save the captured image into the datasets folder
        
	# Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
	elif count>=50:
        	break
end =time.time()
runtime = end-start
print("Run Time : {0} Seconds".format(runtime))
cv2.destroyAllWindows()
