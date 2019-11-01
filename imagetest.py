import cv2 as cv
import numpy as np
import os
from imutils.object_detection import non_max_suppression

for img in os.scandir('group_photo'):
	image = cv.imread(img.path)
	supimage = image.copy()

	face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
	hog = cv.HOGDescriptor()
	hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
	# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

	bodies, weights = hog.detectMultiScale(image, winStride=(4,4), padding=(6,6), scale=1.05)
	supbodies = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bodies])
	supbodies = non_max_suppression(supbodies, overlapThresh=0.65)


	print('{} bodies found.'.format(len(bodies)))
	for x,y,w,h in bodies:
			cv.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
	for x,y,w,h in supbodies:
			cv.rectangle(supimage, (x,y), (x+w,y+h), (0,255,0), 2)
	if len(bodies) > 0 and len(bodies) > 0:
		cv.imshow('no suppression', image)
		cv.imshow('suppression', supimage)
		supimage = image.copy()

	cv.waitKey(0)