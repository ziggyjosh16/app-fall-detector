import cv2 as cv
import numpy as np
from imutils.object_detection import non_max_suppression
from time import sleep

def get_frame(cap):
    if cap.grab():
        retval, image = cap.retrieve()
        if retval:
            return image

cap = cv.VideoCapture('/home/ziggyjosh16/Projects/git/app-fall-detector/videos/FireHouse.mp4')
sleep(2)

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

while(cv.waitKey(1) & 0xFF != ord('q')):
    image = get_frame(cap)
    bodies, weights = hog.detectMultiScale(image, winStride=(4,4), padding=(6,6), scale=1.05)
    supbodies = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bodies])
    supbodies = non_max_suppression(supbodies, overlapThresh=0.65)
    for x,y,w,h in bodies:
        cv.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
    # for x,y,w,h in supbodies:
    #     cv.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    cv.imshow('test_stream', image)
cap.release()
cv.destroyAllWindows()