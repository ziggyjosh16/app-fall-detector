import cv2 as cv
from time import sleep

def get_frame(cap):
    if cap.grab():
        retval, image = cap.retrieve()
        if retval:
            return image

cap = cv.VideoCapture(0)
sleep(2)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
body_cascade = cv.CascadeClassifier('haarcascade_upperbody.xml')

while(cv.waitKey(1) & 0xFF != ord('q')):
    image = get_frame(cap)
    faces = face_cascade.detectMultiScale(image, 1.5, 3)
    bodies = body_cascade.detectMultiScale(image, 1.1, 5)
    for x,y,w,h in faces:    
        cv.rectangle(image,(x,y),(x+w,y+h),(14,201,255),2)
    for x,y,w,h in bodies:
        cv.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
    cv.imshow('test_stream', image)
cap.release()
cv.destroyAllWindows()

