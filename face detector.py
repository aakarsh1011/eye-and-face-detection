#importing the libraries

import numpy as np
import cv2

#defining the cascades

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#capturing the video

cap = cv2.VideoCapture(0)
while True:
    #returning the frame
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        #converting it to a grayscale

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #identifying the faces

    #to draw rectangles
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #drawing the rect with color blue
        roi_gray = gray[y:y+h, x:x+w]       
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:                      #searching eyes in the face
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
