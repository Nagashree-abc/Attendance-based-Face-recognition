import cv2 # open cv for videocapturing
import numpy as np
from apidemo import savestudent

recognizer = cv2.face.LBPHFaceRecognizer_create()
cam=cv2.VideoCapture(0)

recognizer.read('facemodel.yml')
cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#we are reading trained model
d={0:"Nagashree",1:"Anusha",2:"Hemanth"}

while True:
    
    flag,image=cam.read()
    
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    faces=cascade.detectMultiScale(gray,1.1,5)
    
    if len(faces) > 0:
        
        x,y,w,h=faces[0]
        
        cropped=gray[y:y+h,x:x+w]
        cropped=cv2.resize(cropped,(300,300))
        
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        

        label,confidence=recognizer.predict(cropped)

        print(f"Label : {label}, Confidence = {confidence}")
        
        if confidence <60 :
            name=d[label]
            
            savestudent(name)
        else:
            name="Unknown"
            
        
        org = (50, 50)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 255)  # White
        thickness = 2
        lineType = cv2.LINE_AA # is the anti-aliased line

        cv2.putText(image, name, org, fontFace, fontScale, color, thickness, lineType)

        cv2.imshow("testing",image)
        k=cv2.waitKey(1)
         
        if k==ord('q'):
            break