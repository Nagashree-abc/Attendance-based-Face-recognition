import cv2 #which is used for the video capturing
import random # we required random number to store the images which we have captured
cam=cv2.VideoCapture(0) # we need to create an object called as cam through the VideoCapture class available in the library of cv2 0 represents the empty string.
cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# then we need to create cascade object by importing cv2 
# it is to detect the objects in a video stream
#haarcascade is an algorithm which is to detect the frontal face.It works to train detect the images both negative as well as positive images which are superimposed.



while True: # while loop has taken for continuously capturing of the faces
    flag,image=cam.read() # to read the images/camera that have captured by the 
    #web cam we use read function.Here flag is a common variable which takes the values of boolean
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=cascade.detectMultiScale(gray_image,1.1,5) #detectmultiscale is a method that returns the list of rectangles
    #(Scalefactors is to specify how much the image size is reduced at each image scale,optical neighbours)
    if len(faces)>0:

        x,y,w,h=faces[0] # here x,y,w,h are the face locations
        # here faces[0] stores the first detected face.
    for x,y,w,h in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,100,0),2)
                     #(image,start point,end point,color,thickness)

    print(faces)
    
    cv2.imshow("Myimage",image) #image is showed in the other page
    #imshow function is used to show the image
    #This image has been shown in the my image window
    k=cv2.waitKey(2) 
    # waitkey keeps the window active.
     #this allow the user to display the window for some milliseconds here 2 is nothing but delay 
    # here window is opened until we press some key

    if k==ord('q'): #ASCII value of q
        break
    if k==ord('s'):
        cropped=image[y:y+h,x:x+w] # images has been stored in the cropped
        cropped=cv2.resize(cropped,(300,300))
        index=random.randint(1,100) # returns a random integer value b/w the lower and higher limits
        path=f"./dataset/0/person{index}.jpg"
        cv2.imwrite(path,cropped) # here image has been wriitenthat means path and cropped images has been written

#cv2,imshow("My_Gray_image",grey_image)
#cv2.waitKey()


cam.release() #to release the resource.