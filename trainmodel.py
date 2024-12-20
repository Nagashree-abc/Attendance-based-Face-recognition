from os import listdir # os module provides functions to interact with the operating system listdir is used to get all the list of files specified in the directory.
# to iterate the folders
import cv2
import numpy as np # it is a library which works with arrays,linear algebra and matrices

root_dir="./dataset" # .-> is nothing but current directory

features=[]
labels=[]
l=0
for folder in listdir(root_dir): #here we iterating the folder=dataset
    #and we are taking the subfolders present in dataset folder
    print(f"-----files in the folder {folder} are  ---------")
    
    path=f"{root_dir}/{folder}"
    
    for file in listdir(path): # here we want a person image filein the folder present in 0 or 1 or 2
        filepath=f"{path}/{file}"
        
        image=cv2.imread(filepath,0) # in this images has been read that means filepath is nothing but here the images
        cv2.imshow("demo",image) # showing the images in the demo titled window
        cv2.waitKey()
        
        # print(image)
        features.append(image) # we are appending the image here that maeans we need to add the features to image.
        labels.append(l)
    l=l+1



print(f"features are...{features}")
print(f"Labels are {labels}")

print(f"{len(features)},{len(labels)}") # How many images have been present in the file path along with their features and labels

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(features,np.array(labels))

#Here we are training the machine using LBPH

recognizer.save("facemodel.yml")
# trained images are stored in the facemodel.yml file

#Above statement exports trained model to external file
