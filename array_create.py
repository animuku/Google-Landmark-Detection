import os
import cv2
import numpy as np

images=[]
categories=[]
with open('labels.txt','r') as ins:
    for line in ins:
        path,category=line.split(' ')
        img=cv2.imread(path)
        img=img[:,:,::-1]
        images.append(img)
        categories.append(category)
        #print(category)
image=np.array(images,dtype='uint32')

print(image.shape)
image=image/255.0
print(image)




