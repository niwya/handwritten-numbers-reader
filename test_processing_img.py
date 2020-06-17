import sys
import cv2
import numpy as np

path = r'C:\Users\Chloe\Documents\GitRepositories\Handwritten_num_reader\image.jpeg'
givenWindow = 'givenImage_Win'
processedWindow = 'resizedImage_Win'

givenImage=cv2.imread(path,0)
resizedImage=cv2.resize(givenImage, (28,28), interpolation=cv2.INTER_NEAREST)
resizedImage=cv2.bitwise_not(resizedImage) #applies negative filter 
data=np.asarray(resizedImage)
formattedImage=[]
formattedImage.append(0x00000803)
formattedImage.append(1)
formattedImage.append(28)
formattedImage.append(28)
for i in range(28):
    for j in range(28):
        formattedImage.append(data[i][j])
print(formattedImage)
  
# Displaying the images
#cv2.imshow(givenWindow, givenImage)  
#cv2.imshow(processedWindow, resizedImage)
#cv2.waitKey(0)    
#cv2.destroyAllWindows() 
