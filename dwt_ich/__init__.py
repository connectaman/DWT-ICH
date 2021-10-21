import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pywt
import pywt.data


def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
    plt.savefig('haar_dwt1.jpg')
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
    plt.savefig('haar_dwt2.jpg')
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
    plt.savefig('haar_dwt4.jpg',bbox_inches='tight',pad_inches = 0)
  elif ctype=='rgb':
    plt.imshow(img)
    plt.savefig('haar_dwt3.jpg')
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()



class ICH:

    def predict(self,img_path):
        """
        This function accpets image path and applies DWT and various Image Transformation to Segment the Brain Hemorrhage region from Brain CT Scan Image
        
        Input:
        img_path: Path of Image (Brain CT Image)

        Return:
        res: Resultance Image with hemorrhage Segmentation region
        
        """
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #ShowImage('Brain MRI',gray,'gray')

        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
        #ShowImage('Thresholding image',thresh,'gray')

        ret, markers = cv2.connectedComponents(thresh)

        #Get the area taken by each component. Ignore label 0 since this is the background.
        marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
        #Get label of largest component by area
        largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
        #Get pixels which correspond to the brain
        brain_mask = markers==largest_component

        brain_out = img.copy()
        #In a copy of the original image, clear those pixels that don't correspond to the brain
        brain_out[brain_mask==False] = (0,0,0)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        coeffs2 = pywt.dwt2(gray,'haar')
        LL, (LH, HL, HH) = coeffs2
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1


        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,0,0]

        res = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
        ret, thresh = cv2.threshold(res,160,255,cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(thresh,cv2.COLOR_BGR2HSV)
        lower_range = np.array([120,255,255])
        upper_range = np.array([130,255,255])
        mask_r = cv2.inRange(hsv,lower_range,upper_range)
        r,g,b = cv2.split(thresh)
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res, contours, -1, (255,255,0), 3)

        return res
    
