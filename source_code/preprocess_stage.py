#preprocess_stage program 
#The program only takes input image
#To see next figure, close current figure window.
#Copyright 2019, Yeonsu Kim, All Rights Reserved

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Read Image
img_colon = cv.imread('Artificial_colon.png')
img_real = cv.imread('Real_Colon.jpg')

#Conversion to gray-scale
image_colon = cv.cvtColor(img_colon, cv.COLOR_BGR2GRAY)
img_real = cv.cvtColor(img_real, cv.COLOR_BGR2GRAY)

image_colon = cv.resize(image_colon, (300,300))
img_real = cv.resize(img_real,(300,300))

#Gaussian Filtering on artificial colon image
GBlur = cv.GaussianBlur(image_colon,(7,7),1)
GBlur2 = cv.GaussianBlur(image_colon,(13,13),2)
GBlur3 = cv.GaussianBlur(image_colon,(19,19),3)
GBlur4 = cv.GaussianBlur(image_colon,(25,25),4)

#Plot Result
plt.figure(1)
plt.suptitle('Guassian Filtered result: Artificial Colon Image')
plt.subplot(221),plt.imshow(GBlur,'gray'),plt.title('standard deviation=1'),plt.axis('off')
plt.subplot(222),plt.imshow(GBlur2,'gray'),plt.title('standard deviation=2'),plt.axis('off')
plt.subplot(223),plt.imshow(GBlur3,'gray'),plt.title('standard deviation=3'),plt.axis('off')
plt.subplot(224),plt.imshow(GBlur4,'gray'),plt.title('standard deviation=4'),plt.axis('off')
plt.show()


#Gaussian Filtering on real colon image
GBlur_real = cv.GaussianBlur(img_real,(7,7),1)
GBlur2_real = cv.GaussianBlur(img_real,(13,13),2)
GBlur3_real = cv.GaussianBlur(img_real,(19,19),3)
GBlur4_real = cv.GaussianBlur(img_real,(25,25),4)

#Plot Result
plt.figure(2)
plt.suptitle('Guassian Filtered result: Real Colon Image')
plt.subplot(221),plt.imshow(GBlur_real,'gray'),plt.title('standard deviation=1'),plt.axis('off')
plt.subplot(222),plt.imshow(GBlur2_real,'gray'),plt.title('standard deviation=2'),plt.axis('off')
plt.subplot(223),plt.imshow(GBlur3_real,'gray'),plt.title('standard deviation=3'),plt.axis('off')
plt.subplot(224),plt.imshow(GBlur4_real,'gray'),plt.title('standard deviation=4'),plt.axis('off')
plt.show()

#Median Filtering on both artifical and real colong image
mBlur_colon = cv.medianBlur(image_colon,19)
mBlur_real = cv.medianBlur(img_real,19)

#plot result
plt.figure(3)
plt.suptitle('Median Filtered Result on real and artificial colon')
plt.subplot(121),plt.imshow(mBlur_colon,'gray'),plt.title('artificial colon'),plt.axis('off')
plt.subplot(122),plt.imshow(mBlur_real,'gray'),plt.title('real colon'),plt.axis('off')
plt.show()

