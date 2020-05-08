#This 
#The program only takes input image
#To see next figure, close current figure window.
#Copyright 2019, Yeonsu Kim, All Rights Reserved


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#REAL COLON 
Real_Colon = cv.imread('Real_Colon.jpg')
Real_Colon = cv.resize(Real_Colon,(300,300))
Real_Colon_sub = cv.imread('Real_Colon.jpg')
Real_Colon_sub = cv.resize(Real_Colon_sub,(300,300))

#Convert to gray-scale
Real_col_gray = cv.cvtColor(Real_Colon,cv.COLOR_BGR2GRAY)


#Artificial colon
Art_Colon = cv.imread('Artificial_Colon.png')
Art_Colon = cv.resize(Art_Colon,(300,300))
Art_Colon_sub = cv.imread('Artificial_Colon.png')
Art_Colon_sub = cv.resize(Art_Colon_sub,(300,300))

#Convert to gray-scale
Art_col_gray = cv.cvtColor(Art_Colon,cv.COLOR_BGR2GRAY)

#Apply Filter
Real_Gaussian = cv.GaussianBlur(Real_col_gray,(25,25),4)
Real_Median = cv.medianBlur(Real_col_gray,19)
Art_Gaussian = cv.GaussianBlur(Art_col_gray,(25,25),4)
Art_Median = cv.medianBlur(Art_col_gray,19)

#Apply Thresholding to real colon image
ret1,real_thresh_gau = cv.threshold(Real_Gaussian,45,255,cv.THRESH_BINARY_INV)
ret2,real_thresh_med = cv.threshold(Real_Median,45,255,cv.THRESH_BINARY_INV)
ret3,real_otsu_gau = cv.threshold(Real_Gaussian,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret4,real_otsu_med = cv.threshold(Real_Median,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#Plot thresholding result
plt.figure(1)
plt.suptitle('Thresholding Result of Basic thresholding and Otsu')
plt.subplot(221),plt.imshow(real_thresh_gau,'gray'),plt.title('Basic Thresholding'),plt.axis('off')
plt.subplot(222),plt.imshow(real_otsu_gau,'gray'),plt.title('Otsu''s mehtod'),plt.axis('off')
plt.subplot(223),plt.imshow(real_thresh_med,'gray'),plt.title('Basic Thresholding'),plt.axis('off')
plt.subplot(224),plt.imshow(real_otsu_med,'gray'),plt.title('Otsu''s mehtod'),plt.axis('off')
plt.show()

#Thresholding on artificial colon image
ret5,art_thresh_gau = cv.threshold(Art_Gaussian,15,255,cv.THRESH_BINARY_INV)
ret6,art_thresh_med = cv.threshold(Art_Median,15,255,cv.THRESH_BINARY_INV)
ret7,art_otsu_gau = cv.threshold(Art_Gaussian,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret8,art_otsu_med = cv.threshold(Art_Median,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#Plot thresholding result
plt.figure(2)
plt.suptitle('Thresholding Result of Basic thresholding and Otsu on artificial colon image')
plt.subplot(221),plt.imshow(art_thresh_gau,'gray'),plt.title('Basic Thresholding'),plt.axis('off')
plt.subplot(222),plt.imshow(art_otsu_gau,'gray'),plt.title('Otsu''s mehtod'),plt.axis('off')
plt.subplot(223),plt.imshow(art_thresh_med,'gray'),plt.title('Basic Thresholding'),plt.axis('off')
plt.subplot(224),plt.imshow(art_otsu_med,'gray'),plt.title('Otsu''s mehtod'),plt.axis('off')
plt.show()

#Contour detection on real colon
im1,real_cont_gau,hierarchy1 = cv.findContours(real_thresh_gau,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
im2,real_cont_med,hierarchy2 = cv.findContours(real_thresh_med,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
cv.drawContours(Real_Colon,real_cont_gau,0,(255,0,0),3)
cv.drawContours(Real_Colon_sub,real_cont_med,0,(255,0,0),3)


#Plot Result
plt.figure(2)
plt.suptitle('Contour detection result of real colon')
plt.subplot(121),plt.imshow(cv.cvtColor(Real_Colon,cv.COLOR_BGR2RGB)),plt.axis('off'),plt.title('Gaussian Filtered')
plt.subplot(122),plt.imshow(cv.cvtColor(Real_Colon_sub,cv.COLOR_BGR2RGB)),plt.axis('off'),plt.title('Median Filtered')
plt.show()


#Contour detection on artificial colon
im3,art_contour_g,hierarchy3 = cv.findContours(art_thresh_gau,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
im4,art_contour_m,hierarchy4 = cv.findContours(art_thresh_med,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
cv.drawContours(Art_Colon,art_contour_g,0,(255,0,0),3)
cv.drawContours(Art_Colon_sub,art_contour_m,0,(255,0,0),3)

#Plot Result
plt.figure(3)
plt.suptitle('Contour detection result of artificial colon')
plt.subplot(121),plt.imshow(cv.cvtColor(Art_Colon,cv.COLOR_BGR2RGB)),plt.axis('off'),plt.title('Gaussian Filtered')
plt.subplot(122),plt.imshow(cv.cvtColor(Art_Colon_sub,cv.COLOR_BGR2RGB)),plt.axis('off'),plt.title('Median Filtered')
plt.show()


