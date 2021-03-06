#Visual Guidance System for mesh-worm robot 
#Version Final_AB
#Copyright 2019, Yeonsu Kim, All Rights Reserved

import cv2 as cv
import numpy as np

# Load and Read input

cap = cv.VideoCapture('Test.mp4')

#Create array to store frames causing out of index error
non_contour_frame = [] 

#Array to store orientation of each frame
orient_ation = []

count = 0

while True:
    
    #Read input for current frame
    ret1,current_frame = cap.read()
        
    #Print error message if there is no input
    if ret1 == False:
        print('There is no valid input')
        break
        
    #Gray-scale conversion of current input
    current_frame_gray = cv.cvtColor(current_frame,cv.COLOR_BGR2GRAY)
    



    #Comment out median to simulate final version A  
    #Comment out Gaussian to simulate final version B    
    #current_frame_fil = cv.GaussianBlur(current_frame_gray,(25,25),4)
    current_frame_fil = cv.medianBlur(current_frame_gray,19)
        
    #Thresholding 
    ret4,current_frame_thresh = cv.threshold(current_frame_fil,15,255,cv.THRESH_BINARY_INV)
        
    #Contour Detection
    im2,current_frame_cont,hierarchy = cv.findContours(current_frame_thresh,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        
    #Creat array to store detected contours in descending order by their area
    cnt_area_current = sorted(current_frame_cont, key = cv.contourArea, reverse = True) #Sort Area of contour in descending order
        
    try:
            
        #draw contour to original input
        cv.drawContours(current_frame,cnt_area_current[0],-1,(255,0,0),3)
            
        #Moments computation
        M_current = cv.moments(cnt_area_current[0]) #Calculates moments of larget contour area
            
        cx_current = int(M_current['m10']/M_current['m00']) #CENTER IN X-AXIS
        cy_current = int(M_current['m01']/M_current['m00']) #CENTER IN Y-AXIS
            
        #Draw center of contour on original input
        cv.circle(current_frame,(cx_current,cy_current),7,(255,0,0),-1)
            
        #Draw arrow from center of frame to contour center of dark region
        cv.arrowedLine(current_frame,(640,650),(cx_current,cy_current),(0,255,0),10)
            
            
        #Index region for direction
        left_index = int((4*current_frame.shape[1])/10)
        right_index = int((6*current_frame.shape[1])/10)
            
        if cx_current <= left_index:
            
            cv.putText(current_frame,'Move Left',(420,100),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
            
        elif cx_current >= right_index:
            
            cv.putText(current_frame,'Move Right',(420,100),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
            
        else:
            
            cv.putText(current_frame,'Move Forward',(420,100),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
                
                
        #Computes rotatable rectangle that fits to contour of dark region
        min_rec = cv.minAreaRect(cnt_area_current[0])
        orient_ation.append(min_rec[2])
        rotation = abs(orient_ation[count]-orient_ation[count+1])
        count = count+1
        print('change in orientation of contour: %r'%rotation)
            
        #Computes corner points for rotatable rectangle to draw it on orginal image
        box = cv.boxPoints(min_rec)
        box = np.int0(box)
            
        #Draw rotatable rectange to original image
        cv.drawContours(current_frame,[box],0,(0,0,255),2)
            
            
        #Decision of large orientation
        if rotation >= 80 or rotation <= -80:
            print('Too much rotation')
            i=0;
            cv.imwrite('fault_%i.jpg',current_frame)
            i=i+1
                
        #produce output
        cv.imshow('procedure',current_frame)
        cv.imshow('threshold',current_frame_thresh)
           
        #Pass frame where no contour is detected
    except IndexError:
            
        #Add frame to array if it cause out of index error due to small contours
        no_contour = cap.get(cv.CAP_PROP_POS_FRAMES) 
        non_contour_frame.append(no_contour)
        continue
    
    #Press ESC to exit
    if cv.waitKey(30) & 0xFF == 27:
        break
            
#Frame length of the video
length = cap.get(cv.CAP_PROP_FRAME_COUNT)

#Number of frames where out of index error occur
length_none = len(non_contour_frame) 

#Calculate error rate of dark region detection
error_rate = ((length-length_none)/length)*100
print(error_rate)
    
cap.release()
cv.destroyAllWindows()


        


