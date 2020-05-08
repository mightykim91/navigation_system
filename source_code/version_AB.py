#Visual Guidance System for mesh-worm robot 
#Version_A//B
#Copyright 2019, Yeonsu Kim, All Rights Reserved



#To run program save input video file to same directory.



import cv2 as cv
import numpy as np



# Load and Read input
# first_frame works as previous frame while computing orientation of the robot
cap = cv.VideoCapture('Test.mp4')

#Array to store orientation of each frame
orient_ation = []
count = 0
orient_ation.append(count)

while True:
    
    ret,frame = cap.read()
        
    #Print error message if there is no input
    if ret == False:
        print('There is no valid input')
        break
    
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    #frame_fil = cv.GaussianBlur(frame_gray,(25,25),4) #Comment out for version_B
    frame_fil = cv.medianBlur(frame_gray,19) #Comment out for version_A
        
    #Apply threshold
    ret4,frame_thresh = cv.threshold(frame_fil,15,255,cv.THRESH_BINARY_INV)

    #Detect Contours
    im2,frame_cont,hierarchy = cv.findContours(frame_thresh,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
    
    #Draw contour on input image
    cv.drawContours(frame,frame_cont,-1,(255,0,0),3)
        
    #Compute moments of contour
    M_frame = cv.moments(frame_cont[0])
        
    #ignore frame when 'm00'=0
    if M_frame['m00']==0:
            M_frame['m00']=1

    cx_frame = int(M_frame['m10']/M_frame['m00']) #CENTER IN X-AXIS
    cy_frame = int(M_frame['m01']/M_frame['m00']) #CENTER IN Y-AXIS
        
    #Draw contour center
    cv.circle(frame,(cx_frame,cy_frame),7,(255,0,0),-1)
        
    Draw arrow from center of frame to contour center
    cv.arrowedLine(frame,(640,650),(cx_frame,cy_frame),(0,255,0),10)
        
    #Index region for direction
    left_index = int((4*frame.shape[1])/10)
    right_index = int((6*frame.shape[1])/10)
        
        
    #Direction index
    if cx_frame <= left_index:
        
        cv.putText(frame,'Move Left',(420,100),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
        
    elif cx_frame >= right_index:
        
        cv.putText(frame,'Move Right',(420,100),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
        
    else:
        
        cv.putText(frame,'Move Forward',(420,100),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
            
    #Computes rotatable rectangle that fits to contour of dark region
    min_rec = cv.minAreaRect(frame_cont[0])
    orient_ation.append(min_rec[2])
    rotation = abs(orient_ation[count]-orient_ation[count+1])
    count = count+1
            
    #Computes corner points for rotatable rectangle to draw it on orginal image
    box = cv.boxPoints(min_rec)
    box = np.int0(box)
            
    #Draw rotatable rectange to original image
    cv.drawContours(frame,[box],0,(0,0,255),2)
            
            
    #Decision of large orientation
    if rotation >= 80 or rotation <= -80:
        print('Too much rotation')
        i=0;
        cv.imwrite('fault_%i.jpg',frame)
        i=i+1
    
    cv.imshow('procedure',frame)
    cv.imshow('threshold',frame_thresh)
        
       
    if cv.waitKey(20) & 0xFF == 27:
        break
        
cap.release()
cv.destroyAllWindows()


        


