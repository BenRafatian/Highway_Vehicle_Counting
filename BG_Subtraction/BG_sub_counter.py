from ast import Break
import time
from turtle import width
import cv2
import numpy as np
from time import sleep

width_min = 30 # minimum width of the rectangle
height_min = 30 # minimum height of the rectangle

width_min_op = 15
height_min_op = 15

offset1 = 3  # Allowed error between pixels
offset2 = 3
offset_op = 0
delay = 30 

# line descriptions for the coming down line
liney_pos = 280
liney2_pos = 320
lane3_x = 220
lane2_x = lane3_x + 125
lane1_x = lane2_x + 180

# Regarding the camera position we are going to count 4 seperate lanes (3 coming towards the camera and 1 for the opposite direction)
carscrossedopposite = 0

detected_op_1 = []
detected_op_2 = []
counter_op_1 = 0
counter_op_2 = 0

carscrossed1 = 0
carscrossed2 = 0
carscrossed3 = 0

detected_1 = []
detected_2 = []
counter1 = [0, 0, 0]
counter2 = [0, 0, 0]


total_cars = 0

def center_counter(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('./Highway.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = int(width)
height = int(height)


# prepares the output video
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps , (width, height))

# opposite side rectangle coordinates:
o1x, o1y = 450, 115
ow, oh = (width - o1x), 30

frame_number = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    
    # temp = float(1 / delay) # changes the preview speed of the video
    # sleep(temp)

    if frame is None:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_number +=1
    loop_start_time = time.time()
    # background subtraction process    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 5)
    fgmask = fgbg.apply(blur)
    dilation = cv2.dilate(fgmask, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

    # contouring the objects in subtracted picture
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (lane3_x, liney_pos), (lane3_x, height), (255,200,200), 2)
    cv2.line(frame, (lane2_x, liney_pos), (lane2_x, height), (255,200,200), 2)
    cv2.line(frame, (lane1_x, liney_pos), (lane1_x, height), (255,200,200), 2)

    # opposite side counter
    cv2.rectangle(frame, (o1x, o1y), (o1x + ow, o1y + oh), (0,255,0), 1)
    for(i, c) in enumerate(contours):
        if hierarchy[0, i, 3] == -1:
            (x, y, w, h) = cv2.boundingRect(c)
            contour_size_validator = (w >= width_min_op) and (h >= height_min_op)
            if not contour_size_validator:
                continue
            cv2.rectangle(frame, (x,y), (w+x, h+y), (0,255,0), 2)
            center = center_counter(x, y, w, h)
            detected_op_1.append(center)
            detected_op_2.append(center)
            
            cv2.circle(frame, center, 3, (255,0,0), -1)

            for(x, y) in detected_op_1:
                if x >= o1x:
                    if (o1y - offset_op) <= y <= (o1y + offset_op):
                        counter_op_1 += 1

                        cv2.line(frame, (o1x, o1y), (width, o1y), (255, 0, 0), 1)    
                        detected_op_1.remove((x, y))         
                    
            for(x, y) in detected_op_2:
                if x >= o1x:
                    if (o1y + oh - offset_op) <= y <= (o1y + oh + offset_op):
                        counter_op_2 += 1
                        cv2.line(frame, (o1x, o1y + oh), (width, o1y + oh), (255, 0, 0), 1)      
                        detected_op_2.remove((x, y))         
                        

    cv2.line(frame, (0, liney_pos), (width, liney_pos), (255,127,0), 1)
    cv2.line(frame, (0, liney2_pos), (width, liney2_pos), (255,127,0), 1)
    for(i, c) in enumerate(contours):
        if hierarchy[0, i, 3] == -1:    
            (x, y, w, h) = cv2.boundingRect(c)
            contour_size_validator = (w >= width_min) and (h >= height_min)
            if not contour_size_validator:
                continue

            cv2.rectangle(frame, (x,y), (w+x, h+y), (0,255,0), 2)
            center = center_counter(x, y, w, h)
            detected_1.append(center)
            detected_2.append(center)

            cv2.circle(frame, center, 4, (255,0,0), -1)

            for(x, y) in detected_1:
                if y < (liney_pos + offset1) and y > (liney_pos - offset1):
                   
                    if x <lane3_x:
                        counter1[0] += 1
                        # print("count lane 3 : " + str(counter1[0]))
                    elif lane3_x <= x <=lane2_x:
                        counter1[1] += 1
                        # print("count lane 2 : " + str(counter1[1]))
                    else:
                        counter1[2] += 1 
                        # print("count lane 1 : " + str(counter1[2]))       
                   
                    cv2.line(frame, (0, liney_pos), (width, liney_pos), (0,127,255), 1)
                    detected_1.remove((x,y))
                    
            
            for(x, y) in detected_2:
                if y < (liney2_pos + offset2) and y > (liney2_pos - offset2):
                    

                    if x <lane3_x:
                        counter2[0] += 1
                        # print("lane 3 count : " + str(counter2[0]))
                    elif lane3_x <= x <=lane2_x:
                        counter2[1] += 1
                        # print("lane 2 count : " + str(counter2[1]))
                    else:
                        counter2[2] += 1 
                        # print("lane 1 count : " + str(counter2[2]))       
                   
                    cv2.line(frame, (0, liney2_pos), (width, liney2_pos), (0,127,255), 1)
                    detected_2.remove((x,y))




        carscrossedopposite = int((counter_op_1 + counter_op_2) // 2) 
        total_cars = int((sum(counter1) + sum(counter2))/2) + carscrossedopposite
        carscrossed1 = (counter1[2]+counter2[2])//2
        carscrossed2 = (counter1[1]+counter2[1])//2
        carscrossed3 = (counter1[0]+counter2[0])//2                




    temp_fps = int(1.0 / (time.time() - loop_start_time))
    cv2.putText(frame, "FPS: " + str(temp_fps), (0, height-5), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"Lane3: "+ str(carscrossed3) , (0, height-65), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"Lane2: "+ str(carscrossed2) , (0, height-45), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"Lane1: "+ str(carscrossed1) , (0, height-25), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"opposite: "+ str(carscrossedopposite) , (0, height-85), 0, 0.6, (200, 200, 0), 1)
    cv2.putText(frame,"Total Cars : "+ str(total_cars) , (0, height-105), 0, 0.6, (200, 200, 0), 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
    cv2.imshow("Video Original", frame)
    cv2.imshow("masked", closing)

    if cv2.waitKey(1) == 27:
        break
average_fps =  int(frame_number / (time.time() - start_time) ) 

print("Vehicles crossed Lane 1: " + str(carscrossed1))
print("Vehicles crossed Lane 2: " + str(carscrossed2))
print("Vehicles crossed Lane 3: " + str(carscrossed3))
print("Vehicles crossed opposite side: " + str(carscrossedopposite))
print("Total cars: " + str(total_cars))
print("Average FPS: " + str(average_fps))
cv2.destroyAllWindows()
out.release()
cap.release()