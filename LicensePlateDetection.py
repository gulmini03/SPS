import cv2
import numpy as np
import time
import pytesseract

lower_color = np.array([-128, 58, 66])
upper_color = np.array([128, 78, 106])  # HSV values (0-360 degrees)
# Capturing video initialized
cap = cv2.VideoCapture(0)

#reduce flickering
prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
frame_count = 0
frame_buffer = 10

#lower fps
fps = 5 

while True:
    start_time = time.time()
#reading a frame
    ret, frame = cap.read()
#converting the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#gaussian blur - noise reduction
    hsv_frame = cv2.GaussianBlur(hsv_frame, (11, 11), 0)
#creating a mask for the specified color range
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
#edges in the mask
    edges = cv2.Canny(mask, threshold1=30, threshold2=100)
#contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#bounding box that sufice all contours
    if len(contours) > 0:
#merge all the contours into one
        merged_contour = np.vstack(contours)
#compute the bounding box for the merged contour
        x, y, w, h = cv2.boundingRect(merged_contour)
         
        #reduce frames
        if frame_count < frame_buffer:
            x = (x+prev_x)//2
            y = (y+prev_y)//2
            w = (w+prev_w)//2
            h = (h+prev_h)//2
            
            frame_count += 1
        prev_x, prev_y, prev_w, prev_h = x, y, w, h
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#convert the frame to grayscale for text detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#text detection
    text = pytesseract.image_to_string(gray_frame, lang='eng', config='--psm 6 --oem 3')

#text display
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#bounding box
    cv2.imshow('Object Detection and Text Detection', frame)

#time taken for frame processing
    frame_processing_time = time.time() - start_time

#sleep time
    sleep_time = 1.0 / fps - frame_processing_time

    if sleep_time > 0:
        time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

