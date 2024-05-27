import cv2
import numpy as np
import pytesseract
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
IR_PIN = 3
GPIO.setup(IR_PIN, GPIO.IN)

cap = cv2.VideoCapture(0)

lower_color = np.array([-128, 58, 66])
upper_color = np.array([128, 78, 106]) 

prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0 
frame_count = 0
frame_buffer = 10

fps = 5

timer_started = False
start_time = 0
total_fare = 0
elapsed_time = 0
   
while True:
    ir_state = GPIO.input(IR_PIN)   
    
    if ir_state == GPIO.LOW:
        print("YAYYYYYYYY")
        ret, frame = cap.read()
        raw_image = frame.copy()

        if not timer_started:
            start_time = time.time()
            timer_started = True 
 
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame = cv2.GaussianBlur(hsv_frame, (11, 11), 0)
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        edges = cv2.Canny(mask, threshold1 = 30, threshold2 = 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            merged_contour = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(merged_contour)

            if frame_count < frame_buffer:
                x = (x+prev_x) // 2
                y = (y+prev_y) // 2
                w = (w+prev_w) // 2
                h = (h+prev_h) // 2 
            
                frame_count += 1
            
            prev_x, prev_y, prev_w, prev_h = x, y, w, h
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_frame, lang='eng', config = '--psm 6 --oem 3')
    
        frame_height, frame_width, _ = frame.shape
        timer_text = f'Time: {elapsed_time:.2f}s'
        text_position = (10, 50)
        timer_position = (frame.shape[1] - 220, frame.shape[0] - 20)
        
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, timer_text, timer_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Raw Image Captured', raw_image) 
        cv2.imshow('Object Detection and Text Detection', frame)
        

        if timer_started:
            elapsed_time = time.time() - start_time
            total_fare  = elapsed_time * 8.33  #cost per second(1/12 ----- 10min = 50 rupees)
            Payment_Amount = total_fare * 0.01

            print(f'Time elapsed: {elapsed_time:.2f} seconds')
            print(f'Total Fare: {total_fare:.2f}paisa')
            print(f'Payment Amount: Rs. {Payment_Amount:.2f}')
            # timer_started = False
        
        fare_display = np.zeros((100, 400, 3), dtype=np.uint8)
        fare_text = f'Total Fare: Rs.{Payment_Amount:.2f}'
        cv2.putText(fare_display, fare_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Total Fare', fare_display)
        
        if not ret:
            break
        if ret:
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])

            lower_green = np.array([40, 40, 40]) 
            upper_green = np.array([80, 255, 255])

            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([140, 255, 255])
    
            lower_tan = np.array([0, 34, 62])
            upper_tan = np.array([68, 54, 102])
    
            lower_beige = np.array([30, 46, 76])
            upper_beige = np.array([90, 66, 116])
            
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])

            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 30])

            lower_silver = np.array([0, 0, 150])
            upper_silver = np.array([180, 20, 200])

            lower_gray = np.array([0, 0, 100])
            upper_gray = np.array([180, 20, 150])

            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])

            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([20, 255, 255])

            mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
            mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
            mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
            mask_tan = cv2.inRange(hsv_frame, lower_tan, upper_tan)
            mask_beige = cv2.inRange(hsv_frame, lower_beige, upper_beige)
            mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)
            mask_black = cv2.inRange(hsv_frame, lower_black, upper_black)
            mask_silver = cv2.inRange(hsv_frame, lower_silver, upper_silver)
            mask_gray = cv2.inRange(hsv_frame, lower_gray, upper_gray)
            mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv_frame, lower_orange, upper_orange)


            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_tan, _ = cv2.findContours(mask_tan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_beige, _ = cv2.findContours(mask_beige, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_silver, _ = cv2.findContours(mask_silver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_gray, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            


            area_red = sum([cv2.contourArea(c) for c in contours_red])
            area_green = sum([cv2.contourArea(c) for c in contours_green])
            area_blue = sum([cv2.contourArea(c) for c in contours_blue])
            area_tan = sum([cv2.contourArea(c) for c in contours_tan])
            area_beige = sum([cv2.contourArea(c) for c in contours_beige])
            area_white = sum([cv2.contourArea(c) for c in contours_white])
            area_black = sum([cv2.contourArea(c) for c in contours_black])
            area_silver = sum([cv2.contourArea(c) for c in contours_silver])
            area_gray = sum([cv2.contourArea(c) for c in contours_gray])
            area_yellow = sum([cv2.contourArea(c) for c in contours_yellow])
            area_orange = sum([cv2.contourArea(c) for c in contours_orange])

            for contour in contours_red:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color

            for contour in contours_green:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color

            for contour in contours_blue:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color

            for contour in contours_tan:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (210, 180, 140), 2)  # Tan color
            
            for contour in contours_beige:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 245, 220), 2)  # Beige color
            
            for contour in contours_white:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  #White color
                
            for contour in contours_black:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
            
            for contour in contours_silver:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (192, 192, 192), 2)
            
            for contour in contours_gray:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
                
            for contour in contours_yellow:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                
            for contour in contours_orange:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
            
            dominant_color = None

            if area_red > area_green and area_red > area_blue and area_red > area_tan and area_red > area_beige and area_red > area_white and area_red >
            area_black and area_red > area_silver and area_red > area_gray and area_red > area_yellow and area_red > area_orange:
                dominant_color = "Red"
            elif area_green > area_red and area_green > area_blue and area_green > area_tan and area_green > area_beige and area_green > area_white and area_green >
            area_black and area_green > area_silver and area_green > area_gray and area_green > area_yellow and area_green > area_orange:
                dominant_color = "Green"
            elif area_blue > area_red and area_blue > area_green and area_blue > area_tan and area_blue > area_beige:
                dominant_color = "Blue"
            elif area_tan > area_red and area_tan > area_green and area_tan > area_blue and area_tan > area_beige:
                dominant_color = "Tan"
            elif area_beige > area_red and area_beige > area_green and area_beige > area_blue and area_beige > area_tan:
                dominant_color = "Beige"
            elif area_white > area_red and area_beige > area_green and area_beige > area_blue and area_beige > area_tan:
                dominant_color = "White"
            elif area_black > area_red and area_beige > area_green and area_beige > area_blue and area_beige > area_tan:
                dominant_color = "Black"
            elif area_silver > area_red and area_beige > area_green and area_beige > area_blue and area_beige > area_tan:
                dominant_color = "Silver"
            elif area_gray > area_red and area_beige > area_green and area_beige > area_blue and area_beige > area_tan:
                dominant_color = "Gray"
            elif area_yellow > area_red and area_beige > area_green and area_beige > area_blue and area_beige > area_tan:
                dominant_color = "Yellow"
            elif area_orange > area_red and area_beige > area_green and area_beige > area_blue and area_beige > area_tan:
                dominant_color = "Orange"

        color_text = f'Dominant Color: {dominant_color}'
        cv2.putText(frame, color_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Color Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('g'):
            break


cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()



