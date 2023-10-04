import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM) #BCM - Broadcom SOC Channel numbering | BOARD - Number on Raspberry Pi

IR_PIN = 3 # GPIO.BOARD -- PIN = 5

GPIO.setup(IR_PIN, GPIO.IN) # input pin set 

while True:
    ir_state = GPIO.input(IR_PIN)

    if ir_state == GPIO.LOW:
        print("YAAAYYYY!!!")
    else:
        print("No object")
    time.sleep(0.1)


