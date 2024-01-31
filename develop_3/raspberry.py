import time
from enum import Enum
import os

import time
from time import sleep
from robust_serial import Order, decode_order, read_order, write_i8, write_i16, write_order, read_i8
from robust_serial.utils import open_serial_port
import RPi.GPIO as GPIO
import requests
import json

GPIO.setmode(GPIO.BCM)
headers = {'Content-Type': 'application/json'}
url="http://192.168.1.226:8000/predict"

class DistanceSensor:
    def __init__(self, echo_pin, trigger_pin, name=""):
        self.echo_pin = echo_pin
        self.trigger_pin = trigger_pin
        self.name = name
        GPIO.setup(trigger_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)
        GPIO.output(trigger_pin, GPIO.LOW)
        print(f"Sensor {name} initialized")
        sleep(1)

    def read_distance(self):
        GPIO.output(self.trigger_pin, GPIO.HIGH)
        sleep(0.00001)
        GPIO.output(self.trigger_pin, GPIO.LOW)

        while GPIO.input(self.echo_pin) == 0:
            pulse_start_time = time.time()

        while GPIO.input(self.echo_pin) == 1:
            pulse_end_time = time.time()

        pulse_duration = pulse_end_time - pulse_start_time
        distance = round(pulse_duration * 17150, 2)
        return distance  
    
if __name__ == "__main__":
    # --------  CONSTANT DEFINITION --------
    motor_speed = 200
    max_motor_speed = 255
    action_timing = 0.5
    # -------- HCSR04 SENSORS DEFINITION --------
    hc_sr04_sensors = []
    hc_sr04_sensors.append(DistanceSensor(2, 3, "01"))
    hc_sr04_sensors.append(DistanceSensor(4, 17, "02"))   
    hc_sr04_sensors.append(DistanceSensor(27, 22, "03"))    
    hc_sr04_sensors.append(DistanceSensor(10, 9, "04"))    
    hc_sr04_sensors.append(DistanceSensor(11, 0, "05"))
    hc_sr04_sensors.append(DistanceSensor(5, 6, "06"))
    hc_sr04_sensors.append(DistanceSensor(13, 19, "07"))
    hc_sr04_sensors.append(DistanceSensor(14, 15, "08"))
    hc_sr04_sensors.append(DistanceSensor(18, 23, "09"))
    hc_sr04_sensors.append(DistanceSensor(24, 25, "10"))
    hc_sr04_sensors.append(DistanceSensor(8, 7, "11"))
    hc_sr04_sensors.append(DistanceSensor(1, 12, "12"))
    hc_sr04_sensors.append(DistanceSensor(16, 20, "13")) 
    count_hcsr04_sensors = len(hc_sr04_sensors) 

    try:
        serial_file = open_serial_port(baudrate=115200, timeout=None)
        serial_file.flush()
    except Exception as e:
        raise e

    is_connected = False
    # Initialize communication with Arduino
    while not is_connected:
        print("Waiting for arduino...")
        write_order(serial_file, Order.HELLO)
        bytes_array = bytearray(serial_file.read(1))
        if not bytes_array:
            time.sleep(2)
            continue
        byte = bytes_array[0]
        if byte in [Order.HELLO.value, Order.ALREADY_CONNECTED.value]:
            is_connected = True

    print("Connected to Arduino")
    try:
        while True:
            # procedi = input("Procedi? (y/n)")
            # if procedi == "y":
            # 1. read hcsr04 sensors
            distances = []
            for i in range(count_hcsr04_sensors):
                dist = hc_sr04_sensors[i].read_distance()
                # print(f"Sensor {hc_sr04_sensors[i].name} => {dist}")
                sleep(0.05)
                distances.append(dist)

            # 2. send order to read touch sensors left
            write_order(serial_file, Order.HELLO)
            read_order(serial_file)

            write_order(serial_file, Order.TOUCH_L)
            order_l = read_order(serial_file)
            # 3. read touch sensors left
            touch_l = read_i8(serial_file)

            # 4. send order to read touch sensors right
            write_order(serial_file, Order.TOUCH_R)
            order_r = read_order(serial_file)
            # 5. read touch sensors right
            touch_r = read_i8(serial_file)

            # 6. send object to server
            write_order(serial_file, Order.MOTOR)
            write_i8(serial_file, 4)
            order = read_order(serial_file)
            act_val = read_i8(serial_file)
            print("Stop to allow continuity to read QR code...")
            sleep(1)
            print("\tSending data to server...")

            obj_to_send = {
                "hcsr04_1": distances[0],
                "hcsr04_2": distances[1],
                "hcsr04_3": distances[2], 
                "hcsr04_4": distances[3],
                "hcsr04_5": distances[4],
                "hcsr04_6": distances[5],
                "hcsr04_7": distances[6],
                "hcsr04_8": distances[7],
                "hcsr04_9": distances[8],
                "hcsr04_10": distances[9],
                "hcsr04_11": distances[10],
                "hcsr04_12": distances[11],
                "hcsr04_13": distances[12],
                "yl99_r": touch_r,
                "yl99_l": touch_l,
                "motor_left": motor_speed/max_motor_speed,
                "motor_right": motor_speed/max_motor_speed
            }
            print(obj_to_send)
            
            response = requests.post(url, data=json.dumps(obj_to_send), headers=headers)      
            
            action = response.json().get("action")
            # print("Action from server: "+str(action))
            # Ask user to choose action
            # action = int(input("Choose action: "))

            print("Action from server: "+str(action))

            if action == 0:
                write_order(serial_file, Order.MOTOR)
                write_i8(serial_file, 0)
            elif action == 1:
                write_order(serial_file, Order.MOTOR)
                write_i8(serial_file, 1)
            elif action == 2:
                write_order(serial_file, Order.MOTOR)
                write_i8(serial_file, 2)
            elif action == 3:
                write_order(serial_file, Order.MOTOR)
                write_i8(serial_file, 3)
            elif action == 4:
                write_order(serial_file, Order.MOTOR)
                write_i8(serial_file, 4)

            order = read_order(serial_file)
            act_val = read_i8(serial_file)

            if act_val == 0:
                print(f"Action => Ordered received: {str(order.value)} ==> FORWARD")
            elif act_val == 1:
                print(f"Action => Ordered received: {str(order.value)} ==> BACKWARD")
            elif act_val == 2:
                print(f"Action => Ordered received: {str(order.value)} ==> LEFT")
            elif act_val == 3:
                print(f"Action => Ordered received: {str(order.value)} ==> RIGHT")
            elif act_val == 4:
                print(f"Action => Ordered received: {str(order.value)} ==> STOP")
            else:
                print(f"Action => Ordered received: {str(order.value)} ==> UNKNOWN")
                raise Exception("Unknown action")
            
            sleep(action_timing)
            write_order(serial_file, Order.MOTOR)
            write_i8(serial_file, 4)
            order = read_order(serial_file)
            act_val = read_i8(serial_file)

            sleep(0.1)
    except KeyboardInterrupt:
        write_order(serial_file, Order.MOTOR)
        write_i8(serial_file, 4)

        order = read_order(serial_file)
        act_val = read_i8(serial_file)


        


            
