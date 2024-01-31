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

    count_hcsr04_sensors =  len(hc_sr04_sensors)# todo: change this to 13
  
    # 1. read hcsr04 sensors
    while True:
        distances = []
        for i in range(count_hcsr04_sensors):
            dist = hc_sr04_sensors[i].read_distance()
            sleep(0.1)
            distances.append(dist)

        print(distances)
    sleep(0.1)

    