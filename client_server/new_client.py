import requests
import json




# URL of your Flask server
url = "http://192.168.1.226:8000/predict"
#url = 'https://rl-policy-server.onrender.com/predict'

# Send the tensor to the server
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps({
  "hcsr04_1": 40,
  "hcsr04_2": 45,
  "hcsr04_3": 60,
  "hcsr04_4": 80,
  "hcsr04_5": 98,
  "hcsr04_6": 200,
  "hcsr04_7": 210,
  "hcsr04_8": 180,
  "hcsr04_9": 150,
  "hcsr04_10": 90,
  "hcsr04_11": 75,
  "hcsr04_12": 50,
  "hcsr04_13": 50,
  "yl99_r": 0,
  "yl99_l": 0,
  "motor_left": 0,
  "motor_right": 0
}), headers=headers)


# Print the response from the server
print(response.json())
'''
0: "forward"
1: "backward"
2: "left"
3: "right"
4: "no action"
'''
