from flask import Flask, request, jsonify
import torch
import numpy as np
import torch.nn as nn
import copy
import time
import argparse
import math
import numpy as np
import cv2 as cv
from pupil_apriltags import Detector

def normalize_to_range(value, min_val, max_val, new_min, new_max, clip=False):
    """
    Normalizes value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :type value: float
    :param min_val: value's min value, value ∈ [min_val, max_val]
    :type min_val: float
    :param max_val: value's max value, value ∈ [min_val, max_val]
    :type max_val: float
    :param new_min: normalized range min value
    :type new_min: float
    :param new_max: normalized range max value
    :type new_max: float
    :param clip: whether to clip normalized value to new range or not, defaults to False
    :type clip: bool, optional
    :return: normalized value ∈ [new_min, new_max]
    :rtype: float
    """
    value = float(value)
    min_val = float(min_val)
    max_val = float(max_val)
    new_min = float(new_min)
    new_max = float(new_max)

    if clip:
        return np.clip((new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max, new_min, new_max)
    else:
        return (new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max

def calculate_angle(p1, p2, q1, q2):
    """
    Calculate the angle between two vectors formed by points p1->p2 and q1->q2.
    """
    a = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    b = np.array([q2[0] - q1[0], q2[1] - q1[1]])
    angle_radians = np.arctan2(np.linalg.det([a, b]), np.dot(a, b))
    return angle_radians

# def calculate_angle(p1, p2, q1, q2):
#     """
#     Calculate the angle between two vectors formed by points p1->p2 and q1->q2.
#     """
#     a = np.array([p2[0] - p1[0], p2[1] - p1[1]])
#     b = np.array([q2[0] - q1[0], q2[1] - q1[1]])
#     cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#     return np.arccos(np.clip(cos_angle, -1, 1))  # Clip for numerical stability

def compute_distance_angle(cap_device=0, width=150.0, height=75.0, frame_width=960, frame_height=540):
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    at_detector = Detector(
    families='tag36h11',
    nthreads=1,
    quad_decimate=2.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
    )

    # check if camera returns image
    ret, image = cap.read()
    if not ret:
        print('Failed to capture image')
        return None, None

    ##############################################################
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    tags = at_detector.detect(
        image,
        estimate_tag_pose=False,
        camera_params=None,
        tag_size=None,
    )
    #################################################################
        
    if len(tags) == 2:
        print("Tags detected")
        # detect target and robot
        if tags[0].tag_id == 0:
            target_tag = tags[0]
            robot_tag = tags[1]
        else:
            target_tag = tags[1]
            robot_tag = tags[0]
        
        # extract center
        target_center_x = target_tag.center[0]
        target_center_y = target_tag.center[1]
        robot_center_x = robot_tag.center[0]
        robot_center_y = robot_tag.center[1]
        
        
        # extract corners
        target_corners = target_tag.corners
        
        robot_corners = robot_tag.corners
        robot_corner_01 = (int(robot_corners[0][0]), int(robot_corners[0][1]))
        robot_corner_02 = (int(robot_corners[1][0]), int(robot_corners[1][1]))
        
        # extract tag id
        target_tag_id = target_tag.tag_id
        robot_tag_id = robot_tag.tag_id
        
        # Calculate the scale factors
        x_scale = width / frame_width
        y_scale = height / frame_height
        
        # Calculate the distance in centimeters
        distance_centimeters = math.sqrt((target_center_x - robot_center_x)**2 * x_scale**2 + (target_center_y - robot_center_y)**2 * y_scale**2)

        top_center_x = (robot_corner_01[0] + robot_corner_02[0]) / 2
        top_center_y = (robot_corner_01[1] + robot_corner_02[1]) / 2

        sagittal_axis_p1 = (robot_center_x, robot_center_y)
        sagittal_axis_p2 = (top_center_x, top_center_y)

        # Calculate the vector to the target from the robot's center
        target_vector_q1 = (robot_center_x, robot_center_y)
        target_vector_q2 = (target_center_x, target_center_y)

        # Calculate the angle
        angle_radians = calculate_angle(sagittal_axis_p1, sagittal_axis_p2, target_vector_q1, target_vector_q2)
        angle_degrees = np.degrees(angle_radians)

        # Display the angle on the frame
        angle_text = "Angle: {:.2f} degrees".format(angle_degrees)
        print(angle_text)

        # draw distance above the center line
        # round distance to 1 decimal
        distance_centimeters = round(distance_centimeters, 1)
        distance = distance_centimeters/100
        return distance, angle_degrees
    return None, None

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--frame_width", help='frame_width', type=int, default=960)
    parser.add_argument("--frame_height", help='frame_height', type=int, default=540)
    parser.add_argument("--real_width", help='real_width', type=int, default=158.0)
    parser.add_argument("--real_height", help='real_height', type=int, default=75.0)
    args = parser.parse_args()

    return args

def normalize_tensor(tensor):
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Shift the range to start from 0
    shifted_tensor = tensor - min_val

    # Divide by the range (maximum value - minimum value)
    range_tensor = max_val - min_val
    normalized_tensor = shifted_tensor / range_tensor

    # Multiply by 2 and subtract 1 to bring the range to [-1, 1]
    normalized_tensor = normalized_tensor * 2 - 1

    return normalized_tensor
########### INITIALIZATION ###########

# Model definition
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        
        # policy_net (mlp_extractor)
        self.policy_net_0 = nn.Linear(48, 1024)
        self.policy_net_2 = nn.Linear(1024, 512)
        self.policy_net_4 = nn.Linear(512, 256)
        self.action_net = nn.Linear(256, 5)
        
        # value_net (mlp_extractor)
        self.value_net_0 = nn.Linear(48, 2048)
        self.value_net_2 = nn.Linear(2048, 1024)
        self.value_net_4 = nn.Linear(1024, 512)
        self.value_net = nn.Linear(512, 1)

        # Relu activation function
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        # policy_net (mlp_extractor)
        x_ = x
        x = self.relu(self.policy_net_0(x))        
        x = self.relu(self.policy_net_2(x))        
        x = self.relu(self.policy_net_4(x))        
        action = self.action_net(x)

        # value_net (mlp_extractor)
        x = self.relu(self.value_net_0(x_))
        x = self.relu(self.value_net_2(x))
        x = self.relu(self.value_net_4(x))
        value = self.value_net(x)
       
        return action, value
    
# Initialize values
prev_action = [0,0,0,0,0]
state_t_w = None

# Initialize policy
load_path = 'policy.pth'
model = Policy()
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
state_dict = torch.load(load_path, map_location=torch.device(device))
print(f'Running on {device}')
model = Policy()
model.eval()

# Assign the weights using the state_dict
model.policy_net_0.weight.data = state_dict['mlp_extractor.policy_net.0.weight']
model.policy_net_0.bias.data = state_dict['mlp_extractor.policy_net.0.bias']
model.policy_net_2.weight.data = state_dict['mlp_extractor.policy_net.2.weight']
model.policy_net_2.bias.data = state_dict['mlp_extractor.policy_net.2.bias']
model.policy_net_4.weight.data = state_dict['mlp_extractor.policy_net.4.weight']
model.policy_net_4.bias.data = state_dict['mlp_extractor.policy_net.4.bias']
model.action_net.weight.data = state_dict['action_net.weight']
model.action_net.bias.data = state_dict['action_net.bias']

model.value_net_0.weight.data = state_dict['mlp_extractor.value_net.0.weight']
model.value_net_0.bias.data = state_dict['mlp_extractor.value_net.0.bias']
model.value_net_2.weight.data = state_dict['mlp_extractor.value_net.2.weight']
model.value_net_2.bias.data = state_dict['mlp_extractor.value_net.2.bias']
model.value_net_4.weight.data = state_dict['mlp_extractor.value_net.4.weight']
model.value_net_4.bias.data = state_dict['mlp_extractor.value_net.4.bias']
model.value_net.weight.data = state_dict['value_net.weight']
model.value_net.bias.data = state_dict['value_net.bias']

print("All the layers imported successfully")

# Flask server
app = Flask(__name__)

def create_one_hot_vector(action_number, num_possible_values=5):
    if action_number < 0 or action_number >= num_possible_values:
        raise ValueError("Action number is out of range")
    
    one_hot_vector = [0] * num_possible_values
    one_hot_vector[action_number] = 1
    
    return one_hot_vector

def build_initial_state(data, d_t, a_t, motors_speed, prev_action):
    # distance sensors cm to m, clipped to 1.0m
    distance_sensors = [
        min(data['hcsr04_1']/100, 1.0),
        min(data['hcsr04_2']/100, 1.0),
        min(data['hcsr04_3']/100, 1.0),
        min(data['hcsr04_4']/100, 1.0),
        min(data['hcsr04_5']/100, 1.0),
        min(data['hcsr04_6']/100, 1.0),
        min(data['hcsr04_7']/100, 1.0),
        min(data['hcsr04_8']/100, 1.0),
        min(data['hcsr04_9']/100, 1.0),
        min(data['hcsr04_10']/100, 1.0),
        min(data['hcsr04_11']/100, 1.0),
        min(data['hcsr04_12']/100, 1.0),
        min(data['hcsr04_13']/100, 1.0)
    ]
    left_motor_speed = data['motor_left']
    right_motor_speed = data['motor_right']
    d_t = normalize_to_range(d_t, 0.0, 1.0, 0.0, 1.0, clip=True)
    a_t = normalize_to_range(a_t,  -np.pi, np.pi, -1.0, 1.0, clip=True)
    right_touch = data['yl99_r']
    
    if (right_touch != 0 and right_touch != 1):
        response = jsonify({'code': 400, 'message': f'Invalid right touch value, got {right_touch}', 'action': None})
        print("Invalid right touch value, got {right_touch}")
        return None, response
    
    left_touch = data['yl99_l']
    if (right_touch != 0 and right_touch != 1):
        print("Invalid left touch value, got {left_touch}")
        response = jsonify({'code': 400, 'message': f'Invalid left touch value, got {left_touch}', 'action': None})
        return None, response
    
    state = [d_t, a_t, left_motor_speed, right_motor_speed, left_touch, right_touch] + prev_action + distance_sensors
    response = jsonify({'code': 201, 'message': 'Got first state, waiting for second', 'action': None})
    
    return state, response


def create_response(data, policy, motors_speed, d_t, a_t, prev_action, state_t_w):
    # distance sensors cm to m, clipped to 1.0m
    distance_sensors = [
        min(data['hcsr04_1']/100, 1.0),
        min(data['hcsr04_2']/100, 1.0),
        min(data['hcsr04_3']/100, 1.0),
        min(data['hcsr04_4']/100, 1.0),
        min(data['hcsr04_5']/100, 1.0),
        min(data['hcsr04_6']/100, 1.0),
        min(data['hcsr04_7']/100, 1.0),
        min(data['hcsr04_8']/100, 1.0),
        min(data['hcsr04_9']/100, 1.0),
        min(data['hcsr04_10']/100, 1.0),
        min(data['hcsr04_11']/100, 1.0),
        min(data['hcsr04_12']/100, 1.0),
        min(data['hcsr04_13']/100, 1.0)
    ]
    left_motor_speed = data['motor_left']
    right_motor_speed = data['motor_right']
    right_touch = data['yl99_r']
    if (right_touch != 0 and right_touch != 1):
        response = jsonify({'code': 400, 'message': f'Invalid right touch value, got {right_touch}', 'action': None})
        print("Invalid right touch value, got {right_touch}")
        return response, None, None
    
    left_touch = data['yl99_l']
    if (right_touch != 0 and right_touch != 1):
        print("Invalid left touch value, got {left_touch}")
        response = jsonify({'code': 400, 'message': f'Invalid left touch value, got {left_touch}', 'action': None})
        return response, None, None
    
    d_t = normalize_to_range(d_t, 0.0, 1.0, 0.0, 1.0, clip=True)
    a_t = normalize_to_range(a_t,  -np.pi, np.pi, -1.0, 1.0, clip=True)
    state = [d_t, a_t, left_motor_speed, right_motor_speed, left_touch, right_touch] + prev_action + distance_sensors
    print("State: ")
    print(state)
    full_state = state + state_t_w
    full_state = np.array(full_state, dtype=np.float32)
    state_tensor = torch.tensor(full_state).to(device)
    #state_tensor = normalize_tensor(state_tensor)
    with torch.no_grad():
        action, value = policy(state_tensor)
        action_logits = action.cpu()
        value = value.cpu()
    action = torch.argmax(action_logits).item()
    action_vector = create_one_hot_vector(action)
    print("Action chosen: ", action, " Value: ", value)
    response = jsonify({'code':200, 'message':'OK', 'action':action})
    
    return response, state, action_vector

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        global prev_action
        global state_t_w
        # Parse the JSON to get the tensor list
        motors_speed = 0.4
        data = request.get_json()
        d_t, a_t = compute_distance_angle(cap_device, real_width, real_height, frame_width, frame_height)
     
        #a_t = degrees_to_radians(a_t)

        
        if d_t is not None and a_t is not None:
            a_t = -a_t
            print("Angle with sign changed :", a_t)
            if d_t <= target_distance: # termination condition: when the robot is close to the target
                response = jsonify({'code': 202, 'message': 'Target reached!', 'action': 4})
                return response
            if state_t_w is None and d_t is not None and a_t is not None:
                motors_speed = 0.0
                state_t_w, response = build_initial_state(data, d_t, a_t, motors_speed, prev_action)
                return response        
            elif d_t is not None and a_t is not None:
                response, state, action_vector = create_response(data, model, motors_speed, d_t,a_t , prev_action, state_t_w)
                if state is not None and action_vector is not None:
                    prev_action = action_vector
                    state_t_w = state
                else:
                    print('Error occurred!')
        else:
            print('No tags detected or no feed from webcam!')
            response = jsonify({'code': 400, 'message': 'No tags detected or no feed from webcam!', 'action': None})

        
        # Send back the result
        return response
    
if __name__ == '__main__':
    args = get_args()
    # global arguments
    global cap_device
    global frame_width
    global frame_height
    global real_width
    global real_height
    global target_distance
    
    target_distance = 0.1
    cap_device = args.device
    frame_width = args.frame_width
    frame_height = args.frame_height
    real_width = args.real_width
    real_height = args.real_height
    
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)