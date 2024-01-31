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


def get_action_mask(motor_speeds, current_dist_sensors, current_touch_sensors, current_tar_a,
                    ds_thresholds, ds_max, number_of_distance_sensors=13, num_actions=5):
    """
    Returns the mask for the current state. The mask is a list of bools where each element corresponds to an
    action, and if the bool is False the corresponding action is masked, i.e. disallowed.
    Action masking allows the agent to perform certain actions under certain conditions, disallowing illogical
    decisions.

    Mask is modified first by the touch sensors, and if no collisions are detected, secondly by
    the distance sensors.

    :return: The action mask list of bools
    :rtype: list of booleans
    """
    touched_obstacle_left = None
    touched_obstacle_right = None
    mask = [True for _ in range(num_actions)]
    # Mask backward action that will cause the agent to move backwards by default
    if motor_speeds[0] <= 0.0 and motor_speeds[1] <= 0.0:
        mask[1] = False

    # Create various flag lists for the distance sensors
    # Whether any sensor is reading under its minimum threshold, and calculate and store how much
    reading_under_threshold = [0.0 for _ in range(number_of_distance_sensors)]
    # Whether there is any obstacle under half the max range of the distance sensors
    detecting_obstacle = [False for _ in range(number_of_distance_sensors)]
    # Whether there is an obstacle really close, i.e. under half the minimum threshold, in front
    front_under_half_threshold = False
    for i in range(len(current_dist_sensors)):
        if current_dist_sensors[i] <= ds_max[i] / 2:
            detecting_obstacle[i] = True
        # Sensor is reading under threshold, store how much under threshold it reads
        if current_dist_sensors[i] < ds_thresholds[i]:
            reading_under_threshold[i] = ds_thresholds[i] - current_dist_sensors[i]
            # Frontal sensor (index 4 to 8) is reading under half threshold
            if i in [4, 5, 6, 7, 8] and current_dist_sensors[i] < (ds_thresholds[i] / 2):
                front_under_half_threshold = True
    # Split left and right slices to use later
    reading_under_threshold_left = reading_under_threshold[0:5]
    reading_under_threshold_right = reading_under_threshold[8:13]
    
    '''
    left: 0-5,  8.0, 8.0, 8.0,  10.15
    center: 5-8, 14.7, 13.15, 12.7, 13.15, 14.7, 10.15
    right: 8-13  10.15, 8.0, 8.0, 8.0
    
    '''

    # First modify mask using the touch sensors as they are more important than distance sensors
    # Unmask backward and mask forward if a touch sensor is detecting collision
    if any(current_touch_sensors):
        mask[0] = False
        mask[1] = True
        # Set flags to keep masking/unmasking until robot is clear of obstacles
        # Distinguish between left and right touch
        if current_touch_sensors[0]:
            touched_obstacle_left = True
        if current_touch_sensors[1]:
            touched_obstacle_right = True
    # Not touching obstacles and can't detect obstacles with distance sensors,
    # can stop masking forward and unmasking backwards
    elif not any(reading_under_threshold):
        touched_obstacle_left = False
        touched_obstacle_right = False

    # Keep masking forward and unmasking backwards as long as a touched_obstacle flag is True
    if touched_obstacle_left or touched_obstacle_right:
        mask[0] = False
        mask[1] = True

        if touched_obstacle_left and not touched_obstacle_right:
            # Touched on left, mask left action, unmask right action
            mask[2] = False
            mask[3] = True
        if touched_obstacle_right and not touched_obstacle_left:
            # Touched on right, mask right action, unmask left action
            mask[3] = False
            mask[2] = True
    # If there are no touching obstacles, modify mask by distance sensors
    else:
        # Obstacles very close in front
        if front_under_half_threshold:
            # Mask forward
            mask[0] = False

        # Target is straight ahead, no obstacles close-by
        if not any(detecting_obstacle) and abs(current_tar_a) < 0.1:
            # Mask left and right turning
            mask[2] = mask[3] = False

        angle_threshold = 0.1
        # No obstacles on the right and target is on the right or
        # no obstacles on the right and obstacles on the left regardless of target direction
        if not any(reading_under_threshold_right):
            if current_tar_a <= - angle_threshold or any(reading_under_threshold_left):
                mask[2] = False  # Mask left

        # No obstacles on the left and target is on the left or
        # no obstacles on the left and obstacles on the right regardless of target direction
        if not any(reading_under_threshold_left):
            if current_tar_a >= angle_threshold or any(reading_under_threshold_right):
                mask[3] = False  # Mask right

        # Both left and right sensors are reading under threshold
        if any(reading_under_threshold_left) and any(reading_under_threshold_right):
            # Calculate the sum of how much each sensor's threshold is surpassed
            sum_left = sum(reading_under_threshold_left)
            sum_right = sum(reading_under_threshold_right)
            # If left side has obstacles closer than right
            if sum_left - sum_right < -5.0:
                mask[2] = True  # Unmask left
            # If right side has obstacles closer than left
            elif sum_left - sum_right > 5.0:
                mask[3] = True  # Unmask right
            # If left and right side have obstacles on roughly equal distances
            else:
                # Enable touched condition
                touched_obstacle_right = touched_obstacle_left = True
    int_mask = [int(x) for x in mask]
    return int_mask

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

def compute_distance_angle(cap_device=0, width=158.0, height=75.0, frame_width=960, frame_height=540):
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
        print(f"Invalid left touch value, got {left_touch}")
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
    #motor_speeds = [left_motor_speed, right_motor_speed]
    
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

    action_mask = get_action_mask([left_motor_speed, right_motor_speed], distance_sensors, [left_touch, right_touch], a_t,
                                ds_thresholds, ds_max)
    print("Action mask: ", action_mask)
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
    masked_action = action_logits * torch.tensor(action_mask, dtype=torch.float32)
    print(f'Action logits: {action_logits}')
    print(f'Mask: {action_mask}')
    print(f'Masked action: {masked_action}')
    action = torch.argmax(masked_action).item()
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
        #d_t = 1
        #a_t = 0
        a_t = -a_t # change sign of the angle
        print('distance: ', d_t)
        print('angle:', a_t)
        
        if d_t is not None and a_t is not None:
            if d_t < target_distance: # termination condition: when the robot is close to the target
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
    global ds_thresholds
    global ds_max
    
    
    target_distance = 0.05
    cap_device = args.device
    frame_width = args.frame_width
    frame_height = args.frame_height
    real_width = args.real_width
    real_height = args.real_height
    ds_thresholds = [8.0, 8.0, 8.0, 10.15, 14.7, 13.15,
                              12.7,
                              13.15, 14.7, 10.15, 8.0, 8.0, 8.0]

    # ds_thresholds = [0.0 for _ in range(13)]
    
    ds_max=[1.0 for _ in range(13)]
    
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)