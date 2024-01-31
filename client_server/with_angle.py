#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
import math

import cv2 as cv
from pupil_apriltags import Detector
import numpy as np

def calculate_angle(p1, p2, q1, q2):
    """
    Calculate the angle between two vectors formed by points p1->p2 and q1->q2.
    """
    a = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    b = np.array([q2[0] - q1[0], q2[1] - q1[1]])
    angle_radians = np.arctan2(np.linalg.det([a, b]), np.dot(a, b))
    #angle_degrees = np.degrees(angle_radians)
    #print(angle_degrees)
    return angle_radians

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    ################################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    ##############################################################
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    elapsed_time = 0
    width = 36.0
    height = 21.2

    while True:
        start_time = time.time()

        ######################################################
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        ##############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )

        #################################################################
        debug_image = draw_tags(debug_image, tags, elapsed_time)

        elapsed_time = time.time() - start_time

        ################################################ 
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        ##############################################################
        cv.imshow('AprilTag Detect Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
    width=158.0,
    height=75.0,
    frame_width=960,
    frame_height=540
):
    if len(tags) == 2:
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
        
        # draw center
        cv.circle(image, (int(target_center_x), int(target_center_y)), 5, (0, 0, 255), 2)
        cv.circle(image, (int(robot_center_x), int(robot_center_y)), 5, (0, 0, 255), 2)
        
        # extract corners
        target_corners = target_tag.corners
        target_corner_01 = (int(target_corners[0][0]), int(target_corners[0][1]))
        target_corner_02 = (int(target_corners[1][0]), int(target_corners[1][1]))
        target_corner_03 = (int(target_corners[2][0]), int(target_corners[2][1]))
        target_corner_04 = (int(target_corners[3][0]), int(target_corners[3][1]))
        
        robot_corners = robot_tag.corners
        robot_corner_01 = (int(robot_corners[0][0]), int(robot_corners[0][1]))
        robot_corner_02 = (int(robot_corners[1][0]), int(robot_corners[1][1]))
        robot_corner_03 = (int(robot_corners[2][0]), int(robot_corners[2][1]))
        robot_corner_04 = (int(robot_corners[3][0]), int(robot_corners[3][1]))
        
        # draw corners
        cv.line(image, (target_corner_01[0], target_corner_01[1]),
                (target_corner_02[0], target_corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (target_corner_02[0], target_corner_02[1]),
                (target_corner_03[0], target_corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (target_corner_03[0], target_corner_03[1]),
                (target_corner_04[0], target_corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (target_corner_04[0], target_corner_04[1]),
                (target_corner_01[0], target_corner_01[1]), (0, 255, 0), 2)
        
        cv.line(image, (robot_corner_01[0], robot_corner_01[1]),
                (robot_corner_02[0], robot_corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (robot_corner_02[0], robot_corner_02[1]),
                (robot_corner_03[0], robot_corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (robot_corner_03[0], robot_corner_03[1]),
                (robot_corner_04[0], robot_corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (robot_corner_04[0], robot_corner_04[1]),
                (robot_corner_01[0], robot_corner_01[1]), (0, 255, 0), 2)
        
        # extract tag id
        target_tag_id = target_tag.tag_id
        robot_tag_id = robot_tag.tag_id
        
        # draw tag id
        cv.putText(image, 'target', (int(target_center_x) - 10, int(target_center_y) - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
        
        cv.putText(image, 'robot', (int(robot_center_x) - 10, int(robot_center_y) - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

        
        # Calculate the scale factors
        x_scale = width / frame_width
        y_scale = height / frame_height
        
        # Calculate the distance in centimeters
        distance_centimeters = math.sqrt((target_center_x - robot_center_x)**2 * x_scale**2 + (target_center_y - robot_center_y)**2 * y_scale**2)

        # draw center line
        cv.line(image, (int(target_center_x), int(target_center_y)),
                (int(robot_center_x), int(robot_center_y)), (0, 0, 255), 2)
        
        top_center_x = (robot_corner_01[0] + robot_corner_02[0]) / 2
        top_center_y = (robot_corner_01[1] + robot_corner_02[1]) / 2

        sagittal_axis_p1 = (robot_center_x, robot_center_y)
        sagittal_axis_p2 = (top_center_x, top_center_y)

        # Draw the sagittal axis
        cv.line(image, (int(sagittal_axis_p1[0]), int(sagittal_axis_p1[1])), (int(sagittal_axis_p2[0]), int(sagittal_axis_p2[1])), (255, 0, 255), 2)  # Magenta line for the sagittal axis

        # Calculate the vector to the target from the robot's center
        target_vector_q1 = (robot_center_x, robot_center_y)
        target_vector_q2 = (target_center_x, target_center_y)

        # Calculate the angle
        angle_radians = calculate_angle(sagittal_axis_p1, sagittal_axis_p2, target_vector_q1, target_vector_q2)
        angle_degrees = -np.degrees(angle_radians)

        # Display the angle on the frame
        angle_text = "Angle: {:.2f} degrees".format(angle_degrees)
        cv.putText(image, angle_text, (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv.LINE_AA)

        
        # draw distance above the center line
        # round distance to 1 decimal
        distance_centimeters = round(distance_centimeters, 1)
        cv.putText(image, str(distance_centimeters)+' cm', (int((target_center_x + robot_center_x) / 2) - 10, int((target_center_y + robot_center_y) / 2) - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
    
    # 処理時間
    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()