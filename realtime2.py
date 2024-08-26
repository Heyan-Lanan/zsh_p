#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import sys

# import moveit_commander
# import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
# sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf
from tf.transformations import quaternion_from_matrix, rotation_matrix
import geometry_msgs
import threading
import pyrealsense2 as rs
import os
import cv2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from cv2 import aruco
import math
# from cv_bridge.boost.cv_bridge_boost import cvtColor2
from ultralytics import YOLO
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2, Common_pb2
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

import fk
import utilities


RGB_img = None
Vertices = None
rgb_topic = "/camera/color/image_raw"
pc_topic = '/camera/depth_registered/points'
my_model = YOLO(r'/home/ustc/zsh_p/model/best.pt')
rgb_tf = "camera_color_frame"
base_link = "base_link"
camera_link = 'camera_link'
depth_tf = "camera_depth_frame"
RT_tool_camera = np.array([[-1.00000000e+00, 6.28837260e-17, -3.87884169e-15, 5.55111512e-17],
                           [-6.28837260e-17, -1.00000000e+00, -3.28209682e-15, 5.63900000e-02],
                           [-3.87884169e-15, -3.28209682e-15, 1.00000000e+00, -1.23050000e-01],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                          )


def imgmsg_to_cv2(img_msg):
    # print(img_msg.encoding)
    # if img_msg.encoding != "bgr8":
    #     rospy.logerr(
    #         "This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8")  # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                              # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                              dtype=dtype, buffer=img_msg.data)
    # print(len(np.frombuffer(img_msg.data, dtype=np.uint8)))
    # r = image_opencv[:, :, 0]
    # g = image_opencv[:, :, 1]
    # b = image_opencv[:, :, 2]
    image_opencv_2 = image_opencv[:, :, [2, 1, 0]]
    # print(image_opencv_2)
    # image_opencv_2 = cvtColor2(image_opencv, img_msg.encoding, "bgr8")
    # print(image_opencv_2.dtype)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv_2 = image_opencv_2.byteswap().newbyteorder()
    return image_opencv_2


def pc_callback(msg):
    global Vertices
    width = msg.width
    height = msg.height
    row_step = msg.row_step
    data2 = msg.data
    points = point_cloud2.read_points_list(
        msg, field_names=("x", "y", "z"))
    Vertices = points


def rgb_callback(msg):
    global RGB_img
    # bridge = CvBridge()
    # Convert ROS Image message to OpenCV image
    # RGB_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()
    RGB_img = imgmsg_to_cv2(msg)
    # print("Image height:", RGB_img.shape[0])
    # print("Image width:", RGB_img.shape[1])


def f1(base):
    mean_time = 0
    while not rospy.is_shutdown():

        current_time = cv2.getTickCount()

        joint_a_l = []
        feedback = base_cyclic.RefreshFeedback()
        for i in range(7):
            joint_a_l.append(feedback.actuators[i].position)

        img_2 = RGB_img.copy()
        results = my_model(img_2, conf=0.5, device='0')
        boxes = results[0].boxes
        if len(boxes.conf) > 0:
            for index in range(len(boxes.conf)):
                conf = boxes.conf[index]
                id = boxes.cls[index]
                # print(results[0].boxes.xywhn[index])
                xywhn_ = boxes.xywhn[index]
                xywhn = [item.cpu().numpy() for item in xywhn_]
                xyxy_ = boxes.xyxy[index]
                xyxy = [int(item) for item in xyxy_]
                center_x = int((xyxy[0] + xyxy[2]) / 2)
                center_y = int((xyxy[1] + xyxy[3]) / 2)

                center_3d = None
                sign = 0
                for dx in range(-1, 1):
                    for dy in range(-1, 1):
                        if math.isnan(Vertices[640 * (center_y + dy) + center_x + dx][0]) is False:
                            # print('chess in camera_link: ', np.array(Vertices[640 * (center_y + dy) + center_x + dx]))
                            center_3d = np.array(Vertices[640 * (center_y + dy) + center_x + dx])
                            sign = 1
                            break
                    if sign:
                        break
                if center_3d is not None:
                    RT_tool_base = fk.fk(np.deg2rad(joint_a_l))
                    RT_2 = np.dot(RT_tool_base, RT_tool_camera)
                    center_3d_1 = np.hstack((center_3d, [1])).reshape(4, 1)
                    center_3d_base = list(np.array(np.dot(RT_2, center_3d_1)).reshape(4)[0: 3])

                    print('chess_' + str(int(id.cpu().numpy())) + ' in base_link: ', center_3d_base)
                    rospy.set_param("chess_" + str(int(id.cpu().numpy())),
                                    [float(center_3d_base[0]), float(center_3d_base[1]), float(center_3d_base[2])])

                p1 = (xyxy[0], xyxy[1])
                p2 = (xyxy[2], xyxy[1])
                p3 = (xyxy[0], xyxy[3])
                p4 = (xyxy[2], xyxy[3])
                cv2.line(img_2, p1, p2, color=(0, 255, 0), thickness=2)
                cv2.line(img_2, p2, p4, color=(0, 255, 0), thickness=2)
                cv2.line(img_2, p4, p3, color=(0, 255, 0), thickness=2)
                cv2.line(img_2, p3, p1, color=(0, 255, 0), thickness=2)
                # text = 'id: {}, conf: {:.2f}'.format(int(id), conf)
                text = 'id: {}'.format(int(id))
                cv2.putText(img_2, text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print('\n')

        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        text = 'FPS: {}'.format(int(1 / mean_time * 10) / 10)
        cv2.putText(img_2, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.imshow('Color', img_2)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":

    # example = ExampleMoveItTrajectories()
    #
    # # For testing purposes
    # success = example.is_init_success
    # try:
    #     rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
    # except:
    #     pass
    #
    # if example.is_gripper_present and success:
    #     rospy.loginfo("Opening the gripper...")
    #     success &= example.reach_gripper_position(0)
    #     print(success)

    rospy.init_node("chess")
    rospy.loginfo("Use chess!")
    # Subscribe to RGB topics
    rospy.Subscriber(rgb_topic, Image, rgb_callback, queue_size=5)
    rospy.Subscriber(pc_topic, PointCloud2, pc_callback, queue_size=5)

    while RGB_img is None or Vertices is None:
        continue

    args = utilities.parseConnectionArguments()
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        f1(base)

# python3  ./src/test/scripts/realtime2.py
