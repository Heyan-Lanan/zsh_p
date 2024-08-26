#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import fk
import sys
from scipy.spatial.transform import Rotation as R
from math import pi
import rospy
# print(rospy.__file__)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import threading
import pyrealsense2 as rs
import os
import cv2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from cv2 import aruco
import math
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2, Common_pb2
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
import utilities

# if cv2.__version__ < '4.7':
#     aruco.extendDictionary = aruco.Dictionary_create
#     aruco.getPredefinedDictionary = aruco.Dictionary_get
#     aruco.DetectorParameters = aruco.DetectorParameters_create

RGB_img = None
Depth_img = None
Vertices = None

rgb_tf = "camera_color_frame"
base_link = "base_link"
depth_tf = "camera_depth_frame"
camera_link = "camera_link"
rgb_topic = "/camera/color/image_raw"
depth_topic = "/camera/depth_registered/sw_registered/image_rect_raw"
pc_topic = '/camera/depth_registered/points'
RT_camera_tool = np.array([[-1.00000000e+00, 5.20417043e-17, -2.81719092e-15, -7.21644966e-16],
                           [-5.20417043e-17, -1.00000000e+00, -3.48332474e-15, 5.63900000e-02],
                           [-2.81719092e-15, -3.48332474e-15, 1.00000000e+00, 1.23050000e-01],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
                          )
RT_tool_camera = np.array([[-1.00000000e+00, 6.28837260e-17, -3.87884169e-15, 5.55111512e-17],
                           [-6.28837260e-17, -1.00000000e+00, -3.28209682e-15, 5.63900000e-02],
                           [-3.87884169e-15, -3.28209682e-15, 1.00000000e+00, -1.23050000e-01],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                          )
test_point = [0.058 * 3 + 0.002, 0.054 * 3 + 0.002, 0.0]


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


def rgb_callback(msg):
    global RGB_img
    bridge = CvBridge()
    # Convert ROS Image message to OpenCV image
    RGB_img = imgmsg_to_cv2(msg).copy()
    # print("Image height:", RGB_img.shape[0])
    # print("Image width:", RGB_img.shape[1])


def depth_callback(msg):
    global Depth_img
    bridge = CvBridge()
    Depth_img = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1").copy()
    # cv2.imshow("Depth Image", Depth_img)
    # cv2.waitKey(1)


def pc_callback(msg):
    global Vertices
    width = msg.width
    height = msg.height
    row_step = msg.row_step
    # print(width, height, row_step)
    # Convert the ROS PointCloud2 message to a PCL point cloud
    data2 = msg.data
    # print(data2[0])
    points = point_cloud2.read_points_list(
        msg, field_names=("x", "y", "z"))
    Vertices = points
    # print(Vertices[640 * 379 + 279])


def quaternion_to_rotation_matrix(q):  # x, y ,z ,w
    rot_matrix = np.array(
        [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
         [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
         [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
    )
    return rot_matrix


if __name__ == "__main__":

    rospy.init_node("qr")
    rospy.loginfo("Use rgbd!")
    rospy.Subscriber(rgb_topic, Image, rgb_callback, queue_size=10)
    rospy.Subscriber(pc_topic, PointCloud2, pc_callback, queue_size=10)
    while RGB_img is None or Vertices is None:
        continue

    args = utilities.parseConnectionArguments()
    joint_a_l = []
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        feedback = base_cyclic.RefreshFeedback()
        for i in range(7):
            joint_a_l.append(feedback.actuators[i].position)
        # print(joint_a_l)

    sign_param = 1
    while not rospy.is_shutdown():

        gray = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)
        params = aruco.DetectorParameters()
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        detector = aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
        frame_markers = aruco.drawDetectedMarkers(RGB_img.copy(), corners, ids)

        # gray = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)
        # aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        # parameters = aruco.DetectorParameters()
        # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # frame_markers = aruco.drawDetectedMarkers(RGB_img.copy(), corners, ids)

        points = []
        Oab = None
        Pxb = None
        Pyb = None
        test_point = [0.058 * 3 + 0.002, 0.054 * 3 + 0.002, 0.0]

        # print(corners)
        gcps = []
        if ids is not None:
            # print(ids)
            for i in range(ids.size):
                j = ids[i][0]
                if j != 0:
                    continue
                    # calculate center of aruco code
                x = int(round(np.average(corners[i][0][:, 0])))
                y = int(round(np.average(corners[i][0][:, 1])))
                # print(corners[i][0][:, 0], corners[i][0][:, 1])
                for k in range(4):
                    x1 = int(corners[i][0][:, 0][k])
                    y1 = int(corners[i][0][:, 1][k])
                    sign = 0
                    for dx in range(-1, 1):
                        for dy in range(-1, 1):
                            if math.isnan(Vertices[640 * (y1 + dy) + x1 + dx][0]) is False:
                                # print(np.array(Vertices[640 * (y1 + dy) + x1 + dx]))
                                # print(x1, y1)
                                if k == 0:
                                    Oab = np.array(Vertices[640 * (y1 + dy) + x1 + dx])
                                elif k == 3:
                                    Pxb = np.array(Vertices[640 * (y1 + dy) + x1 + dx])
                                elif k == 1:
                                    Pyb = np.array(Vertices[640 * (y1 + dy) + x1 + dx])
                                sign = 1
                                break
                        if sign:
                            break

                gcps.append((x, y, j, corners[i][0]))

            if Oab is not None and Pxb is not None and Pyb is not None:

                x1 = (Pxb - Oab) / np.linalg.norm(Pxb - Oab)
                y1 = (Pyb - Oab) / np.linalg.norm(Pyb - Oab)
                z1 = np.cross(x1, y1)

                length = np.linalg.norm(z1)
                z1 = z1 / length
                Rab = np.matrix([x1, y1, z1]).transpose()
                Tab = np.matrix(Oab).transpose()
                temp = np.hstack((Rab, Tab))
                RT_ab = np.vstack((temp, [0, 0, 0, 1]))
                RT_ba = np.linalg.inv(RT_ab)
                RT_chess_camera = RT_ab

                RT_tool_base = fk.fk(np.deg2rad(joint_a_l))
                RT_2 = np.dot(RT_tool_base, RT_tool_camera)
                RT_chess_base = np.dot(RT_2, RT_chess_camera)
                RT_param = []
                for i in np.array(RT_chess_base):
                    for j in i:
                        RT_param.append(float(j))
                if sign_param == 1:
                    sign_param = 0
                    rospy.set_param("RT_chess_base", RT_param)
                    print('success set RT_chess_base!')
                    print(RT_chess_base)
                    test_point_1 = np.hstack((test_point, [1])).reshape(4, 1)
                    tmp3 = np.array(np.dot(RT_chess_base, test_point_1)).reshape(4)[0: 3]
                    print(tmp3)
                    # print(RT_1)
                    # print(RT_2)

        cv2.imshow('Color', frame_markers)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
