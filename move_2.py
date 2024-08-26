import torch
import torchmin
from scipy.optimize import minimize
import scipy
# print(scipy.__version__)
import sys
import os
import time
import threading
import numpy as np
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2, Common_pb2
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

import rospy
# import fk
from fk import fk, A, fk_j6
import utilities

base = None
target_pose = None
wrist = None
chess_new_pose = [[0.054 * (-4) + 0.001 * (-3), 0.058 * 0 + 0.001 * 0, 0.15],
          [0.054 * (-4) + 0.001 * (-3), 0.058 * 4 + 0.001 * 3, 0.15],
          [0.054 * (-4) + 0.001 * (-3), 0.058 * 3 + 0.001 * 2, 0.15]
          ]
pose_arm =np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
])

def loss_1(q):
    q = np.deg2rad(q)
    new_pose = fk(q)
    # if new_pose[2][3] < 0.01:
    #     return 100.00
    pose_m = A(target_pose[0], target_pose[1], target_pose[2])
    # loss1 = np.sum((pose - pose_m) ** 2) ** 0.5
    loss2 = np.sum((new_pose[:3, :3] - pose_m[:3, :3]) ** 2) * 0.2 + np.sum((new_pose[:3, 3] - pose_m[:3, 3]) ** 2)
    # loss3 = 0.0006 / np.sum((new_pose[:3, 3] - wrist) ** 2)
    # print(loss3, loss2)
    # loss1 = np.linalg.norm(pose - target_pose, ord=2)
    return loss2

def loss_2(q):
    q = np.deg2rad(q)
    new_pose = fk(q)
    new_pose_j6 = fk_j6(q)
    if new_pose[2][3] < 0.02:
        return 100.00
    loss1 = 1.0 / np.sum((new_pose[:3, 3] - wrist) ** 2)
    loss2 = 1.0 / np.sum((new_pose_j6[:3, 3] - wrist) ** 2)
    loss3 = np.sum((new_pose[:3, :3] - pose_arm) ** 2) * 1.0
    # print(loss1, loss2, loss3)
    return loss1 + loss2 + loss3


def move_to(base_cyclic, tolerance=0.01, speed_mu_1=0.5):
    global wrist
    # print('Move to: ', target_pose)
    while True:
        # t1 = time.time()
        while not rospy.is_shutdown():
            if rospy.has_param('wrist'):
                wrist = rospy.get_param('wrist')
                break

        feedback = base_cyclic.RefreshFeedback()
        cur_x = feedback.base.tool_pose_x
        cur_y = feedback.base.tool_pose_y
        cur_z = feedback.base.tool_pose_z
        cur_pose = np.array([cur_x, cur_y, cur_z])
        cur_dis = np.linalg.norm(cur_pose[:2] - target_pose[:2], ord=2)
        # print(target_pose, cur_pose, cur_dis)
        if cur_dis < tolerance and target_pose[2] + 0.004 > cur_pose[2] > target_pose[2] - 0.004:
            # print('Actual pose: ', cur_pose)
            break

        # print(cur_pose, cur_dis)
        cur_q = []
        for i in range(7):
            cur_q.append(feedback.actuators[i].position)
        # cur_q = torch.tensor(cur_q)
        cur_q = np.array(cur_q)
        wrist_tool_dis = np.sum((cur_pose - wrist) ** 2) ** 0.5
        loss = loss_1
        speed_mu = speed_mu_1
        # print(wrist_tool_dis)
        if wrist_tool_dis < 0.2:
            loss = loss_2
            speed_mu = 0.4
        result = minimize(loss, cur_q, method="l-bfgs-b", options={'maxiter': 5})
        # t2 = time.time()
        l1 = loss(result.x)
        l2 = loss(cur_q)
        delta = result.x - cur_q
        # print(l2, l1)
        # print(result.x)

        speeds = delta * speed_mu
        i = 0
        joint_speeds = Base_pb2.JointSpeeds()
        for speed in speeds:
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i
            joint_speed.value = speed
            joint_speed.duration = 0
            i = i + 1
        base.SendJointSpeedsCommand(joint_speeds)

    return True
def move_to_2(base_cyclic, tolerance=0.01, speed_mu_1=0.5):
    global wrist
    # print('Move to: ', target_pose)
    while True:
        # t1 = time.time()
        while not rospy.is_shutdown():
            if rospy.has_param('wrist'):
                wrist = rospy.get_param('wrist')
                break

        feedback = base_cyclic.RefreshFeedback()
        cur_x = feedback.base.tool_pose_x
        cur_y = feedback.base.tool_pose_y
        cur_z = feedback.base.tool_pose_z
        cur_pose = np.array([cur_x, cur_y, cur_z])
        cur_dis = np.linalg.norm(cur_pose - target_pose, ord=2)
        # print(target_pose, cur_pose, cur_dis)
        if cur_dis < tolerance:
            # print('Actual pose: ', cur_pose)
            break

        # print(cur_pose, cur_dis)
        cur_q = []
        for i in range(7):
            cur_q.append(feedback.actuators[i].position)
        # cur_q = torch.tensor(cur_q)
        cur_q = np.array(cur_q)
        wrist_tool_dis = np.sum((cur_pose - wrist) ** 2) ** 0.5
        loss = loss_1
        speed_mu = speed_mu_1
        # print(wrist_tool_dis)
        if wrist_tool_dis < 0.2:
            loss = loss_2
            speed_mu = 0.4
        result = minimize(loss, cur_q, method="l-bfgs-b", options={'maxiter': 5})
        # t2 = time.time()
        l1 = loss(result.x)
        l2 = loss(cur_q)
        delta = result.x - cur_q
        # print(l2, l1)
        # print(result.x)

        speeds = delta * speed_mu
        i = 0
        joint_speeds = Base_pb2.JointSpeeds()
        for speed in speeds:
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i
            joint_speed.value = speed
            joint_speed.duration = 0
            i = i + 1
        base.SendJointSpeedsCommand(joint_speeds)

    return True

def reach(id):
    global base
    global target_pose
    param_name = '/chess_' + str(id)
    chess_base = None
    while not rospy.is_shutdown():
        if rospy.has_param(param_name):
            chess_base = rospy.get_param(param_name)
            break

    args = utilities.parseConnectionArguments()
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        target_pose = chess_base

        target_pose[2] = 0.036
        move_gripper(0.00)
        move_to(base_cyclic, tolerance=0.008, speed_mu_1=0.6)
        move_gripper(0.50)

        target_pose[2] = 0.200
        move_to_2(base_cyclic, speed_mu_1=0.6)

        target_pose = chess_new_pose[id]
        move_to_2(base_cyclic, speed_mu_1=0.6)
        move_gripper(0.00)


def move_gripper(value):
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.value = value
    finger.finger_identifier = 1
    base.SendGripperCommand(gripper_command)
    time.sleep(0.8)


def pick():
    return 1


def print_loss(base_cyclic):
    global wrist
    while True:
        if rospy.has_param('wrist'):
            wrist = rospy.get_param('wrist')
        wrist = np.array(wrist)
        feedback = base_cyclic.RefreshFeedback()
        cur_x = feedback.base.tool_pose_x
        cur_y = feedback.base.tool_pose_y
        cur_z = feedback.base.tool_pose_z
        cur_pose = np.array([cur_x, cur_y, cur_z])
        cur_dis = np.linalg.norm(cur_pose - wrist, ord=2)
        print(cur_dis)

    return True


if __name__ == "__main__":
    while not rospy.is_shutdown():
        if rospy.has_param('/RT_chess_base'):
            RT_chess_base = np.array(rospy.get_param("/RT_chess_base"))
            for i in range(len(chess_new_pose)):
                target_1 = np.hstack((chess_new_pose[i], [1])).reshape(4, 1)
                RT_chess_base_1 = RT_chess_base.reshape((4, 4))
                target_base = np.array(np.dot(RT_chess_base_1, target_1)).reshape(4)[0: 3]
                chess_new_pose[i] = target_base
            break
    # print(chess_new_pose)
    reach(0)
    # args = utilities.parseConnectionArguments()
    # with utilities.DeviceConnection.createTcpConnection(args) as router:
    #     base = BaseClient(router)
    #     base_cyclic = BaseCyclicClient(router)
    #     print_loss(base_cyclic)
