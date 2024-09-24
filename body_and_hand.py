import copy
import sys

# sys.path.append('/home/kinova/catkin_ws_zsh2/src/test/scripts')

import numpy as np
import pyrealsense2 as rs
import cv2
import torch
from trajectory_msgs.msg import JointTrajectoryPoint
from openpose_light import OpenposeLight
import roslib
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Polygon
from tf import transformations  # rotation_matrix(), concatenate_matrices()
# import rviz_tools
from config import OPENPOSE_PATH
from cv2 import aruco
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from cvzone.HandTrackingModule import HandDetector


hand_detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
# [1 nose, 2 left eye, 3 right eye, 4 left ear, 5 right ear, 6 left shoulder, 7 right shoulder, 8 left elbow,
# 9 right elbow, 10 left wrist, 11 right wrist, 12 left hip, 13 right hip, 14 left knee, 15 right knee,
# 16 left ankle, 17 right ankle]
skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

kpt_names = ['nose', 'neck',
             'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
             'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
             'r_eye', 'l_eye',
             'r_ear', 'l_ear']
BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]

# w1 = 320
# h1 = 240
# w2 = 640
# h2 = 480

w1 = 640
h1 = 480
w2 = 640
h2 = 480

# rospy.init_node('human_pose', anonymous=False, log_level=rospy.INFO, disable_signals=False)
RT_camera_chess_center = None
RT_chess_base = None


def kpts(kpts, shape=None, radius=5, kpt_line=True, im=None):
    """
    Plot keypoints on the image.

    Args:
        kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
        shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
        radius (int, optional): Radius of the drawn keypoints. Default is 5.
        kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                   for human pose. Default is True.

    Note:
        `kpt_line=True` currently only supports human pose plotting.
    """

    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim in {2, 3}
    kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]] if is_pose else colors(i)
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            cv2.line(im, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
            # print(list(map(int, his_wrist[0:2])))
            # start = list(map(int, his_wrist[0:2]))
            # end = list(map(int, now_wrist[0:2]))

        im = np.asanyarray(im)
        # print(im.shape)
        # im = im[0:480, 0:640, :]
        # print(im)
        return im

def trans_chess_base(p):
    p1 = np.hstack((p, [1.0])).reshape(4, 1)
    p1_base = np.array(np.dot(RT_chess_base, p1)).reshape(4)[0: 3]
    p1 = Point(*p1_base)
    return p1


def trans_camera_base(p_3d):
    p_3d_4 = np.hstack((p_3d, [1])).reshape(4, 1)
    p_3d_chess = np.array(np.dot(RT_camera_chess_center, p_3d_4)).reshape(4)
    # p_3d_chess[0] += 0.275
    # p_3d_chess[1] += 0.178
    p_3d_base = np.array(np.dot(RT_chess_base, p_3d_chess)).reshape(4)[0: 3]
    return p_3d_base


class Trajectory():

    def __init__(self):
        self.v_list = [[0.0, 0.0, 0.0]] * 20
        self.his_wrist = None
        self.now_wrist = None
        self.v_all = [0.0, 0.0, 0.0]
        self.gamma = 0.9

    def init_wrist(self, new_wrist):
        self.now_wrist = new_wrist
        # del self.v_list[0]
        # self.v_list.append([0.01, 0.01, 0.01])

    def update_wrist(self, new_wrist):
        self.his_wrist = self.now_wrist
        self.now_wrist = new_wrist
        v = [self.now_wrist[i] - self.his_wrist[i] for i in range(3)]
        # print(np.linalg.norm(v))
        if np.linalg.norm(v) > 0.05:
            del self.v_list[0]
            self.v_list.append(v)

        for v1 in self.v_list:
            for j in range(len(v1)):
                self.v_all[j] = self.v_all[j] * self.gamma
                self.v_all[j] = self.v_all[j] + v1[j]

        norm = np.linalg.norm(self.v_all)
        if norm == 0.0:
            return [[0.0, 0.0, 0.0]]
        std_v = [i / norm for i in self.v_all]
        # print(std_v)
        end_list = []
        for g in [0.3, 0.6, 0.9]:
            end = [i + g * j for (i, j) in zip(self.now_wrist, std_v)]
            end_list.append(end)
        return end_list


class RealsensePose:
    def __init__(self, checkpoints_path, w=640, h=480):
        self.openpose = OpenposeLight(checkpoints_path)
        self.yolo_model = YOLO('/home/ustc/zsh_p/model/yolov8m-pose.pt')

        self.pipeline = rs.pipeline()

        self.pc = rs.pointcloud()

        self.align = rs.align(rs.stream.color)

        self.init_realsense(w, h)

        rospy.init_node('hand_tracker', anonymous=True)
        self.handpose_pub = rospy.Publisher('/hand_traj', JointTrajectoryPoint, queue_size=10)
        self.hand_pose = JointTrajectoryPoint()
        self.t1, self.t2 = 0, 0
        self.v = np.array([0, 0, 0])
        self.wp1 = np.array([0, 0, 0])
        self.wp2 = np.array([0, 0, 0])
        self.check_init = 0

    def init_realsense(self, w, h):
        config = rs.config()
        config.enable_device('220422302842')
        config.enable_stream(rs.stream.depth, w1, h1, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, w2, h2, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, w1, h1, rs.format.y8, 30)
        # config.enable_stream(rs.stream.color, w2, h2, rs.format.rgb8, 30)

        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        return color_frame, depth_frame

    def get_vertices_1(self, color_frame, depth_frame):
        points = self.pc.calculate(depth_frame)
        self.pc.map_to(color_frame)
        vertices = points.get_vertices()
        vertices = np.asanyarray(vertices).view(np.float32).reshape(-1, 3)  # xyz
        return vertices

    def keypoint_at(self, poses, index):
        point = [0, 0]
        if poses and poses[0].keypoints[index].min() >= 0:
            point = poses[0].keypoints[index].tolist()
        return tuple(point)

    def run(self):
        global RT_chess_base
        global RT_camera_chess_center
        mean_time = 0
        qr_sign = 1

        while not rospy.is_shutdown():
            if rospy.has_param('/RT_chess_base'):
                RT_chess_base = np.array(rospy.get_param("/RT_chess_base"))
                RT_chess_base = RT_chess_base.reshape((4, 4))
                break

        while True:
            current_time = cv2.getTickCount()

            color_frame, depth_frame = self.get_frames()
            vertices = self.get_vertices_1(color_frame, depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if qr_sign:

                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                params = aruco.DetectorParameters()
                aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
                detector = aruco.ArucoDetector(aruco_dict, params)
                corners, ids, _ = detector.detectMarkers(gray)

                points = []
                Oab = None
                Pxb = None
                Pyb = None

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

                            if k == 0:
                                Oab = np.array(vertices[w2 * y1 + x1])
                            elif k == 3:
                                Pxb = np.array(vertices[w2 * y1 + x1])
                            elif k == 1:
                                Pyb = np.array(vertices[w2 * y1 + x1])
                        gcps.append((x, y, j, corners[i][0]))
                    if Oab is not None and Pxb is not None and Pyb is not None:
                        # print(Oab, Pxb, Pyb)
                        x1 = (Pxb - Oab) / np.linalg.norm(Pxb - Oab)
                        y1 = (Pyb - Oab) / np.linalg.norm(Pyb - Oab)
                        z1 = np.cross(x1, y1)

                        length = np.linalg.norm(z1)
                        z1 = z1 / length
                        Rab = np.matrix([x1, y1, z1]).transpose()
                        Tab = np.matrix(Oab).transpose()
                        temp = np.hstack((Rab, Tab))
                        RT_ab = np.vstack((temp, [0, 0, 0, 1]))
                        RT_camera_chess_center = np.linalg.inv(RT_ab)
                        Oab_1 = np.hstack((Oab, [1])).reshape(4, 1)
                        # print(RT_ba)

            if RT_camera_chess_center is None:
                continue
            qr_sign = 0
            # print(RT_camera_chess_center)

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            params = aruco.DetectorParameters()
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
            detector = aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
            frame_markers = aruco.drawDetectedMarkers(color_image.copy(), corners, ids)

            # poses = self.openpose.predict(color_image)

            hands, frame_markers = hand_detector.findHands(frame_markers.copy(), draw=True, flipType=True)
            # frame_markers = color_image.copy()
            set_ids = [0, 12]
            if hands:
                # cv2.circle(color_image, hands[0]['lmList'][0][:2], 5, (0,0,255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][4][:2], 5, (0, 0, 255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][8][:2], 5, (0, 0, 255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][12][:2], 5, (0, 0, 255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][16][:2], 5, (0, 0, 255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][20][:2], 5, (0, 0, 255), -1)
                # print(hands[0]['lmList'])
                fingers_l = []
                for finger_id in range(21):
                    x = hands[-1]['lmList'][finger_id][0]
                    y = hands[-1]['lmList'][finger_id][1]
                    finger_3d = vertices[min(w2 * int(y) + int(x), 307200 - 1)]
                    # print(wrist_3d)
                    if finger_3d[0] != 0.0 and finger_3d[1] != 0.0 and finger_3d[2] != 0.0:
                        finger_3d_base = trans_camera_base(finger_3d)
                        # print(wrist_3d_base)
                        w_param = [float(finger_3d_base[0]), float(finger_3d_base[1]), float(finger_3d_base[2])]

                        rospy.set_param("finger_" + str(finger_id), w_param)
                        if finger_id == 10:
                            text = '{:.2f}, {:.2f}, {:.2f}'.format(w_param[0], w_param[1], w_param[2])
                            # cv2.putText(frame_markers, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        fingers_l.append(w_param)
                if len(fingers_l) > 0:
                    fingers_mean = np.mean(fingers_l, axis=0)
                    text = '{:.2f}, {:.2f}, {:.2f}'.format(fingers_mean[0], fingers_mean[1], fingers_mean[2])
                    cv2.putText(frame_markers, text, (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # if finger_id == 0:
                        #     self.t1 = rospy.get_time()
                        #     self.wp1 = finger_3d_base
                        #     delta_t = self.t1 - self.t2
                        #     if self.check_init > 0 and 0 < delta_t < 1:
                        #         self.v = (self.wp1 - self.wp2) / delta_t
                        #     else:
                        #         self.v = [0, 0, 0]
                        #     self.check_init += 1
                        #     self.wp2 = self.wp1
                        #     self.t2 = copy.copy(self.t1)
                        #     self.hand_pose.positions = self.wp1
                        #     self.hand_pose.velocities = self.v
                        #     self.hand_pose.time_from_start = rospy.Time.now()
                        #     self.handpose_pub.publish(self.hand_pose)

            # print(torch.cuda.is_available())
            results = self.yolo_model(color_image, device=0)
            result = results[0]
            keypoints = result.keypoints
            keypoints = np.array(keypoints.data.cpu())[0]
            if len(keypoints) > 0:
                wrist = keypoints[10]
                frame_markers = kpts(kpts=keypoints, im=frame_markers.copy(), shape=color_image.shape)

            # rendered_image = self.openpose.draw_poses(frame_markers, poses)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # wrist = self.keypoint_at(poses, 4)
                x = int(wrist[0])
                y = int(wrist[1])
                wrist_3d = vertices[w2 * int(y) + int(x)]
            # print(wrist_3d)
                if wrist_3d[0] != 0.0 and wrist_3d[1] != 0.0 and wrist_3d[2] != 0.0:
                    wrist_3d_base = trans_camera_base(wrist_3d)
                # print(wrist_3d_base)
                    rospy.set_param("wrist", [float(wrist_3d_base[0]), float(wrist_3d_base[1]), float(wrist_3d_base[2])])


                    # print(self.hand_pose)
                    # print(wrist_3d_base)
                    # text = '{:.2f}, {:.2f}, {:.2f}'.format(wrist_3d_base[0], wrist_3d_base[1], wrist_3d_base[2])
                    # cv2.putText(frame_markers, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            text = 'FPS: {}'.format(int(1 / mean_time * 10) / 10)
            cv2.putText(frame_markers, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            # images = np.hstack((rendered_image, depth_colormap))
            cv2.imshow('Color and Depth', frame_markers)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    rs_pose = RealsensePose(OPENPOSE_PATH)
    rs_pose.run()

# conda activate yolo
# python ./src/test/scripts/body_and_hand.py
