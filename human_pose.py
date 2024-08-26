import sys

# sys.path.append('/home/kinova/catkin_ws_zsh2/src/test/scripts')

import numpy as np
import pyrealsense2 as rs
import cv2
from openpose_light import OpenposeLight
import roslib
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Polygon
from tf import transformations  # rotation_matrix(), concatenate_matrices()
from config import OPENPOSE_PATH
from cv2 import aruco

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
w2 = 1280
h2 = 720

rospy.init_node('human_pose', anonymous=False, log_level=rospy.INFO, disable_signals=False)
RT_camera_chess_center = None
RT_chess_base = None


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

        self.pipeline = rs.pipeline()

        self.pc = rs.pointcloud()

        self.align = rs.align(rs.stream.color)

        self.init_realsense(w, h)

    def init_realsense(self, w, h):
        config = rs.config()
        config.enable_stream(rs.stream.depth, w1, h1, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, w2, h2, rs.format.bgr8, 30)

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
                # color_image = aruco.drawDetectedMarkers(color_image.copy(), corners, ids)
                # cv2.imshow('Color', color_image)

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

                            if k == 0:
                                Oab = np.array(vertices[640 * y1 + x1])
                            elif k == 3:
                                Pxb = np.array(vertices[640 * y1 + x1])
                            elif k == 1:
                                Pyb = np.array(vertices[640 * y1 + x1])
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
                        test_point_1 = np.hstack((test_point, [1])).reshape(4, 1)
                        tmp = np.array(np.dot(RT_ab, test_point_1)).reshape(4)
                        # print(RT_ba)

            if RT_camera_chess_center is None:
                continue
            qr_sign = 0
            # print(RT_camera_chess_center)

            poses = self.openpose.predict(color_image)
            rendered_image = self.openpose.draw_poses(color_image, poses)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            wrist = self.keypoint_at(poses, 4)
            wrist_3d = vertices[w2 * wrist[1] + wrist[0]]
            wrist_3d_base = trans_camera_base(wrist_3d)
            print(wrist_3d_base)
            rospy.set_param("wrist", [float(wrist_3d_base[0]), float(wrist_3d_base[1]), float(wrist_3d_base[2])])

            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            text = 'FPS: {}'.format(int(1 / mean_time * 10) / 10)
            cv2.putText(rendered_image, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            # images = np.hstack((rendered_image, depth_colormap))
            cv2.imshow('Color and Depth', rendered_image)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    rs_pose = RealsensePose(OPENPOSE_PATH)
    rs_pose.run()
