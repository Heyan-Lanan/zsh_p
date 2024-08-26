import pyrealsense2 as rs
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import rospy
from cv2 import aruco
from ultralytics import YOLO


w1 = 320
h1 = 240
w2 = 640
h2 = 480
RGB_img = None
Vertices = None
my_model = YOLO(r'/home/ustc/zsh_p/model/best.pt')
hand_detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
rospy.init_node('human_pose', anonymous=False, log_level=rospy.INFO, disable_signals=False)
RT_camera_chess_center = None
RT_chess_base = None


def trans_camera_base(p_3d):
    p_3d_4 = np.hstack((p_3d, [1])).reshape(4, 1)
    p_3d_chess = np.array(np.dot(RT_camera_chess_center, p_3d_4)).reshape(4)
    p_3d_base = np.array(np.dot(RT_chess_base, p_3d_chess)).reshape(4)[0: 3]
    return p_3d_base


class RealsensePose:
    def __init__(self, w=640, h=480):
        self.pipeline = rs.pipeline()
        self.pc = rs.pointcloud()
        self.align = rs.align(rs.stream.color)
        self.init_realsense(w, h)

    def init_realsense(self, w, h):
        config = rs.config()
        config.enable_device('f0233113')
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

            chess_available = []
            finger_12_3d = [0.0, 0.0, 0.0]
            hands, frame_markers = hand_detector.findHands(color_image.copy(), draw=True, flipType=True)
            if hands:
                # cv2.circle(color_image, hands[0]['lmList'][0][:2], 5, (0,0,255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][4][:2], 5, (0, 0, 255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][8][:2], 5, (0, 0, 255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][12][:2], 5, (0, 0, 255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][16][:2], 5, (0, 0, 255), -1)
                # cv2.circle(color_image, hands[0]['lmList'][20][:2], 5, (0, 0, 255), -1)
                # print(hands[0]['lmList'])
                x = hands[-1]['lmList'][12][0]
                y = hands[-1]['lmList'][12][1]
                finger_12_3d = vertices[w2 * int(y) + int(x)]
                # print(wrist_3d)
                if finger_12_3d[0] != 0.0 and finger_12_3d[1] != 0.0 and finger_12_3d[2] != 0.0:
                    finger_12_3d_base = trans_camera_base(finger_12_3d)
                    # print(wrist_3d_base)

            img_2 = color_image.copy()
            results = my_model(img_2, conf=0.3, device='0')
            boxes = results[0].boxes
            if len(boxes.conf) > 0:
                for index in range(len(boxes.conf)):
                    id = boxes.cls[index]
                    xyxy_ = boxes.xyxy[index]
                    xyxy = [int(item) for item in xyxy_]
                    center_x = int((xyxy[0] + xyxy[2]) / 2)
                    center_y = int((xyxy[1] + xyxy[3]) / 2)
                    chess_3d = vertices[w2 * int(center_y) + int(center_x)]
                    if chess_3d[0] != 0.0 and chess_3d[1] != 0.0 and chess_3d[2] != 0.0:
                        chess_3d_base = trans_camera_base(chess_3d)
                        chess_param = [float(chess_3d_base[0]), float(chess_3d_base[1]), float(chess_3d_base[2])]
                        rospy.set_param("chess_" + str(int(id.cpu().numpy())), chess_param)
                        text = '{:.3f}, {:.3f}, {:.3f}'.format(chess_param[0], chess_param[1], chess_param[2])
                        cv2.putText(img_2, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        if np.sqrt(np.sum((chess_3d - finger_12_3d) ** 2)) > 0.05:
                            chess_available.append(int(id.cpu().numpy()))

                    p1 = (xyxy[0], xyxy[1])
                    p2 = (xyxy[2], xyxy[1])
                    p3 = (xyxy[0], xyxy[3])
                    p4 = (xyxy[2], xyxy[3])
                    cv2.line(frame_markers, p1, p2, color=(0, 255, 0), thickness=2)
                    cv2.line(frame_markers, p2, p4, color=(0, 255, 0), thickness=2)
                    cv2.line(frame_markers, p4, p3, color=(0, 255, 0), thickness=2)
                    cv2.line(frame_markers, p3, p1, color=(0, 255, 0), thickness=2)
                    # text = 'id: {}, conf: {:.2f}'.format(int(id), conf)
                    text = 'id: {}'.format(int(id))
                    cv2.putText(frame_markers, text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            rospy.set_param("chess_available", chess_available)
            text = ''
            for item in chess_available:
                text += str(item) + ', '
            cv2.putText(frame_markers, text, (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            text = 'FPS: {}'.format(int(1 / mean_time * 10) / 10)
            cv2.putText(frame_markers, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.imshow('chess', frame_markers)
            # cv2.imshow('hand', frame_markers)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    rs_pose = RealsensePose()
    rs_pose.run()