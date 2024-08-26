import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os
import sys

if __name__ == '__main__':


    # 确定图像的输入分辨率与帧率
    resolution_width = 640  # pixels
    resolution_height = 480  # pixels
    frame_rate = 15  # fps

    # 注册数据流，并对其图像
    align = rs.align(rs.stream.color)
    rs_config_1 = rs.config()
    rs_config_1.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    rs_config_1.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    rs_config_2 = rs.config()
    rs_config_2.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
    rs_config_2.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, 30)
    ### d435i
    #
    # rs_config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, frame_rate)
    # rs_config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, frame_rate)
    # check相机是不是进来了
    connect_device = []
    for d in rs.context().devices:
        print('Found device: ',
              d.get_info(rs.camera_info.name), ' ',
              d.get_info(rs.camera_info.serial_number))
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))


    if len(connect_device) < 2:
        print('Registrition needs two camera connected.But got one.')
        exit()

    # 确认相机并获取相机的内部参数
    pipeline1 = rs.pipeline()
    rs_config_1.enable_device('220422302842')
    # pipeline_profile1 = pipeline1.start(rs_config)
    pipeline1.start(rs_config_1)


    try:

        while True:

            # 等待数据进来
            frames1 = pipeline1.wait_for_frames()


            # 将进来的RGBD数据对齐
            aligned_frames1 = align.process(frames1)


            # 将对其的RGB—D图取出来
            color_frame1 = aligned_frames1.get_color_frame()
            depth_frame1 = aligned_frames1.get_depth_frame()



            color_image1 = np.asanyarray(color_frame1.get_data())
            depth_image1 = np.asanyarray(depth_frame1.get_data())



            depth_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image1, alpha=0.03), cv2.COLORMAP_JET)
            images1 = np.hstack((color_image1, depth_colormap1))


            cv2.imshow('RealSense1', images1)

            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline1.stop()
