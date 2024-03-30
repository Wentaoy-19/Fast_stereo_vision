from collections import defaultdict

import cv2
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

from ultralytics import YOLO
import math
import stereoconfig_gopro as stereoconfig   # Camera parameters
from stereo import getRectifyTransform, rectifyImage, preprocess, undistortion, draw_line, stereoMatchSGBM
import torch

# from hitnet_impl.hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig

def draw_disparity(disparity_map):

    disparity_map_ = disparity_map.astype(np.float32)
    print("max: ", np.max(disparity_map_))
    print("min: ", np.min(disparity_map_))
    norm_disparity_map = (255*((disparity_map_-np.min(disparity_map_))/(np.max(disparity_map_) - np.min(disparity_map_))))

    # return cv2.cvtColor(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)


import sgm_cuda_py


SHOW_L = True
SHOW_R = True

model_l = YOLO('../yolov8n.pt') if SHOW_L else None

# model_type = ModelType.middlebury
# model_path = "hitnet_impl/models/middlebury_d400.pb"

# # Store baseline (m) and focal length (pixel)
# camera_config = CameraConfig(0.1, 320)
# max_distance = 5

# # Initialize model
# hitnet_depth = HitNet(model_path, model_type, camera_config)


fps = 24
spf = 1/fps
SAVE = False
# if(SAVE):
    # fourcc = cv2.VideoWriter_fourcc(*'mp4')
    # out_cap = cv2.VideoWriter('speed_detection.mp4',-1, 24, (1280,720))



# Open the video file
# video_path_l = "left_dec.MP4"
# video_path_r = "right_dec.MP4"
video_path_l = "Cam_L.mp4"
video_path_r = "Cam_R.mp4"
cap_l = cv2.VideoCapture(video_path_l) if SHOW_L else None
cap_r = cv2.VideoCapture(video_path_r) if SHOW_R else None

# if(SHOW_L):
#     cv2.namedWindow("Camera_L")
# if(SHOW_R):
#     cv2.namedWindow("Camera_R")


jump_frames = 5
cap_l.set(cv2.CAP_PROP_POS_FRAMES, jump_frames + 2000)
cap_r.set(cv2.CAP_PROP_POS_FRAMES, 2000)

height, width = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
config = stereoconfig.stereoCamera()
map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # Obtain the mapping matrix used for distortion correction and stereo correction and the reprojection matrix used to calculate pixel space coordinates


# SGBM configs
img_channels = 1
blockSize = 3
paraml = {'minDisparity': 0,
            'numDisparities': 512,
            'blockSize': blockSize,
            'P1': 4 * img_channels * blockSize ** 2,
            # 'P1': 7,
            'P2': 32 * img_channels * blockSize ** 2,
            # 'P2': 84,
            'disp12MaxDiff': 1,
            'preFilterCap': 63,
            'uniquenessRatio': 15,
            'speckleWindowSize': 100,
            'speckleRange': 1,
            'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
            }

# SGBM Object 
# left_matcher = cv2.StereoSGBM_create(0, 128, 3, img_channels * blockSize ** 2, 32 * img_channels * blockSize ** 2, 1, 63, 15, 100, 1, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
left_matcher = cv2.StereoSGBM_create(**paraml)
paramr = paraml
paramr['minDisparity'] = -paraml['numDisparities']
right_matcher = cv2.StereoSGBM_create(**paramr)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 5)
plt.legend()
trucks = {}
truck_labels = {}


fps_tester = cv2.getTickCount()
fps_tester_cnt = 0

while cap_l.isOpened() and cap_r.isOpened():
    # print("Frame: %d" % cap_l.get(cv2.CAP_PROP_POS_FRAMES))
    # Read a frame from the video
    success_l, frame_l = cap_l.read()
    success_r, frame_r = cap_r.read()
    # cv2.line(frame_l, (791, 462), (1013, 300), (46,162,112), 3)
    # cv2.line(frame_l, (61,462),(750,300), (46,162,112), 3)
    
    if success_l and success_r:
        # cv2.imshow("Camera_L", frame_l)
        # cv2.imshow("Camera_R", frame_r)
        # if(SAVE):
            # out_cap.write(annotated_frame_l)
        
        # print("frame_l shape: ", frame_l.shape)
        # frame_l = cv2.resize(frame_l, (960, 540))
        # frame_r = cv2.resize(frame_r, (960, 540))

        iml_rectified, imr_rectified = rectifyImage(frame_l, frame_r, map1x, map1y, map2x, map2y)
        # iml_rectified, imr_rectified = frame_l, frame_r # skip rectification
        results_l = model_l.track(iml_rectified, persist=True, verbose=False)
        annotated_frame_l = results_l[0].plot()
        line = draw_line(annotated_frame_l, imr_rectified)

        # stereo matching
        # frame_l_, frame_r_ = preprocess(frame_l, frame_r)  # preprocess with lights (optional)
        frame_l_, frame_r_ = frame_l, frame_r

        iml_rectified_l, imr_rectified_r = rectifyImage(frame_l_, frame_r_, map1x, map1y, map2x, map2y)
        # iml_rectified_l, imr_rectified_r = frame_l_, frame_r_
        iml_rectified_l = cv2.cvtColor(iml_rectified_l, cv2.COLOR_BGR2GRAY)
        imr_rectified_r = cv2.cvtColor(imr_rectified_r, cv2.COLOR_BGR2GRAY)
        iml_rectified_l = cv2.resize(iml_rectified_l, (960, 540))
        imr_rectified_r = cv2.resize(imr_rectified_r, (960, 540))

        sgbm_time_start = cv2.getTickCount()
        disparity_left_sgbm = left_matcher.compute(iml_rectified_l, imr_rectified_r)
        # iml_rectified_l_torch = torch.from_numpy(iml_rectified_l).cuda()
        # imr_rectified_r_torch = torch.from_numpy(imr_rectified_r).cuda()
        # disparity_left_sgbm = sgm_cuda_py.sgm_cuda(iml_rectified_l_torch, imr_rectified_r_torch, 7, 84)
        # disparity_left_hitnet = hitnet_depth(iml_rectified_l, imr_rectified_r)

        sgbm_time_end = cv2.getTickCount()
        # print("Time: ", (sgbm_time_end - sgbm_time_start) / cv2.getTickFrequency() * 1000, "ms")

        # disparity_left_sgbm = disparity_left_sgbm.cpu().numpy().astype(np.float32) / 128.
        disparity_left_sgbm = (disparity_left_sgbm).astype(np.float32) / 16.

        # cv2.imshow("HitNet Disparity", draw_disparity(disparity_left_hitnet))
        cv2.imshow("SGBM Disparity", draw_disparity(disparity_left_sgbm))

        # points_3d = cv2.reprojectImageTo3D(disparity_left_hitnet, Q)

        # boxes = results_l[0].boxes.xywh.cpu()
        # track_ids = results_l[0].boxes.id.int().cpu().tolist()

        # for box, track_id in zip(boxes, track_ids):
        #     x, y, w, h = box
        #     center = (int(x) + int(w/2), int(y) + int(h/2))
        #     center_3d = points_3d[center[1], center[0]]
        #     center_3d = center_3d / 1000
        #     # print("track_id: ", track_id)
        #     # print("center: ", center)
        #     # print("3D point: ", center_3d)
        #     # center_3d[2] = 0 # TODO: remove this line after fixing the z-axis
        #     if track_id in trucks:
        #         trucks[track_id].set_data((center_3d[0], center_3d[1]))
        #         trucks[track_id].set_3d_properties(center_3d[2])
        #         truck_labels[track_id].set_position((center_3d[0], center_3d[1]))
        #         truck_labels[track_id].set_3d_properties(center_3d[2], zdir='z') 
        #     else:
        #         trucks[track_id], = ax.plot([center_3d[0]], [center_3d[1]], [center_3d[2]], 'o')
        #         truck_labels[track_id] = ax.text(center_3d[0], center_3d[1], center_3d[2], f'{track_id}')
        #     plt.draw()
        #     plt.pause(0.001)

        fps_tester_cnt += 1
        if (cv2.getTickCount() - fps_tester ) / cv2.getTickFrequency() > 3:
            print("FPS: ", fps_tester_cnt / ((cv2.getTickCount() - fps_tester ) / cv2.getTickFrequency()))
            fps_tester = cv2.getTickCount()
            fps_tester_cnt = 0

        cv2.imshow("Verify Image", line)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap_l.release()
cap_r.release()
# if(SAVE):
    # out_cap.release()
cv2.destroyAllWindows()