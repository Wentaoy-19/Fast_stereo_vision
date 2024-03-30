from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

SHOW_L = True
SHOW_R = True

# Load the YOLOv8 model
model_l = YOLO('yolov8n.pt') if SHOW_L else None
model_r = YOLO('yolov8n.pt') if SHOW_R else None

line1 = [(659, 0), (659, 914)]
line2 = [(958,0), (958, 914)]

# Open the video file
video_path_l = "132827_Cam_L_20231030_17521052.mp4"
video_path_r = "132830_Cam_R_20231030_17403079.mp4"
cap_l = cv2.VideoCapture(video_path_l) if SHOW_L else None
cap_r = cv2.VideoCapture(video_path_r) if SHOW_R else None

# Store the track history
track_history_l = defaultdict(lambda: [])
track_history_r = defaultdict(lambda: [])


if(SHOW_L):
    cv2.namedWindow("YOLOv8 Tracking1")
if(SHOW_R):
    cv2.namedWindow("YOLOv8 Tracking2")       
# cv2.setMouseCallback('YOLOv8 Tracking1',click_event)


def process_result(results,track_history, _results):
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    _track_ids = _results[0].boxes.id.int().cpu().tolist()
    annotated_frame = results[0].plot()
    for i in range(len(track_ids)):
        box = boxes[i]
        track_id = track_ids[i]
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        
        if(i < len(_track_ids)):
            # print((track_ids[i],_track_ids[i]))
            cv2.putText(annotated_frame, f'({_track_ids[i]})', (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
        
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)
    return annotated_frame

# jump frames
fps = 23.98
jump_second = 3
caml_i = 0
jump_frame = 63
while cap_l.isOpened() and caml_i<jump_frame:
    _,_ = cap_l.read()
    caml_i += 1


# Loop through the video frames
while cap_l.isOpened():
    # Read a frame from the video
    success_l, frame_l = cap_l.read()
    success_r, frame_r = cap_r.read()
    cv2.line(frame_l, line1[0], line1[1], (46,162,112), 3)
    cv2.line(frame_l, line2[0], line2[1], (46,162,112), 3)
    
    if success_l and success_r:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results_l = model_l.track(frame_l, persist=True)
        results_r = model_r.track(frame_r, persist=True)
        
        # processing tracking results
        annotated_frame_l = process_result(results_l,track_history_l, results_r)
        annotated_frame_r = process_result(results_r,track_history_r, results_l)
        
        cv2.imshow("YOLOv8 Tracking1", annotated_frame_l)
        cv2.imshow("YOLOv8 Tracking2", annotated_frame_r)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap_l.release()
cap_r.release()
cv2.destroyAllWindows()