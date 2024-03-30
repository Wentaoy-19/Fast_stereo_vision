from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import math


SHOW_L = True

# Load the YOLOv8 model
model_l = YOLO('yolov8n.pt') if SHOW_L else None

# line1 = [(659, 0), (659, 914)]
# line2 = [(958,0), (958, 914)]

fps = 24
spf = 1/fps
real_length = 9.144 
pixel_length = 958 - 659
pp_length = real_length/pixel_length
SAVE = True 
if(SAVE):
    # fourcc = cv2.VideoWriter_fourcc(*'mp4')
    out_cap = cv2.VideoWriter('speed_detection.mp4',-1, 24, (1280,720))


# Open the video file
video_path_l = "132827_Cam_L_20231030_17521052.mp4"
cap_l = cv2.VideoCapture(video_path_l) if SHOW_L else None

# Store the track history
track_history_l = defaultdict(lambda: [])
speed_history_l = defaultdict(lambda: [])



if(SHOW_L):
    cv2.namedWindow("YOLOv8 Tracking1")

def get_line_equation(p1,p2):
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0]*p2[1] - p2[0]*p1[1]
    return (A,B,C)

class line_curve:
    def __init__(self,A,B,C) -> None:
        self.A = A
        self.B = B
        self.C = C
        pass
    def get_x(self,y):
        return (-self.C-self.B*y)/self.A
    def get_y(self,x):
        return (-self.C-self.A*x)/self.B
    def get_v(self,p):
        return self.A*p[0] + self.B*p[1] + self.C 

def within_2line(line1,line2,p):
    v1 = line1.get_v(p)
    v2 = line2.get_v(p)
    return v1*v2>=0

def within_range(x,y):
    if(x>=659 and x<=958):
        return True
    else:
        return False
    

A1,B1,C1= get_line_equation((913,373),(957, 341))
A2,B2,C2= get_line_equation((656,322),(571, 342))
line1 = line_curve(A1,B1,C1)
line2 = line_curve(A2,B2,C2)


def cal_speed(track_id,track, speed_history):
    if(len(track)>=2):
        p1 = track[-1]
        p2 = track[-2]
        s =  abs(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)*pp_length/spf)   
        speed_history[track_id].append(s) 
        return s
    else:
        return None

def avg_spd(track_id, speed_history):
    if(track_id in speed_history):
        print(len(speed_history[track_id]))
        speed = 9/(spf*(len(speed_history[track_id])+1))*3.6
        # speed = sum(speed_history[track_id])/len(speed_history[track_id])
        return speed
    return None

def process_result(results,track_history, speed_history):
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    annotated_frame = results[0].plot()
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        if(within_2line(line1,line2,(float(x),float(y)+0.5*float(h)))):
            track = track_history[track_id]
            track.append((float(x),float(y)+0.5*float(h)))  # x, y center point
            cal_speed(track_id,track, speed_history)
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
            # Draw the tracking lines
            # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)
        else:
            speed = avg_spd(track_id, speed_history)
            if(speed!=None):
                cv2.putText(annotated_frame, f'{round(speed,3)}', (int(x),int(float(y)+0.5*float(h))),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
            continue
    return annotated_frame



# jump frames
fps = 23.98
jump_second = 3
caml_i = 0
while cap_l.isOpened() and caml_i<fps*jump_second:
    _,_ = cap_l.read()
    caml_i += 1


# Loop through the video frames
while cap_l.isOpened():
    # Read a frame from the video
    success_l, frame_l = cap_l.read()
    cv2.line(frame_l, (791, 462), (1013, 300), (46,162,112), 3)
    cv2.line(frame_l, (61,462),(750,300), (46,162,112), 3)
    
    if success_l :
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results_l = model_l.track(frame_l, persist=True)
        
        # processing tracking results
        annotated_frame_l = process_result(results_l,track_history_l,speed_history_l)
        
        cv2.imshow("YOLOv8 Tracking1", annotated_frame_l)
        if(SAVE):
            out_cap.write(annotated_frame_l)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap_l.release()
if(SAVE):
    out_cap.release()
cv2.destroyAllWindows()