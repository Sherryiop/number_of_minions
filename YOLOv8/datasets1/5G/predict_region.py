from ultralytics import YOLO
import cv2
import numpy as np
import math

import argparse
from collections import defaultdict
from pathlib import Path

from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

def box_center(left, top, right, bottom):
    width = right - left
    height = bottom - top
    center_x = left + int((right-left)/2)
    center_y = top + int((bottom-top)/2)
    
    return center_x, center_y

def calculate_distance(center1, center2):
    x1, y1 = center1[0], center1[1]
    x2, y2 = center2[0], center2[1]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2)*0.5

def check_leave_init_coord(object_queues, init_coords, delay, distance_threshold):
    count = 0
    if len(object_queues) >= delay: 
        for i in range(len(object_queues)-delay, len(object_queues)):
            # print(f'object queues is {type(object_queues[i][0])}')
            distence = calculate_distance(object_queues[i], init_coords)
            if (distence >= distance_threshold).any():
                count += 1
            else:
                count = 0
            if count >= delay:
                return True
    return False

def check_init_coord(init_coord, boxcenter_coord, delay):
    count = 0
    if len(boxcenter_coord) >= delay:
        for i in range(len(boxcenter_coord)-delay, len(boxcenter_coord)):
            # print(f'距離:{calculate_distance(init_coord, boxcenter_coord[i])}')
            if calculate_distance(init_coord, boxcenter_coord[i]) >= 20  and calculate_distance(init_coord, boxcenter_coord[i]) < 350:
                count += 1
            else: 
                count = 0
            if count >= delay :
                init_coord = boxcenter_coord[i]
                count = 0
    return init_coord

def mouse_callback(event, x, y, flags, param):
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False

#===========================================================#
# 模型和影片路徑設定
#===========================================================#
# model_path = 'runs/detect/train10/weights/best.pt'
model_path = r"C:\Users\Hi1pp\5G_YOLOv8\yolov8n.pt"
video_path = r"C:\Users\Hi1pp\5G_YOLOv8\datasets\5G\Abnormal_1.mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO(model_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID') #XVID
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('datasets/5G/output.mp4',fourcc, 30.0, size)

#===========================================================#
# 區域框大小設定
#===========================================================#
current_region = None
counting_regions = [
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(200, 50), (1500, 50), (1500, 500), (200, 500)]),  # Polygon points
        # "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]

count_frame = 0
object_num = 0
delay = 20
distance_threshold = 500
brush_boxcenter_coord = []
screw_boxcenter_coord = []
fps = 30
boxcenter_coord = []
init_coord = []
action_brush = True
action_screw = True

action_queue_brush = []
action_queue_screw = []
count_time = [0,0]
line_thickness = 2
region_thickness = 5 #區域框線寬度

names = model.model.names
while True:
        ret, frame = cap.read()
        
        if ret == True: 
            boxcenter_coord = []
            results = model.predict(frame, save=True)
           
            if count_frame >= 20: #超過20楨才開始算物件中心點
                for result in results:
                    frame = result.plot()
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    for i, box in enumerate(result.boxes):
                        left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=np.int32).squeeze()
                        center_x, center_y = box_center(left, top, right, bottom)  # 每個物件框的中心點

                        object_name = result.names[box.cls.item()]
                        if  object_name == 'scewr_driver':
                            screw_boxcenter_coord.append([center_x, center_y])
                        elif object_name == 'brush':
                            brush_boxcenter_coord.append([center_x, center_y])
                        cv2.circle(frame, (center_x, center_y), 5,(0,0,255),-1) #畫出每個物件框的中心點

                        # Check if detection inside region
                        # 有東西在這個區域才會進判斷
                        for region in counting_regions:
                            if region["polygon"].contains(Point((center_x, center_y))):
                                cv2.putText(frame, 'It is tests', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)
                    # 畫框
                    for region in counting_regions:
                        region_label = str(region["counts"])
                        region_color = region["region_color"]
                        region_text_color = region["text_color"]

                        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
                        centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

                        text_size, _ = cv2.getTextSize(region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness)
                        text_x = centroid_x - text_size[0] // 2
                        text_y = centroid_y + text_size[1] // 2
                        cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)


                    # for region in counting_regions:  # Reinitialize count for each region
                    #     region["counts"] = 0

            #         if len(init_coord) == 0 : #給定物件初始位置
            #             init_coord.append(screw_boxcenter_coord[0])
            #             init_coord.append(brush_boxcenter_coord[0])
            #         else :
            #             init_coord[0] = check_init_coord(init_coord[0],screw_boxcenter_coord, 20) #螺絲刀放回原位時若有晃動則改變初始位置
            #             init_coord[1] = check_init_coord(init_coord[1],brush_boxcenter_coord, 20) #刷具放回原位時若有晃動則改變初始位置
            #             for circle in init_coord:
            #                 cv2.circle(frame, circle, 50, (255,0,0), 5)

            #         if action_brush :
            #             action_brush = check_leave_init_coord(brush_boxcenter_coord, init_coord[1], 10, distance_threshold )
            #             if action_brush == False:
            #                 count_time[1] = 0 
            #                 action_queue_brush.append(0)
            #                 action_brush = True
            #             else :
            #                 count_time[1] += round(1/fps,3)
            #                 action_queue_brush.append(1)
            #         if action_screw :
            #             action_screw = check_leave_init_coord(screw_boxcenter_coord, init_coord[0], 10, distance_threshold )
            #             if action_screw == False:
            #                 count_time[0] = 0 
            #                 action_queue_screw.append(0)
            #                 action_screw = True
            #             else :
            #                 count_time[0] += round(1/fps,3)
            #                 action_queue_screw.append(1)

            #         cv2.putText(frame, 'screw driver:' + str(round(count_time[0],3)) + 's', (50,100), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),2) 
            #         cv2.putText(frame, 'brush:' + str(round(count_time[1],3)) + 's', (50,150), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),2) 
                    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    out.write(frame)
                    cv2.imshow('video', frame)

            
            count_frame += 1
        
        # if view_img:
        #     if count_frame == 1:
        #         cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
        #         cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
        #     cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        # if save_img:
        #     out.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        key = cv2.waitKey(1)
        if (key==27):# 按esc結束影片
            break 
  
del count_frame
out.release()
cap.release()
cv2.destroyAllWindows()
