from ultralytics import YOLO
import cv2
import numpy as np
import math

count_frame = 0
model_path = 'runs/detect/train10/weights/best.pt'
video_path = 'datasets/5G/Abnormal_1.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO(model_path)
# model.predict('datasets/5G/Abnormal_1.mp4',classes=[0,1,2], save = True)
# model.predict(video_path, classes=[0,1,2], save = True, device = 0)
# print(f'我要測試{model.names[0]}')

fourcc = cv2.VideoWriter_fourcc(*'XVID') #XVID
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('datasets/5G/output.mp4',fourcc, 30.0, size)

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



object_num = 0
count_frame= 0
delay = 20
distance_threshold = 500
brush_boxcenter_coord = []
screw_boxcenter_coord = []
sticker_boxcenter_coord = []
fps = 30
boxcenter_coord = []
init_coord = []
action_brush = True
action_screw = True

action_queue_brush = []
action_queue_screw = []
count_time = [0,0]

while True:
    ret, frame = cap.read()

    if ret == True: 
        boxcenter_coord = []
        results = model.predict(frame, save=True)
        if count_frame >= 20: #超過20楨才開始算物件中心點
            for result in results:
                new_frame = result.plot()
                new_frame = cv2.cvtColor(new_frame,cv2.COLOR_BGR2RGB)
                for i, box in enumerate(result.boxes):
                    left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=np.int32).squeeze()
                    center_x, center_y = box_center(left, top, right, bottom)  # 每個物件框的中心點
                    # if i == 0: screw_boxcenter_coord.append([center_x, center_y])
                    # elif i == 1: brush_boxcenter_coord.append([center_x, center_y])
                    object_name = result.names[box.cls.item()]
                    if  object_name == 'scewr_driver':
                        screw_boxcenter_coord.append([center_x, center_y])
                    elif object_name == 'brush':
                        brush_boxcenter_coord.append([center_x, center_y])
                    cv2.circle(new_frame, (center_x, center_y), 5,(0,0,255),-1) #畫出每個物件框的中心點



                if len(init_coord) == 0 : #給定物件初始位置
                    # copy = boxcenter_coord.copy()
                    # init_coord = copy   
                    init_coord.append(screw_boxcenter_coord[0])
                    init_coord.append(brush_boxcenter_coord[0])
                else :
                    init_coord[0] = check_init_coord(init_coord[0],screw_boxcenter_coord, 20) #螺絲刀放回原位時若有晃動則改變初始位置
                    init_coord[1] = check_init_coord(init_coord[1],brush_boxcenter_coord, 20) #刷具放回原位時若有晃動則改變初始位置
                    for circle in init_coord:
                        cv2.circle(new_frame, circle, 50, (255,0,0), 5)

                if action_brush :
                    action_brush = check_leave_init_coord(brush_boxcenter_coord, init_coord[1], 10, distance_threshold )
                    if action_brush == False:
                        count_time[1] = 0 
                        action_queue_brush.append(0)
                        action_brush = True
                    else :
                        count_time[1] += round(1/fps,3)
                        action_queue_brush.append(1)
                if action_screw :
                    action_screw = check_leave_init_coord(screw_boxcenter_coord, init_coord[0], 10, distance_threshold )
                    if action_screw == False:
                        count_time[0] = 0 
                        action_queue_screw.append(0)
                        action_screw = True
                    else :
                        count_time[0] += round(1/fps,3)
                        action_queue_screw.append(1)

                print(f'box的名字{result.names}')
                cv2.putText(new_frame, 'Action:', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2) 
                cv2.putText(new_frame, 'screw driver:' + str(round(count_time[0],3)) + 's', (50,100), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),2) 
                cv2.putText(new_frame, 'brush:' + str(round(count_time[1],3)) + 's', (50,150), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),2) 
                new_frame = cv2.cvtColor(new_frame,cv2.COLOR_RGB2BGR)
                out.write(new_frame)
                cv2.imshow('video', new_frame)

        print(f'brush_coord is {action_queue_screw}\n\n')
        # print(f'init_coord is {init_coord}')
        # print(f'刷具的動作{action_queue_screw}')
        
        count_frame += 1
    key = cv2.waitKey(1)
    if (key==27):# 按esc結束影片
        break 
  
cap.release()
cv2.destroyAllWindows()

