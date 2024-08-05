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
import time

def box_center(left, top, right, bottom):
    width = right - left
    height = bottom - top
    center_x = left + int((right-left)/2)
    center_y = top + int((bottom-top)/2)
    
    return center_x, center_y

#===========================================================#
# 模型和影片路徑設定
#===========================================================#
model_path = r'C:\Users\Hi1pp\5G_YOLOv8\runs\detect\train12\weights\best.pt'
source = r'C:\Users\Hi1pp\5G_YOLOv8\datasets\count_minions\Minion.jpg'
output_path = r'C:\Users\Hi1pp\5G_YOLOv8\datasets\count_minions\output_minions_1.jpg'
# frame = cv2.imread(source)
model = YOLO(model_path)

#===========================================================#
# 區域框大小設定
#===========================================================#
current_region = None
counting_regions = [
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(20, 50), (500, 50), (500, 450), (20, 450)]),  # Polygon points
        # "counts": 0,
        "dragging": False,
        "region_color": (37, 80, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]

region_thickness = 5 #區域框線寬度
minions_x = 0
minions_y = 0
count_minions = 0
names = model.model.names

# results = model.predict(frame, save=True, device=0)
results = model(source)
            
for result in results:
    frame = result.plot()
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    for i, box in enumerate(result.boxes):
        left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=np.int32).squeeze()
        center_x, center_y = box_center(left, top, right, bottom)  # 每個物件框的中心點

        object_name = result.names[box.cls.item()]
        if  object_name == 'minions': 
            minions_x, minions_y = center_x, center_y
                      
            print('helloworld')
        count_minions += 1
        cv2.putText(frame, str(count_minions), (minions_x, minions_y), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255),3)         
        
    # if counting_regions[0]["polygon"].contains(Point(minions_x, minions_y)):
    #     count_minions += 1
    #     print('123')
    #     cv2.putText(frame, count_minions, (minions_x, minions_y), cv2.FONT_HERSHEY_COMPLEX, 50, (255,0,0),20)            
                            
    # 畫框 start
    # polygon_coords = np.array(counting_regions[0]["polygon"].exterior.coords, dtype=np.int32) #藍色框
    # cv2.polylines(frame, [polygon_coords], isClosed=True, color=counting_regions[0]['region_color'], thickness=region_thickness)
    #  畫框 end
    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

cv2.imwrite(output_path, frame)
cv2.imshow('Image Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()