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
model_path = # YOLO model path
source = # test picture path
output_path =  # picture output path
model = YOLO(model_path)
#===========================================================#

region_thickness = 5 #區域框線寬度
minions_x = 0
minions_y = 0
count_minions = 0
names = model.model.names

results = model(source)
            
for result in results:
    frame = result.plot()
    for i, box in enumerate(result.boxes):
        left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=np.int32).squeeze()
        center_x, center_y = box_center(left, top, right, bottom)  # 每個物件框的中心點

        object_name = result.names[box.cls.item()]
        if  object_name == 'minions': 
            minions_x, minions_y = center_x, center_y
            count_minions += 1
        cv2.putText(frame, str(count_minions), (minions_x, minions_y), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255),3)         
        
cv2.imwrite(output_path, frame)
cv2.imshow('Image Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()