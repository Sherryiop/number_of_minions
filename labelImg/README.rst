LabelImg
========

Made changes to the original LabelImg tool (from this repo here - https://github.com/tzutalin/labelImg/ ) to propagate the bounding boxes from one image to the other in a video extracted frame sequence, thereby reducing the effort to manually draw a bounding box on each and every image. This should help in annotating video sequences where the successive frames look very similar to the previous ones. To propagate the bounding box check the option in View menu - Propagate bounding box to the next image. Also an attempt is made to fix few issues with the orignal tool.

For installations and use follow the same instructions of the original LabelImg tool here as well.

## Installation
```
pip install pyqt5 lxml
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```
# pealse use python3.8+  