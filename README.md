# summer-class
本專案將利用YOLOv8，辨識一張圖片中有幾隻小小兵  

  
若要了解更多YOLOv8相關資料，請參考: [YOLOv8 Github](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)  |  [YOLOv8 語法](https://docs.ultralytics.com/)
## 下載專案

#### 方法一
若有安裝[Git](https://ithelp.ithome.com.tw/articles/10322227)，在terminal確認要下載的目的地後，輸入以下文字:
```bash
git clone https://github.com/Sherryiop/summer-class.git --depth 1 
```

#### 方法二
直接點Download.zip下載

## 貼標
### 環境建立
在terminal將路徑指定到labellmg後輸入
  
```bash
pip install -r requirement.txt
```
### 確認物件名稱
請至predefined_classes.txt確認物件名稱及數量，文件路徑為labelImg\data\predefined_classes.txt

本次物件只有一個，所以predefined_classes.txt內容只有一行
```bash
minions
```
### 使用方法
![圖片1](https://github.com/user-attachments/assets/843f7075-90df-4f01-97e5-9f4ebed1f9a5)

## xml檔 -> txt檔
### 環境建立
由於labellmg輸出的程式為xml檔，但YOLO訓練只讀txt檔，需要進行檔案轉錄

路徑指定到XmlToTxt後，安裝環境
```bash
pip install -r requirements.txt
```
原始來源:[XmlToTxt Github](https://github.com/isabek/XmlToTxt/tree/master)

### 轉錄檔案
把要轉錄的xml檔案放置XmlToTxt\xml，如果沒有該資料夾，需要自己建立  

確認XmlToTxt\classes.txt 檔案內容跟summer-class\labelImg\data\predefined_classes.txt 一樣後，在terminal輸入
```bash
python xmltotxt.py -xml xml -out out
```
txt檔會在XmlToTxt\out

## YOLO模型訓練
### 環境建立
  
**環境要求:Python>=3.8**

請確認python>=3.8後，再安裝YOLOv8套件
```bash
pip install ultralytics
```
### 配置yaml檔
yaml檔為YOLO的訓練檔案，其格式如下

```bash
path:  # dataset root dir 
train:  # training set root dir
val:  # val set root dir
test: # test images (optional)

nc : 1 # the number of class

# Classes
names:
  0 : minions
```
### 模型訓練
**預訓練模型下載**:[YOLOv8](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)

**Dataset 格式**  

資料格式必須遵守以下結構
  ```
 count_minions
  ├── images
  │   ├── train
  │   └── val
  ├── labels
  │   ├── train
  │   └── val
  ```
**訓練指令**  

確認資料夾格式正確後，輸入訓練指令
```bash
yolo detect train data=yaml路徑 model=yolov8n.pt epochs=250 imgsz=640 patience=50 device=0 batch=-1
```

