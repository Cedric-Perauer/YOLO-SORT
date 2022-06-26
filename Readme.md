## Simple YOLOV5 + Sort Tracker

Combination of YOLOV5 + Sort Tracker in Pytorch for runtime constraint enviornments. Based on [Yolov5](https://github.com/ultralytics/yolov5) and [SORT tracker](https://github.com/abewley/sort) repos. Runs with about 38 FPS on M1 Air using YOLOv5n.pt with image size 320. 

For slower but more accurate tracking, see methods such as: 
    
1) [DeepSort](https://github.com/nwojke/deep_sort)     
2) [StrongSort](https://github.com/dyhBUPT/StrongSORT)

## Requirements 

For required pip packages check [Yolov5](https://github.com/ultralytics/yolov5) and [SORT tracker](https://github.com/abewley/sort). 

## Usage 
1) Download [MOT dataset](https://www.kaggle.com/datasets/kmader/mot2d-2015?resource=download)
2) Create sym link ```ln -s /path/to/MOT2015_challenge/data/2DMOT2015 /path/to/repo/mot_benchmark```
3) Run : ```python track.py --weights <insert yolo weights here as yolo5n/s/m/.pt --imgsz <insert image size>```



