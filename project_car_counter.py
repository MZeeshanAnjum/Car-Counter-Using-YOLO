from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *
 
 
 #THIS IS THE COCOUNT CLASS USED TO TRAIN THE MODEL
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

a=cv2.VideoCapture(r"D:\OpenCV Projects\YOLO\cars.mp4")
model=YOLO(r"D:\OpenCV Projects\YOLO_weights\yolov8l.pt")
mask=cv2.imread(r"D:\OpenCV Projects\YOLO\mask.png")
# TRACKING
#IOU means RATIO OF INTERSECTION OVER UNION
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits = [250,350,680,350]
count=[]
while True:
    success,img=a.read()
    masked_img=cv2.bitwise_and(img,mask)# APPLY THE MASK ON THE STREAM
    graphics_car=cv2.imread(r"D:\OpenCV Projects\YOLO\graphics_car.png",cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(img,graphics_car,(0,0))
    results=model(masked_img,stream=True)
    detections=np.empty((0,5))# THIS IS FOR THE TRACKER, WE ADD THE [X1,Y1,X2,Y2,CONF] IN THIS NUMPY ARRAY
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            conf = math.ceil((box.conf[0] * 100)) / 100 #CALCULATING CONFIDENCE
            # Class Name
            cls = int(box.cls[0]) # CALCULATE THE CLASS
            current_class=classNames[cls]
            if current_class=="car" or "bus" or "motorbike"or "truck"  and conf > 0.3:
            #THIS COMMNAD FIXES THE TEXT WITH THE BOUNDING BOX
                #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                #cvzone.putTextRect(img, f'{current_class} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=3)
                current_list=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,current_list))
                
                
    #cv2.imshow("img",img)
    tracker_result=tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
    #ADD OVERALY IMAGE IN CVZONE IE: ADD PICTURE IN THE LIVE STREAM
    
    #imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    #img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    
    for result in tracker_result:
        x1,y1,x2,y2,id=result#ID TRACKER BIUNDING BOX
        #print(results)
        x1,y1,x2,y2,id=int(x1),int(y1),int(x2),int(y2),int(id)
        w,h=x2-x1,y2-y1# CALCULATE WIDTH AND HEIGHT
        cx,cy=x1+w//2,y1+h//2
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)# ID TRACKER BIUNDING BOX
        cvzone.putTextRect(img, f'Car ID {id} ', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=3)
        cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        if (limits[0]<cx<limits[2] and limits[1]-15 <cy< limits[3]+15):
            if(count.count(id)==0):
                count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (50, 50, 50), 5)
        cv2.putText(img,f"{len(count)}",(250,100),cv2.FONT_HERSHEY_PLAIN,3,(50,50,255),7)
        #cvzone.putTextRect(img, f'{current_class} {id} ', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=3)
    cv2.imshow("masked_img",img)
    key=cv2.waitKey(1)
    if key ==ord("q"):
        break
    
    