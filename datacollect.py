import cv2
from cvzone.HandTrackingModule import HandDetector
#from Hand_Tracking_Module import HandDetector
import numpy as np
import math
import time
cap= cv2.VideoCapture(0)
detector= HandDetector(maxHands=1)

offset=20
imgSize=300
foldr="data/C"
count=0

while True:
    success, img =cap.read()
    hands, img =detector.findHands(img)
    if hands:
        hand= hands[0]
        x, y, w, h = hand['bbox']
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        
        imgCrop =img[y-offset : y+h+offset , x-offset : x+w+offset]

        imgCropShape= imgCrop.shape      
        

        ratio=h/w
        if ratio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            resize= cv2.resize(imgCrop, (wCal, imgSize))
            imgresizeShape= resize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap]=resize

        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            resize= cv2.resize(imgCrop, (imgSize, hCal))
            imgresizeShape= resize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = resize


        cv2.imshow("imageCrop",imgCrop)
        cv2.imshow("imageWhite",imgWhite)

    cv2.imshow("image", img)
    key=cv2.waitKey(1)
    if key==ord('s'):
        count+=1
        cv2.imwrite(f'{foldr}/Image_{time.time()}.jpg',imgWhite)
        print(count)
    