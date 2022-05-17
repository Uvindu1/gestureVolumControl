import cv2
import numpy as np
import time
import handTrackingModule as htm
import math

###############################################
wCam, hCam = 640, 480
###############################################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime =0
vol = 0

detector = htm.handDetector(detectionCon=0.7)# mehide kale htm eka athule handDetector kiyana cls eken object ekak hedala eka detector ta asain kar
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!= 0:#meya danna hethuwa api arambayedi atha pennana kalinma code eka read karai evita lmList[2] kiyala nethinisa error dei, mulinma balano lmList ekak thiyeda kiyala
        #print(lmList[4][1], lmList[8][1])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        c1, c2 = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 3, cv2.FILLED)
        cv2.circle(img, (c1, c2), 10, (0, 0, 0),  cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, (30, 200),(400, 150))
        print(length)

        if length<30:
            cv2.circle(img, (c1, c2), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85,400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol)), (85, 400), (0, 255, 0), cv2.FILLED)


    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
