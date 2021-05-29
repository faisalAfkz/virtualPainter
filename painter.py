import cv2
import mediapipe as mp
import time
import numpy as np
import os


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
        else:
            fingers.append(0)
        return fingers

##################################
brushThickness = 10
eraserThikness = 80
folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0 , 255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = handDetector(detectionCon=0.8)
xp,yp = 0 , 0

imgCanvas = np.zeros((720,1280,3),np.uint8)


while True:
    #1. Import image
    success, img = cap.read()
    img = cv2.flip(img,1)

    #2. Find land mark
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0 :
        #print(lmList)

        #tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #3. Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        #4. If selction mode - 2 fingers
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("selection mode")

            #checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv2.FILLED)
        #5. if drawing - index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1),15, drawColor, cv2.FILLED)
            print("Drawing mode")

            if xp == 0 and yp == 0:
                xp,yp = x1 , y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThikness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThikness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)

    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    #setting header image

    img [0:125,0:1280] = header
    #img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


