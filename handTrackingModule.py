import cv2
import time
import math
import numpy as np
import mediapipe as mp

class HandDetector :
    def __init__(self ,mode=False , maxHand=2 , detectCon=0.5 ,trackCon=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpHand = mp.solutions.hands
        self.Hands = self.mpHand.Hands(self.mode,self.maxHand,1,self.detectCon,self.trackCon)
        # Sadrži način otkrivanja i način praćenja kada nemamo dostupnih informacija za praćenje, prelazi u način otkrivanja.
        # Hands = mpHand.Hands(False,2,min_tracking_confidence=0.5), može i tako, ali je već postavljeno kao zadano pa ih možemo ostaviti.
        # Imajte na umu da ova biblioteka (ruka) samo uzima RGB sliku.
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findhands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.processHand = self.Hands.process(imgRGB)
        # print(processHand.multi_hand_landmarks)
        if self.processHand.multi_hand_landmarks:  # Možete koristiti processHand.multi_hand_landmarks[0] ili [1] ovu ruku.
            for handLMS in self.processHand.multi_hand_landmarks:
                if draw :
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHand.HAND_CONNECTIONS)

    def findPosition(self ,img ,handsNo=0 ):
        self.lmlist=[]
        xList=[]
        ylist=[]
        bbox=[]
        if self.processHand.multi_hand_landmarks:  # Možete koristiti processHand.multi_hand_landmarks[0] ili [1] ovu ruku.
            try:
                myHand = self.processHand.multi_hand_landmarks[handsNo]
                for id, lm in enumerate(myHand.landmark):
                    # print(id, lm)  
                    # Ruke imaju 21 točku [0 do 20] i svaka točka označava neki dio ruke.
                    # Problem je što je 'lm' u decimalnom obliku i trebamo piksel npr.:(200 w, 300 h).
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)  # Sada imamo (id,cx,cy), pa možemo učiniti bilo što.
                    self.lmlist.append([id,cx,cy])
                    xList.append(cx)
                    ylist.append(cy)
                xmin,xmax = min(xList),max(xList)
                ymin,ymax = min(ylist),max(ylist)
                bbox=xmin,ymin,xmax,ymax
            except:
                pass

        return self.lmlist,bbox

    def fingersUp(self):
        fingers = []
        
        # Palac
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Prsti
        for id in range(1, 5):

            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length,[x1, y1, x2, y2, cx, cy]

def main():
    cap = cv2.VideoCapture(0)
    detector=HandDetector()
    while True:
        sucess, img = cap.read()
        detector.findhands(img)
        lmLIST,bbox = detector.findPosition(img)

        if len(lmLIST)!=0:
            fingers = detector.fingersUp()
            length,bbox = detector.findDistance(8,12,img)
            print(length)
            print(fingers)

        cv2.imshow("Output", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ =='__main__':
    main()