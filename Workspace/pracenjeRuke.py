import cv2  # "pip install opencv-python"
import time
import math
import numpy as np
import mediapipe as mp  # "pip install mediapipe"


class detektorRuke():
    def __init__(self, mode=False, maxHands=2, detectionCon=False, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def pronadjiRuku(self, slika, draw=True):    # Pronalazak ruke u okviru
        slikaRGB = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(slikaRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(slika, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return slika

    def pronadjiPoziciju(self, slika, handNo=0, draw=True):   # Dohvaćanje položaja ruke
        xLista = []
        yLista = []
        bbox = []
        self.lmLista = []
        if self.results.multi_hand_landmarks:
            mojaRuka = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(mojaRuka.landmark):
                v, s, c = slika.shape
                cx, cy = int(lm.x * s), int(lm.y * v)
                xLista.append(cx)
                yLista.append(cy)
                self.lmLista.append([id, cx, cy])
                if draw:
                    cv2.circle(slika, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xLista), max(xLista)
            ymin, ymax = min(yLista), max(yLista)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(slika, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmLista, bbox

    def prstPodignut(self):    # Provjeravanje je li prst podignut
        prsti = []
        # Palac
        if self.lmLista[self.tipIds[0]][1] > self.lmLista[self.tipIds[0] - 1][1]:
            prsti.append(1)
        else:
            prsti.append(0)

        # Prsti
        for id in range(1, 5):

            if self.lmLista[self.tipIds[id]][2] < self.lmLista[self.tipIds[id] - 2][2]:
                prsti.append(1)
            else:
                prsti.append(0)

        # ukupnoPrsti = prsti.count(1)

        return prsti

    # Pronalaženje udaljenosti između dvaju prstiju
    def pronadjiUdaljenost(self, p1, p2, slika, draw=True, r=15, t=3):
        x1, y1 = self.lmLista[p1][1:]
        x2, y2 = self.lmLista[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(slika, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(slika, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(slika, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(slika, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        duljina = math.hypot(x2 - x1, y2 - y1)

        return duljina, slika, [x1, y1, x2, y2, cx, cy]


def main():
    pVrijeme = 0
    cVrijeme = 0
    sadrzaj = cv2.VideoCapture(1)
    detektor = detektorRuke()
    while True:
        success, slika = sadrzaj.read()
        slika = detektor.pronadjiRuku(slika)
        lmLista, bbox = detektor.pronadjiPoziciju(slika)
        if len(lmLista) != 0:
            print(lmLista[4])

        cVrijeme = time.time()
        fps = 1 / (cVrijeme - pVrijeme)
        pVrijeme = cVrijeme

        cv2.putText(slika, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Slika", slika)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
