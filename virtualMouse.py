import cv2
import autopy
import numpy as np
import handTrackingModule as htm

wCam , hCam = 640,480
wScr , hScr =autopy.screen.size()
frameR = 150
smootheing = 5
plocX , plocY = 0, 0
clocX , clocY = 0,0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detector = htm.HandDetector()

def Mouse(img):
    global frameR
    global smootheing
    global plocX
    global plocY
    global clocX
    global clocY
    global wScr
    global wCam
    global hScr
    global hCam

    # 1. Pronalazak ruke.
    detector.findhands(img)
    lmlist, bbox = detector.findPosition(img)

    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    # 2. Uzmi vrh kažiprsta i srednjeg prsta.
    if len(lmlist) != 0:
        Xindex, Yindex = lmlist[8][1], lmlist[8][2]
        Xmidel, Ymidel = lmlist[12][1], lmlist[12][2]

        # 3. Provjeri koji je podignut.
        fingers = detector.fingersUp()

        # 4. Kažiprst: način kretanja.
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Koordinacija položaja (cam: 640 × 480) - (screen: 2560 × 1600).
            xMOUSE = np.interp(Xindex, (frameR, wCam - frameR), (0, wScr))
            yMOUSE = np.interp(Yindex, (frameR, hCam - frameR), (0, hScr))

            # 6. Uglađivanje vrijednosti.
            clocX = plocX + (xMOUSE - plocX) / smootheing
            clocY = plocY + (yMOUSE - plocY) / smootheing

            # 7. Pokret kursora.
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (Xindex, Yindex), 15, (20, 180, 90), cv2.FILLED)
            plocY, plocX = clocY, clocX

        # 8. Kažiprst i srednji prst su gore: način rada za klikanje.
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. Pronalaženje udaljenosti.
            length, bbox = detector.findDistance(8, 12, img)
            #print(length)

            # 10. Klik ako je udaljenost odgovarajuća.
            if length < 40:
                autopy.mouse.click()
    return img


def main():
    while True:
        sucess, img = cap.read()
        img = cv2.flip(img, 1)

        img = Mouse(img)

        # 11. Prikaz/Ispis.

        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=='__main__':
    main()