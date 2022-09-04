import cv2
import time
import autopy
import numpy as np
import pracenjeRuke as pr

# Incijalizacija i deklaracija varijabli
pVrijeme = 0               # Koristi se za izračunavanje fps-a
sirina = 640             # Širina kamere
visina = 480            # Visina kamere
frekvencijaOkvira = 100            # Brzina promjene okvira
izgladjivanje = 8         # Faktor izglađivanja
prethodni_x, prethodni_y = 0, 0   # Prethodne koordinate
trenutni_x, trenutni_y = 0, 0   # Trenutne koordinate

sadrzaj = cv2.VideoCapture(0)   # Dohvaćanje videosadržaja s web kamere
sadrzaj.set(3, sirina)           # Podešavanje veličine
sadrzaj.set(4, visina)

detektor = pr.detektorRuke(maxHands=1)                  # Otkrivanje ruku/e
# Dohvaćanje veličine zaslona
sirinaZaslona, visinaZaslona = autopy.screen.size()
while True:
    success, slika = sadrzaj.read()
    # Pronalaženje ruke
    slika = detektor.pronadjiRuku(slika)
    lmLista, bbox = detektor.pronadjiPoziciju(
        slika)           # Dohvaćanje položaja ruke

    if len(lmLista) != 0:
        x1, y1 = lmLista[8][1:]
        x2, y2 = lmLista[12][1:]

        prsti = detektor.prstPodignut()      # Provjeravanje jesu li prsti podignuti
        cv2.rectangle(slika, (frekvencijaOkvira, frekvencijaOkvira), (sirina - frekvencijaOkvira,
                      visina - frekvencijaOkvira), (255, 0, 255), 2)   # Stvaranje rubnog okvira
        if prsti[1] == 1 and prsti[2] == 0:     # Ako je kažiprst gore, a srednji prst dolje
            x3 = np.interp(x1, (frekvencijaOkvira, sirina -
                           frekvencijaOkvira), (0, sirinaZaslona))
            y3 = np.interp(y1, (frekvencijaOkvira, visina -
                           frekvencijaOkvira), (0, visinaZaslona))

            trenutni_x = prethodni_x + (x3 - prethodni_x) / izgladjivanje
            trenutni_y = prethodni_y + (y3 - prethodni_y) / izgladjivanje

            autopy.mouse.move(sirinaZaslona - trenutni_x,
                              trenutni_y)    # Pomicanje pokazivača
            cv2.circle(slika, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
            prethodni_x, prethodni_y = trenutni_x, trenutni_y

        if prsti[1] == 1 and prsti[2] == 1:     # Ako su kažiprst i srednji prst gore
            duljina, slika, linijaInfo = detektor.pronadjiUdaljenost(
                8, 12, slika)

            if duljina < 40:     # Ako su oba prsta jako blizu jedan drugome
                cv2.circle(
                    slika, (linijaInfo[4], linijaInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()    # "klik"

    cVrijeme = time.time()
    fps = 1/(cVrijeme-pVrijeme)
    pVrijeme = cVrijeme
    cv2.putText(slika, str(int(fps)), (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Slika", slika)
    cv2.waitKey(1)
