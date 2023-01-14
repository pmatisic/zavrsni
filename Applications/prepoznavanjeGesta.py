import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model

# Inicijalizacija MediaPipe-a.
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Učitavanje modela koji prepoznaje gestikulacije.
model = load_model('model')

# Učitavanje biblioteke naziva gestikulacija.
f = open('geste.imena', 'r')
imenaKlasa = f.read().split('\n')
f.close()
print(imenaKlasa)

# Inicijalizacija web-kamere.
sadrzaj = cv2.VideoCapture(0)

while True:

    # Čitanje s kamere.
    _, slika = sadrzaj.read()

    x, y, c = slika.shape

    # Okretanje slike.
    slika = cv2.flip(slika, 1)
    slikaRGB = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)

    # Dohvaćanje predikcije prikaza ruke.
    rezultat = hands.process(slikaRGB)

    # print(rezultat)

    imeKlase = ''

    # Naknadna (dodatna) obrada rezultata.
    if rezultat.multi_hand_landmarks:
        landmarks = []
        for handslms in rezultat.multi_hand_landmarks:
            for lm in handslms.landmark:
                #print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Crtanje orijentira na prikazu.
            mpDraw.draw_landmarks(slika, handslms, mpHands.HAND_CONNECTIONS)

            # Predikcija gestikulacije.
            predikcija = model.predict([landmarks])
            # print(predikcija)
            klasaID = np.argmax(predikcija)
            imeKlase = imenaKlasa[klasaID]

    # Prikazivanje predikcije.
    cv2.putText(slika, imeKlase, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Slika", slika)

    if cv2.waitKey(1) == ord('q'):
        break

# Gašenje svih aktivnih prozora.
sadrzaj.release()

cv2.destroyAllWindows()
