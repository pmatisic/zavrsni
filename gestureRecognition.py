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
model = load_model('mp_hand_gesture')

# Učitavanje biblioteke naziva gestikulacija.
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Inicijalizacija web-kamere.
cap = cv2.VideoCapture(0)

while True:

    # Čitanje 'frame-ova' s kamere.
    _, frame = cap.read()

    x, y, c = frame.shape

    # Okretanje 'frame-a' okomito.
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dohvaćanje predikcije prikaza ruke.
    result = hands.process(framergb)

    #print(result)
    
    className = ''

    # Naknadna (dodatna) obrada rezultata.
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                #print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Crtanje orijentira na 'frame-u'.
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predikcija gestikulacije.
            prediction = model.predict([landmarks])
            #print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # Prikazivanje predviđanja na 'frame-u'.
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# Otpuštanje web-kamere i gašenje svih aktivnih prozora.
cap.release()

cv2.destroyAllWindows()