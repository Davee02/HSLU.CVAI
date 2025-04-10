import cv2
import mediapipe as mp
import numpy as np

# MediaPipe-Hands initialisieren
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Für Videoverarbeitung
    max_num_hands=2,          # Maximal 2 Hände erkennen
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Webcam öffnen
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Webcam konnte nicht gelesen werden.")
        continue
    
    # Bild für MediaPipe vorbereiten (in RGB konvertieren)
    image = cv2.flip(image, 1)  # Horizontal spiegeln für intuitivere Steuerung
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Verarbeitung mit MediaPipe
    results = hands.process(image_rgb)
    
    # Ergebnisse anzeigen
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Hand-Landmarks zeichnen
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
    
    # Bild anzeigen
    cv2.imshow('Handgestenerkennung', image)
    
    # Beenden mit ESC-Taste
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()
hands.close()