import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def fingers_up(hand_landmarks):
    # Se basa en la posición relativa de los dedos
    finger_tips = [8, 12, 16, 20]  # Índice, medio, anular, meñique
    fingers = []
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

cap = cv2.VideoCapture(0)
prev_state = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            fingers = fingers_up(hand_landmarks)
            total_fingers = sum(fingers)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if total_fingers == 4 and not prev_state:
                print("✋ Todos los dedos arriba: minimizando ventana...")
                pyautogui.hotkey('command', 'm')  # o 'command' + 'm' en macOS
                prev_state = True
            elif total_fingers < 4:
                prev_state = False

    cv2.imshow("Detección de Mano", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
