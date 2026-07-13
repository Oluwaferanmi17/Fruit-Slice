import math
import cv2
import mediapipe as mp
import random

# print(mp)
# print(mp.__file__)
score = 0
fruit_x = 300
fruit_y = 300
radius = 30
fruit_vx = 5
fruit_vy = -18

gravity = 0.7

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, c = img.shape
            index_tip = hand_landmarks.landmark[8]
            cx = int(index_tip.x * w)
            cy = int(index_tip.y * h)
            cv2.circle(img, (cx, cy), 12, (0, 255, 255), -1)

            cv2.circle(img, (int(fruit_x), int(fruit_y)), radius, (0, 255, 0), -1)
            fruit_x += fruit_vx
            fruit_y += fruit_vy
            fruit_vy += gravity
            if fruit_y > img.shape[0] + 50:
                fruit_x = random.randint(100, img.shape[1] - 100)
                fruit_y = img.shape[0] + 50
                fruit_vx = random.randint(-5, 5)
                fruit_vy = random.randint(-22, -18)

            distance = math.sqrt((cx - fruit_x) ** 2 + (cy - fruit_y) ** 2)
            if distance < radius:
                score += 1
                fruit_x = random.randint(100, img.shape[1] - 100)
                fruit_y = img.shape[0] + 50
                fruit_vx = random.randint(-5, 5)
                fruit_vy = random.randint(-22, -18)

    cv2.putText(
        img,
        f"Score: {score}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(score)
