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
prev_x = None
prev_y = None
gravity = 0.7
split = False

left_x = 0
left_y = 0
right_x = 0
right_y = 0

left_vx = 0
left_vy = 0

right_vx = 0
right_vy = 0


apple = cv2.imread("pic/apple.png", cv2.IMREAD_UNCHANGED)
mango = cv2.imread("pic/mango.png", cv2.IMREAD_UNCHANGED)
pineapple = cv2.imread("pic/pineaple.png", cv2.IMREAD_UNCHANGED)
banana = cv2.imread("pic/banana.png", cv2.IMREAD_UNCHANGED)
print("Apple:", apple is None)
print("Mango:", mango is None)
print("Pineapple:", pineapple is None)
print("Banana:", banana is None)
apple = cv2.resize(apple, (60, 60))
mango = cv2.resize(mango, (60, 60))
pineapple = cv2.resize(pineapple, (60, 60))
banana = cv2.resize(banana, (60, 60))

fruits = [apple, mango, pineapple, banana]
fruit_img = random.choice(fruits)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


def overlay_png(background, overlay, x, y):
    h, w = overlay.shape[:2]

    # Make sure the image stays inside the frame
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return

    # Split color and alpha channels
    overlay_rgb = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0

    # Region of interest
    roi = background[y : y + h, x : x + w]

    # Blend the images
    for c in range(3):
        roi[:, :, c] = alpha * overlay_rgb[:, :, c] + (1 - alpha) * roi[:, :, c]

    background[y : y + h, x : x + w] = roi


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
            if prev_x is not None and prev_y is not None:
                cv2.line(img, (prev_x, prev_y), (cx, cy), (0, 255, 255), 6)
            prev_x = cx
            prev_y = cy
            cv2.circle(img, (cx, cy), 12, (0, 255, 255), -1)

            fruit_x += fruit_vx
            fruit_y += fruit_vy
            fruit_vy += gravity

            if not split:
                overlay_png(img, fruit_img, int(fruit_x - 30), int(fruit_y - 30))

            if split:

                left_x += left_vx
                left_y += left_vy
                left_vy += gravity

                right_x += right_vx
                right_y += right_vy
                right_vy += gravity

                overlay_png(img, fruit_img, int(left_x - 30), int(left_y - 30))
                overlay_png(img, fruit_img, int(right_x - 30), int(right_y - 30))
                if left_y > img.shape[0] + 50:
                    split = False
                    fruit_x = random.randint(100, img.shape[1] - 100)
                    fruit_y = img.shape[0] + 50
                    fruit_vx = random.choice([-5, -4, -3, -2, 2, 3, 4, 5])
                    fruit_vy = random.randint(-22, -18)
                    fruit_img = random.choice(fruits)

            overlay_png(img, fruit_img, int(fruit_x - 30), int(fruit_y - 30))
            if fruit_y > img.shape[0] + 50:
                fruit_x = random.randint(100, img.shape[1] - 100)
                fruit_y = img.shape[0] + 50
                fruit_vx = random.randint(-5, 5)
                fruit_vy = random.randint(-22, -18)
                fruit_img = random.choice(fruits)
                h, w = fruit_img.shape[:2]

                left_half = fruit_img[:, : w // 2]
                right_half = fruit_img[:, w // 2 :]

            distance = math.sqrt((cx - fruit_x) ** 2 + (cy - fruit_y) ** 2)
            if distance < radius:
                score += 1

                split = True

                left_x = fruit_x
                left_y = fruit_y

                right_x = fruit_x
                right_y = fruit_y

                left_vx = -6
                right_vx = 6

                left_vy = fruit_vy
                right_vy = fruit_vy

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
