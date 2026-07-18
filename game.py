import math
import cv2
import mediapipe as mp
import random

# print(mp)
# print(mp.__file__)
score = 0
radius = 30
prev_x = None
prev_y = None
gravity = 0.7

apple = cv2.imread("pic/apple.png", cv2.IMREAD_UNCHANGED)
mango = cv2.imread("pic/mango.png", cv2.IMREAD_UNCHANGED)
pineapple = cv2.imread("pic/pineaple.png", cv2.IMREAD_UNCHANGED)
banana = cv2.imread("pic/banana.png", cv2.IMREAD_UNCHANGED)
bomb = cv2.imread("pic/bomb.png", cv2.IMREAD_UNCHANGED)

apple = cv2.resize(apple, (60, 60))
mango = cv2.resize(mango, (60, 60))
pineapple = cv2.resize(pineapple, (60, 60))
banana = cv2.resize(banana, (60, 60))
bomb = cv2.resize(bomb, (60, 60))

fruits = [apple, mango, pineapple, banana]
fruit_list = []

for i in range(3):
    is_bomb = random.random() < 0.2
    fruit = {
        "x": random.randint(100, 500),
        "y": random.randint(300, 500),
        "vx": random.choice([-5, -4, -3, 3, 4, 5]),
        "vy": random.randint(-22, -18),
        "img": bomb if is_bomb else random.choice(fruits),
        "bomb": is_bomb,
    }

    fruit_list.append(fruit)


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

            for fruit in fruit_list:

                # Move
                fruit["x"] += fruit["vx"]
                fruit["y"] += fruit["vy"]
                fruit["vy"] += gravity

                # Draw
                overlay_png(
                    img,
                    fruit["img"],
                    int(fruit["x"] - 30),
                    int(fruit["y"] - 30),
                )

                # Respawn
                if fruit["y"] > img.shape[0] + 50:
                    fruit["x"] = random.randint(100, img.shape[1] - 100)
                    fruit["y"] = img.shape[0] + 50
                    fruit["vx"] = random.choice([-5, -4, -3, 3, 4, 5])
                    fruit["vy"] = random.randint(-22, -18)

                    is_bomb = random.random() < 0.2
                    fruit["bomb"] = is_bomb
                    fruit["img"] = bomb if is_bomb else random.choice(fruits)

                # Collision
                distance = math.sqrt((cx - fruit["x"]) ** 2 + (cy - fruit["y"]) ** 2)

                if distance < radius:
                    if fruit["bomb"]:

                        # Dark overlay
                        overlay = img.copy()
                        cv2.rectangle(
                            overlay,
                            (0, 0),
                            (img.shape[1], img.shape[0]),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

                        # Game Over text
                        text = "GAME OVER"

                        (text_width, text_height), baseline = cv2.getTextSize(
                            text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            5,
                        )
                        text_x = (img.shape[1] - text_width) // 2
                        text_y = (img.shape[0] + text_height) // 2
                        cv2.putText(
                            img,
                            text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            5,
                        )

                        # Final score
                        score_text = f"Final Score: {score}"

                        (score_width, score_height), _ = cv2.getTextSize(
                            score_text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            2,
                        )

                        score_x = (img.shape[1] - score_width) // 2
                        score_y = text_y + 60

                        cv2.putText(
                            img,
                            score_text,
                            (score_x, score_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )

                        cv2.imshow("Image", img)
                        cv2.waitKey(3000)  # Wait 3 seconds

                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                    score += 1

                    fruit["x"] = random.randint(100, img.shape[1] - 100)
                    fruit["y"] = img.shape[0] + 50
                    fruit["vx"] = random.choice([-5, -4, -3, 3, 4, 5])
                    fruit["vy"] = random.randint(-22, -18)

                    is_bomb = random.random() < 0.2
                    fruit["bomb"] = is_bomb
                    fruit["img"] = bomb if is_bomb else random.choice(fruits)

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
