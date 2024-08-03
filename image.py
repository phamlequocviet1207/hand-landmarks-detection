import cv2
import mediapipe as mp

drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

img_path = './data/1.png'

with handsModule.Hands(static_image_mode=True) as hands:
    img = cv2.imread(img_path)

    width, height, _ = img.shape
    width = width // 100 * 50
    height = height // 100 * 50

    img = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)

    for handLandmarks in result.multi_hand_landmarks:
        drawingModule.draw_landmarks(img, handLandmarks, handsModule.HAND_CONNECTIONS)

    cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
