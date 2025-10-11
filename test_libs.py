import mediapipe as mp
import cv2
import numpy as np
import pickle

from data.utils import CLASSES

def bounding_box_2(width_frame, height_frame, hand_landmarks):
    hand_landmarks = np.array(hand_landmarks)
    xmin, xmax = hand_landmarks[:, 0].min(), hand_landmarks[:, 0].max()
    ymin, ymax = hand_landmarks[:, 1].min(), hand_landmarks[:, 1].max()
    width, height = xmax - xmin, ymax - ymin
    if width > height:
        delta_x = 0.1 * width
        delta_y = delta_x + (width - height) / 2
    else:
        delta_y = 0.1 * height
        delta_x = delta_y + (height - width) / 2
    start_x = max(int((xmin - delta_x) * width_frame), 0)
    start_y = max(int((ymin - delta_y) * height_frame), 0)
    end_x   = min(int((xmax + delta_x) * width_frame), width_frame)
    end_y   = min(int((ymax + delta_y) * height_frame), height_frame)
    landmarks_norm = []
    for i in range(len(hand_landmarks)):
        x = hand_landmarks[i, 0]
        y = hand_landmarks[i, 1]
        cx, cy = int(x * width_frame), int(y * height_frame)
        x_norm = (cx - start_x) / (end_x - start_x)
        y_norm = (cy - start_y) / (end_y - start_y)
        landmarks_norm.append([x_norm, y_norm])

    result = {'landmarks': landmarks_norm,
              'bounding_box_start': (start_x, start_y),
              'bounding_box_end': (end_x, end_y)}
    return result


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

DATA = []
LABELS = []

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    file_name = 'out-checkpoints/17-18/MLPClassifier.p'
    model = pickle.load(open(file_name, 'rb'))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 255, 0)

    while True:
        x_ = []
        y_ = []
        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" hoặc "Right"

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,  # nối các điểm
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                x_landmarks = []
                y_landmarks = []
                hand_data = []
                if label == "Left":
                    for lm in hand_landmarks.landmark:
                        hand_data.append([1 - lm.x, lm.y])
                else:
                    for lm in hand_landmarks.landmark:
                        hand_data.append([lm.x, lm.y])

                if len(hand_data) == 21:
                    output_bounding_box = bounding_box_2(W, H, hand_data)
                    landmarks_norm = output_bounding_box['landmarks']
                    DATA.append(landmarks_norm)
                    (start_x, start_y) = output_bounding_box['bounding_box_start']
                    (end_x, end_y) = output_bounding_box['bounding_box_end']
                    landmarks_norm = np.array(landmarks_norm, dtype=np.float32).flatten()

                    prediction = model.predict([np.array(landmarks_norm)])

                    text=f'{label}: {CLASSES[int(prediction[0])]}'

                    (text_w1, text_h1), _ = cv2.getTextSize(text, font, font_scale, thickness)

                    if label == "Left":
                        x = (W - text_w1) // 4
                        y = int(H * 0.4)
                    else:
                        x = (W - text_w1) * 3 // 4
                        y = int(H * 0.4)

                    cv2.putText(frame, text, (x, y),font, font_scale, color, thickness)

        cv2.imshow("Hands with Bounding Box", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()