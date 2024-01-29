import cv2 as cv
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

# mediapipe initialization
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
)

# create opencv initialization
cam = cv.VideoCapture(0)

# keep camera open
while cam.isOpened():

    # return frame or fail
    success, frame = cam.read()
    if not success:
        print("Camera Not Available")
        continue

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    hands_detected = hands.process(frame)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )

    cv.imshow("HaFlex", frame)


    if cv.waitKey(20) & 0xff == ord("q"):
        break

# destroy opencv instance 
cam.release()

