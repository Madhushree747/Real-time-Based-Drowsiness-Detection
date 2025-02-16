from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from pygame import mixer
import numpy as np
import time

mixer.init()
mixer.music.load(r"C:\Users\madhu\OneDrive\Desktop\drowsii (2)\drowsii\bleep-41488.mp3")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 150  # Approximately 5 seconds if ~30 FPS
head_movement_threshold = 20  # Threshold for head movement detection
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r'C:\Users\madhu\OneDrive\Desktop\drowsii (2)\drowsii\shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
nose_idx = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]

cap = cv2.VideoCapture(0)
flag = 0
alarm_status = False
prev_nose_position = None

def start_alarm():
    if not mixer.music.get_busy():
        mixer.music.play()

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        nose = shape[nose_idx[0]:nose_idx[1]].mean(axis=0)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                if not alarm_status:
                    print("Drowsy - Eyes closed for 5 seconds")
                    start_alarm()
                    alarm_status = True
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0
            alarm_status = False

        # Check head movement
        if prev_nose_position is not None:
            movement = np.linalg.norm(nose - prev_nose_position)
            if movement > head_movement_threshold:
                print("Head movement detected - Possible unconsciousness")
                start_alarm()
                cv2.putText(frame, "****HEAD MOVEMENT ALERT!****", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        prev_nose_position = nose

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()