import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance
from keras.models import load_model
import pygame
import time

# Initialize Pygame for sound alerts
pygame.mixer.init()

# Constants
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
FRAME_CHECK = 20

# Load models and predictor
predictor_path = "C:\DriveSafeWebsite\login-signup\backend\shape_predictor_68_face_landmarks.dat"
model_path = "C:\DriveSafeWebsite\login-signup\backend\eye_yawn_model.h5"
alarm_sound_path = "C:\DriveSafeWebsite\login-signup\backend\music.wav"

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
yawn_model = load_model(model_path)

# Indices for facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']


# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# MAR calculation
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    B = distance.euclidean(mouth[12], mouth[16])
    mar = A / B
    return mar


# Detection loop (can run in a thread)
def start_detection(status_callback=None):
    cap = cv2.VideoCapture(0)
    flag = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = face_detector(gray, 0)

        for subject in subjects:
            shape = shape_predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            is_drowsy = False
            if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
                flag += 1
                if flag >= FRAME_CHECK:
                    is_drowsy = True
                    pygame.mixer.music.load(alarm_sound_path)
                    pygame.mixer.music.play()
            else:
                flag = 0

            if status_callback:
                status_callback({
                    "ear": ear,
                    "mar": mar,
                    "drowsy": is_drowsy,
                    "timestamp": time.time()
                })

            break  # process only first detected face

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
