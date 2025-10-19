from flask import Flask, jsonify, request
from threading import Thread
import time
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance
from keras.models import load_model
from twilio.rest import Client
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Twilio API Credentials
TWILIO_SID = '         '  # Replace with your actual Twilio SID
TWILIO_AUTH_TOKEN = 'your_auth_token_here'  # Replace with your actual Twilio Auth Token
FROM_PHONE = '+13194627060'  # Replace with your Twilio phone number
TO_PHONE = '+1234567890'  # Replace with the recipient phone number

# Twilio Client setup
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Shared detection status
status = {
    "ear": 0.0,
    "mar": 0.0,
    "drowsy": False,
    "timestamp": time.time()
}

# Detection control flag
is_running = False

# Drowsiness thresholds
EAR_THRESH = 0.25
MAR_THRESH = 0.6
FRAME_CHECK = 20
frame_counter = 0

# Load models
predictor_path = r"C:\DriveSafeWebsite\login-signup\backend\shape_predictor_68_face_landmarks.dat"
model_path = r"C:\DriveSafeWebsite\login-signup\backend\eye_yawn_model.h5"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
yawn_model = load_model(model_path)

# Landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    B = distance.euclidean(mouth[12], mouth[16])
    return A / B


def send_alert():
    """ Send an alert via Twilio when drowsiness is detected. """
    try:
        message = client.messages.create(
            body="Alert: Driver is Drowsy! Please Wake Up!",
            from_=FROM_PHONE,
            to=TO_PHONE
        )
        print(f"Alert sent! Message SID: {message.sid}")
    except Exception as e:
        print(f"Error sending alert: {e}")


def run_detection():
    global is_running, status, frame_counter
    cap = cv2.VideoCapture(0)

    while is_running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            shape = shape_predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            drowsy = False
            if ear < EAR_THRESH or mar > MAR_THRESH:
                frame_counter += 1
                if frame_counter >= FRAME_CHECK:
                    drowsy = True
                    send_alert()  # Send alert when drowsy is detected
            else:
                frame_counter = 0

            status.update({
                "ear": round(ear, 2),
                "mar": round(mar, 2),
                "drowsy": drowsy,
                "timestamp": time.time()
            })
            break  # only process first face

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()


@app.route("/start", methods=["POST"])
def start_detection():
    global is_running
    if not is_running:
        is_running = True
        Thread(target=run_detection, daemon=True).start()
        return jsonify({"status": "Detection started"})
    return jsonify({"status": "Already running"})


@app.route("/stop", methods=["POST"])
def stop_detection():
    global is_running
    is_running = False
    return jsonify({"status": "Detection stopped"})


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(status)


if __name__ == "__main__":
    app.run(port=5000, debug=True)

