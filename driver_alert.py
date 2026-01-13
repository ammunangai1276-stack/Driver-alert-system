import cv2
import dlib
import time
import json
import threading
import requests
from scipy.spatial import distance
from playsound import playsound
from imutils import face_utils

# ===================== CONFIG =====================
EAR_THRESHOLD = 0.25
DROWSY_TIME = 1.5   # seconds
ALERT_SOUND = "alert.wav"

VEHICLE_ID = "VH_102"
DRIVER_ID = "DR_45"

CLOUD_API = "https://example.com/api/driver-alert"  # mock
# =================================================


# ---------- EAR calculation ----------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# ---------- Alert sound (continuous) ----------
def play_alert():
    while alert_active:
        playsound(ALERT_SOUND)


# ---------- Send event to cloud ----------
def send_to_cloud(event):
    try:
        headers = {"Content-Type": "application/json"}
        requests.post(CLOUD_API, data=json.dumps(event), headers=headers, timeout=2)
    except:
        print("Cloud not reachable. Event stored locally.")


# ---------- Initialize ----------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

start_time = None
alert_active = False

print("Driver Alert System Started")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eyes
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,255,0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,0), 1)

        # ----- Drowsiness logic -----
        if ear < EAR_THRESHOLD:
            if start_time is None:
                start_time = time.time()
            elapsed = time.time() - start_time

            if elapsed >= DROWSY_TIME:
                if not alert_active:
                    alert_active = True
                    threading.Thread(target=play_alert, daemon=True).start()

                    event = {
                        "vehicle_id": VEHICLE_ID,
                        "driver_id": DRIVER_ID,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event": "DROWSINESS_DETECTED",
                        "duration": round(elapsed, 2)
                    }

                    print("DROWSINESS ALERT:", event)
                    threading.Thread(
                        target=send_to_cloud, args=(event,), daemon=True
                    ).start()

                cv2.putText(frame, "DROWSY!", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        else:
            start_time = None
            alert_active = False

    cv2.imshow("Driver Alert", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()
print("System stopped")
