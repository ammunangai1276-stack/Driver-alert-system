"""
Reliable Eye Drowsiness Detection – OpenCV Only (macOS)
- Rectangles around face and eyes
- Alarm continues until eyes open
- Plays alarm continuously with no noticeable gap
"""

import cv2
import time
import subprocess
from threading import Thread

# ==========================
# CONFIG
# ==========================
EYES_CLOSED_SECONDS = 1.0  # seconds eyes must be closed
MIN_EYES = 1               # minimum eyes to consider "open"
ALARM_SOUND = "/System/Library/Sounds/Sosumi.aiff"

# ==========================
# GLOBAL STATE
# ==========================
alarm_active = False
alarm_thread = None

# ==========================
# ALARM FUNCTIONS
# ==========================
def alarm_loop():
    """Continuously play alarm sound with no delay"""
    while alarm_active:
        # Run afplay and block until it finishes, then immediately restart
        subprocess.run(["afplay", ALARM_SOUND], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def start_alarm():
    global alarm_active, alarm_thread
    if alarm_active:
        return
    alarm_active = True
    alarm_thread = Thread(target=alarm_loop, daemon=True)
    alarm_thread.start()

def stop_alarm():
    global alarm_active
    alarm_active = False

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    cap = cv2.VideoCapture(0)
    eyes_closed_start = None

    print("Drowsiness Detection Started")
    print("Press Q to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        h, w = frame.shape[:2]

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        eyes_detected = 0
        face_present = len(faces) > 0

        for (x, y, fw, fh) in faces:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 0, 0), 2)
            roi_gray = gray[y:y + fh // 2, x:x + fw]
            roi_color = frame[y:y + fh // 2, x:x + fw]

            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            eyes_detected += len(eyes)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        now = time.time()

        # DROWSINESS LOGIC
        if face_present and eyes_detected < MIN_EYES:
            if eyes_closed_start is None:
                eyes_closed_start = now
            elif (now - eyes_closed_start) >= EYES_CLOSED_SECONDS:
                start_alarm()
        else:
            eyes_closed_start = None
            stop_alarm()

        # UI
        closed_time = (now - eyes_closed_start) if eyes_closed_start else 0
        status = "ALERT!" if alarm_active else "Eyes Open"
        color = (0, 0, 255) if alarm_active else (0, 255, 0)

        cv2.putText(frame, f"Status: {status}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Eyes Closed Time: {closed_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if alarm_active:
            cv2.putText(frame, "DROWSINESS ALERT!", (w // 2 - 220, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Program exited cleanly")

if __name__ == "__main__":
    main()

