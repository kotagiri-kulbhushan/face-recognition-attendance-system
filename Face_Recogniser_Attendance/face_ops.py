import cv2
import os
import csv
import time
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_IMG = os.path.join(BASE_DIR, "TrainingImage")
TRAIN_LABEL = os.path.join(BASE_DIR, "TrainingImageLabel")
ATTENDANCE = os.path.join(BASE_DIR, "Attendance")

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_LABEL, exist_ok=True)
os.makedirs(ATTENDANCE, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
camera = cv2.VideoCapture(0)

# shared status (shown in UI)
current_status = "Idle"


# ---------------- VIDEO STREAM ----------------
def gen_frames():
    global current_status

    while True:
        success, frame = camera.read()
        if not success:
            break

        # draw status text on frame
        cv2.putText(
            frame,
            current_status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# ---------------- REGISTER PERSON ----------------
def register_person(name):
    global current_status
    current_status = f"Registering: {name}"

    count = 0
    last = time.time()

    while count < 7:
        success, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if time.time() - last > 0.6:
                face = gray[y:y+h, x:x+w]
                cv2.imwrite(f"{TRAIN_IMG}/{name}.{count}.jpg", face)
                count += 1
                last = time.time()

                current_status = f"Captured image {count}/7"

    train_model()
    current_status = f"Registration completed for {name}"


# ---------------- TRAIN MODEL ----------------
def train_model():
    faces, labels = [], []
    label_map = {}
    label_id = 0

    for img in os.listdir(TRAIN_IMG):
        name = img.split(".")[0]

        if name not in label_map.values():
            label_map[label_id] = name
            current = label_id
            label_id += 1
        else:
            current = list(label_map.keys())[list(label_map.values()).index(name)]

        image = cv2.imread(f"{TRAIN_IMG}/{img}", cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        labels.append(current)

    recognizer.train(faces, np.array(labels))
    recognizer.save(f"{TRAIN_LABEL}/Trainer.yml")

    with open(f"{TRAIN_LABEL}/labels.txt", "w") as f:
        for k, v in label_map.items():
            f.write(f"{k},{v}\n")


# ---------------- FACE RECOGNITION ----------------
def recognize(action):
    global current_status

    recognizer.read(f"{TRAIN_LABEL}/Trainer.yml")

    labels = {}
    with open(f"{TRAIN_LABEL}/labels.txt") as f:
        for line in f:
            k, v = line.strip().split(",")
            labels[int(k)] = v

    current_status = "Looking for face..."
    start_time = time.time()

    while True:
        success, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, conf = recognizer.predict(face)

            if conf < 60:
                name = labels.get(label, "Unknown")
                current_status = f"Recognized: {name}"
                mark_attendance(name, action)
                return
            else:
                current_status = "Unknown face detected"

        if time.time() - start_time > 5:
            current_status = "No valid face recognized"
            return


# ---------------- ATTENDANCE ----------------
def mark_attendance(name, action):
    global current_status

    if name == "Unknown":
        current_status = "Access denied: Unknown face"
        return

    date = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")
    file = f"{ATTENDANCE}/Attendance_{date}.csv"

    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "PunchIn", "PunchOut"])

    rows = []
    with open(file, "r") as f:
        rows = list(csv.reader(f))

    for row in rows[1:]:
        if row[0] == name:
            if action == "out" and row[3] == "":
                row[3] = time_now
                current_status = f"{name} checked out at {time_now}"
            with open(file, "w", newline="") as fw:
                csv.writer(fw).writerows(rows)
            return

    if action == "in":
        with open(file, "a", newline="") as f:
            csv.writer(f).writerow([name, date, time_now, ""])
        current_status = f"{name} checked in at {time_now}"
