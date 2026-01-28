import cv2
import os
import time

# ensure folder exists
os.makedirs("TrainingImage", exist_ok=True)

name = input("Enter your name: ").strip()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

count = 0
last_capture = time.time()

print("Look at the camera.")
print("Slowly move your face left, right, up, and down.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # capture one image every 0.6 seconds
        if time.time() - last_capture >= 0.6:
            face = gray[y:y+h, x:x+w]
            file_path = f"TrainingImage/{name}.{count}.jpg"
            cv2.imwrite(file_path, face)
            count += 1
            last_capture = time.time()
            print(f"Captured image {count}")

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Face Capture - Press ESC to stop", frame)

    if cv2.waitKey(1) == 27 or count >= 10:
        break

cam.release()
cv2.destroyAllWindows()

print(f"Captured {count} images for {name}")

