# Face Recognition Attendance System

This project is a real-time face recognition based attendance system built using Flask and OpenCV.  
It allows employees to register their face, mark check-in and check-out, and automatically saves attendance records.

The main goal of this project is to demonstrate how face recognition can be integrated with a web application for practical use.

## Features

- Register new employees using a live camera
- Automatically capture face images and train the model
- Face-based check-in and check-out
- Detect and block unknown faces
- Show live camera feed in the browser
- Save attendance data in CSV format

## Technologies Used

- Python
- Flask
- OpenCV (LBPH Face Recognizer)
- HTML & CSS

## How the System Works

- Face detection is done using OpenCV’s Haar Cascade classifier.
- Face recognition is implemented using the LBPH algorithm, which works well for small datasets.
- During registration, multiple face images are captured automatically from the camera.
- The model is retrained whenever a new person is registered.
- While checking in or out, the system recognizes the face and validates it using a confidence threshold.
- If the face is not recognized, attendance is not marked.

## Attendance Logic

- The system provides separate options for Check-In and Check-Out.
- Check-In records the entry time for the day.
- Check-Out updates the exit time for the same user.
- Each day’s attendance is stored in a separate CSV file.

Sample CSV format:
Name,Date,PunchIn,PunchOut
kulbhushan,2026-01-28,09:15:20,18:05:10

## Real-Time Camera

- The webcam feed is displayed directly in the browser.
- The camera is visible during employee registration, check-in, and check-out.
- Recognition status (recognized or unknown) is shown in real time.

## Unknown Face Handling

- A confidence threshold is applied during face recognition.
- Faces that do not meet the confidence requirement are treated as unknown.
- Unknown faces are not allowed to mark attendance.


## How to Run the Project

1. Install required packages:
pip install -r requirements.txt

2. Run the application:
python app.py

3. Open the browser and go to:
http://127.0.0.1:5000

## Demo Video

A short demo showing all features of the system:

[Watch Demo on Google Drive](https://drive.google.com/file/d/1SOH97ZfyUPz_PFIxFjanMdVM52yvcn-T/view)

The demo includes:
- Registering a new user
- Face-based Check-In
- Face-based Check-Out
- Handling unknown faces
- Viewing attendance CSV


## Limitations

- The system performs best in good lighting conditions.
- Accuracy depends on the quality of captured face images.
- This project is a prototype and not intended for production use.
- Advanced spoof detection is not implemented.

## Summary

This project demonstrates practical face recognition, backend and frontend integration, real-time camera handling, and clean attendance logic.  
It is designed as a functional prototype suitable for an AI/ML intern assignment.
 HTML & CSS

