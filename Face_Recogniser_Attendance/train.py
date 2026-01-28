import cv2
import os
import numpy as np

# ensure folders exist
os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)

faces = []
labels = []
label_map = {}
label_id = 0

images = os.listdir("TrainingImage")

if len(images) == 0:
    print("No training images found. Please add images to TrainingImage folder.")
    exit()

for img in images:
    name = img.split(".")[0]

    if name not in label_map.values():
        label_map[label_id] = name
        current_label = label_id
        label_id += 1
    else:
        current_label = list(label_map.keys())[
            list(label_map.values()).index(name)
        ]

    img_path = os.path.join("TrainingImage", img)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        faces.append(image)
        labels.append(current_label)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

recognizer.save("TrainingImageLabel/Trainer.yml")

with open("TrainingImageLabel/labels.txt", "w") as f:
    for k, v in label_map.items():
        f.write(f"{k},{v}\n")

print("Training completed successfully")
