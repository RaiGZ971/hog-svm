import cv2
from inference.pipeline import Pipeline

pipeline = Pipeline()

video_path = "/home/code871/Git/fsl-svm/server3/clips/0/17.MOV"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = pipeline.predict(frame)

    if result:
        print("Prediction:", result)
