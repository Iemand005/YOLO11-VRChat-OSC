from ultralytics import YOLO

import cv2

import numpy as np

cap = cv2.VideoCapture(0)

model = YOLO("yolo11m")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (640, 480))

    results = model(img)

    img = results[0].plot()

    cv2.imshow("frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


