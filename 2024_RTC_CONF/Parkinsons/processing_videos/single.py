# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py

import cv2
import mediapipe as mp
import numpy as np
import sys
import csv

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

frame_number = 0
csv_data = []


def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(
            f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})"
        )
        csv_data.append(
            [
                frame_number,
                mp_pose.PoseLandmark(idx).name,
                landmark.x,
                landmark.y,
                landmark.z,
            ]
        )
    print("\n")


pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(sys.argv[1])

if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

outdir, inputflnm = (
    sys.argv[1][: sys.argv[1].rfind("/") + 1],
    sys.argv[1][sys.argv[1].rfind("/") + 1 :],
)
inflnm, inflext = inputflnm.split(".")
out_mp4 = f"{outdir}{inflnm}_annotated.{inflext}"
out_csv = f"{outdir}{inflnm}_annotated.csv"
out = cv2.VideoWriter(
    out_mp4,
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    10,
    (frame_width, frame_height),
)

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    if results.pose_landmarks:
        # Add the landmark coordinates to the list and print them
        write_landmarks_to_csv(results.pose_landmarks.landmark, frame_number, csv_data)

        # create video with landmarks
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # original image
        image = np.zeros_like(image)  # Create a black image

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        out.write(image)

    frame_number += 1
pose.close()
cap.release()
out.release()

# Save the CSV data to a file
with open(out_csv, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["frame_number", "landmark", "x", "y", "z"])
    csv_writer.writerows(csv_data)
