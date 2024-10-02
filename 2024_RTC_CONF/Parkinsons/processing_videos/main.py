# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import sys
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


min_num_of_frames = 0


def knn_impute_csv(input_file_path, output_file_path, n_neighbors=5):
    """
    Reads a CSV file, sorts it by frame and landmark, checks for missing values,
    and applies K-Nearest Neighbors (KNN) imputation for missing data only.
    Outputs a warning if any modifications are made per frame per landmark and
    saves the imputed data to a new CSV file.

    Parameters:
    file_path (str): The path to the input CSV file.
    output_file_path (str): The path to save the imputed CSV file.
    n_neighbors (int): The number of neighbors to use for KNN imputation.

    Returns:
    pd.DataFrame: DataFrame with missing values imputed (if any).
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)

    # Ensure the CSV is sorted by 'frame' and 'landmark'
    if "frame" in df.columns and "landmark" in df.columns:
        df.sort_values(by=["frame", "landmark"], inplace=True)
    else:
        raise ValueError("The input CSV must contain 'frame' and 'landmark' columns.")

    # Check for missing values
    if df.isnull().values.any():
        print("WARNING: Missing values detected. Applying KNN imputation.")

        # Apply KNN imputation for missing values
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Output a warning per frame (row) and landmark (column) if any modifications were made
        num_rows, num_cols = df.shape
        for i in range(num_rows):
            for j in range(num_cols):
                if pd.isnull(df.iloc[i, j]):  # Check if the original value was missing
                    imputed_val = df_imputed.iloc[i, j]
                    log_message = (
                        f"\n\n\n"
                        f"-------------------------------------------------------------"
                        f"WARNING: Imputed missing value in frame {df.iloc[i]['frame']}, "
                        f"landmark '{df.iloc[i]['landmark']}' with value {imputed_val}."
                        f"-------------------------------------------------------------"
                        f"\n\n\n"
                    )
                    print(log_message)

                    # Write the log message to a log file
                    log_file = "log.txt"
                    with open(log_file, "a+") as file:
                        file.write(log_message)

        # Save the imputed DataFrame to the specified output CSV file
        df_imputed.to_csv(output_file_path, index=False, float_format="%.15g")
        print(f"Imputed CSV file saved to: {output_file_path}")

        return df_imputed

    else:
        print("No missing values detected. No imputation needed.")

        # Save the sorted original DataFrame to the specified output CSV file
        df.to_csv(output_file_path, index=False, float_format="%.15g")
        print(f"No changes made. Sorted original CSV file saved to: {output_file_path}")

        return df


def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    # print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        if mp_pose.PoseLandmark(idx).name in [
            "NOSE",
            "LEFT_EYE_INNER",
            "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER",
            "RIGHT_EYE_OUTER",
            "LEFT_EAR",
            "RIGHT_EAR",
            "MOUTH_LEFT",
            "MOUTH_RIGHT",
            "LEFT_EYE",
            "LEFT_FOOT_INDEX",
            "LEFT_INDEX",
            "LEFT_PINKY",
            "LEFT_THUMB",
            "RIGHT_EYE",
            "RIGHT_FOOT_INDEX",
            "RIGHT_INDEX",
            "RIGHT_PINKY",
            "RIGHT_THUMB",
        ]:
            continue

        # print(
        #     f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})"
        # )
        csv_data.append(
            [
                frame_number,
                mp_pose.PoseLandmark(idx).name,
                landmark.x,
                landmark.y,
                landmark.z,
            ]
        )
    # print("\n")


import pandas as pd


def check_duplicates_in_csv(file_path):
    """
    This function checks for duplicate entries in a CSV file based on 'frame' and 'landmark' columns.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - None: Prints whether duplicates were found and the duplicate rows if any.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if the required columns are present
        if "frame" not in df.columns or "landmark" not in df.columns:
            print("Error: The CSV file must contain 'frame' and 'landmark' columns.")
            return

        # Find duplicates based on 'frame' and 'landmark' columns
        duplicates = df[df.duplicated(subset=["frame", "landmark"], keep=False)]

        # Report findings
        if not duplicates.empty:
            log_message = (
                f"\n\n\n"
                f"-------------------------------------------------------------"
                f"Duplicates found in the CSV file:"
                f"\n\n"
                f"{duplicates}"
                f"\n\n"
                f"-------------------------------------------------------------"
                f"\n\n\n"
            )
            print(log_message)

            # Write the log message to a log file
            log_file = "log.txt"
            with open(log_file, "a+") as file:
                file.write(log_message)
        else:
            print("\n\n")
            print("No duplicates found in the CSV file.")
            print("\n\n")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


def process_mp4(mp4file: str):
    print(f"Processing mp4: {mp4file}")

    frame_number = 0
    csv_data = []

    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(mp4file)

    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    outdir, inputflnm = (
        mp4file[: mp4file.rfind("/") + 1],
        mp4file[mp4file.rfind("/") + 1 :],
    )
    videooutdir = outdir.replace("raw", "video")
    dataoutdir = outdir.replace("raw", "tabular")

    # make input dir
    if not os.path.exists(videooutdir):
        os.makedirs(videooutdir)
    if not os.path.exists(dataoutdir):
        os.makedirs(dataoutdir)

    inflnm, inflext = inputflnm.split(".")
    out_stick_mp4 = f"{videooutdir}{inflnm}_stick.{inflext}"
    out_overlay_mp4 = f"{videooutdir}{inflnm}_overlay.{inflext}"
    out_pre_csv = f"{dataoutdir}{inflnm}_original.csv"
    out_imputed_csv = f"{dataoutdir}{inflnm}_imputed.csv"

    out_stick = cv2.VideoWriter(
        out_stick_mp4,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        10,
        (frame_width, frame_height),
    )
    out_overlay = cv2.VideoWriter(
        out_overlay_mp4,
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
            mp4fileonly = mp4file[mp4file.rfind("/") + 1 :]

            write_landmarks_to_csv(
                results.pose_landmarks.landmark, frame_number, csv_data
            )

            # create video with landmarks
            image.flags.writeable = True
            image_overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # original image
            image_stick = np.zeros_like(image)  # Create a black image

            mp_drawing.draw_landmarks(
                image_overlay, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                image_stick, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            out_overlay.write(image_overlay)
            out_stick.write(image_stick)

            # only count if we have landmark data
            frame_number += 1

    # Clean up
    pose.close()
    cap.release()
    out_overlay.release()
    out_stick.release()

    if frame_number < 50:
        log_message = (
            f"\n\n\n"
            f"--------------------------------- Not enough landmarks found in {mp4file}"
            f"\n\n\n"
        )
        print(log_message)

        # Write the log message to a log file
        log_file = "log.txt"
        with open(log_file, "a+") as file:
            file.write(log_message)

        # remove any empty files
        if os.path.exists(out_stick_mp4):
            os.remove(out_stick_mp4)
        if os.path.exists(out_overlay_mp4):
            os.remove(out_overlay_mp4)
        return

    # Save the CSV data to a file
    with open(out_pre_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "landmark", "x", "y", "z"])
        csv_writer.writerows(csv_data)

    # Check for duplicates in the CSV file
    check_duplicates_in_csv(out_pre_csv)

    # Impute missing values in the CSV file
    knn_impute_csv(out_pre_csv, out_imputed_csv)

    print(f"{mp4file} frame count: {frame_number}")
    global min_num_of_frames
    if min_num_of_frames == 0:
        min_num_of_frames = frame_number
        print(f"initial min_num_of_frames: {min_num_of_frames}")
    elif frame_number < min_num_of_frames:
        min_num_of_frames = frame_number
        print(f"new min_num_of_frames: {min_num_of_frames}")
    else:
        print(f"same min_num_of_frames: {min_num_of_frames}")


# iterate over all mp4 files in the raw directory
def process_all_mp4(inputdir: str):
    for file in os.listdir(inputdir):
        if os.path.isdir(f"{inputdir}/{file}"):
            # print(f"Processing directory: {inputdir}/{file}")
            process_all_mp4(f"{inputdir}/{file}")
        elif file.endswith(".mp4") or file.endswith(".avi"):
            # print(f"Processing mp4: {inputdir}/{file}")
            process_mp4(f"{inputdir}/{file}")


if __name__ == "__main__":
    intputdir = "raw"
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        intputdir = sys.argv[1]
    process_all_mp4(intputdir)
    print(f"final min_num_of_frames: {min_num_of_frames}")
