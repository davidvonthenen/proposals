import os
import sys
import csv

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Initialize MediaPipe utilities for drawing and pose estimation.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variable to track the smallest number of valid landmark frames among all processed videos.
min_num_of_frames = 0


def knn_impute_csv(input_file_path, output_file_path, n_neighbors=5):
    """
    Reads a CSV file containing per-frame landmark data, sorts it by 'frame' and 'landmark',
    checks for missing values, and applies K-Nearest Neighbors (KNN) imputation if needed.
    Any imputed values are logged in 'log.txt', and the final CSV is saved.
    """
    df = pd.read_csv(input_file_path)

    # Sort the CSV by 'frame' and 'landmark' to ensure consistent ordering.
    if "frame" in df.columns and "landmark" in df.columns:
        df.sort_values(by=["frame", "landmark"], inplace=True)
    else:
        raise ValueError("The input CSV must contain 'frame' and 'landmark' columns.")

    # Check for missing values and apply KNN imputation if any are found.
    if df.isnull().values.any():
        print("WARNING: Missing values detected. Applying KNN imputation.")

        # Initialize the KNN imputer with the given number of neighbors.
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        # Perform the imputation over the entire dataframe.
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Loop through each element in the original dataframe to log imputation details.
        num_rows, num_cols = df.shape
        for i in range(num_rows):
            for j in range(num_cols):
                if pd.isnull(df.iloc[i, j]):  # Only log values that were missing originally.
                    imputed_val = df_imputed.iloc[i, j]
                    log_message = (
                        "\n\n\n-------------------------------------------------------------\n"
                        f"WARNING: Imputed missing value in frame {df.iloc[i]['frame']}, "
                        f"landmark '{df.iloc[i]['landmark']}' with value {imputed_val}.\n"
                        "-------------------------------------------------------------\n\n\n"
                    )
                    print(log_message)
                    # Append the log message to the log file.
                    with open("log.txt", "a+") as file:
                        file.write(log_message)

        # Save the imputed dataframe as a new CSV file with high precision formatting.
        df_imputed.to_csv(output_file_path, index=False, float_format="%.15g")
        print(f"Imputed CSV file saved to: {output_file_path}")
        return df_imputed
    else:
        print("No missing values detected. No imputation needed.")
        # If no imputation was needed, simply save the sorted dataframe.
        df.to_csv(output_file_path, index=False, float_format="%.15g")
        print(f"No changes made. Sorted CSV file saved to: {output_file_path}")
        return df


def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    """
    Processes the list of landmarks detected in a frame.
    Filters out certain facial landmarks to focus on the relevant body parts, and
    appends the frame number, landmark name, and 3D coordinates (x, y, z) to the csv_data list.
    """
    # Define a set of landmarks that are not needed for this analysis.
    ignored_landmarks = {
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
    }

    # Loop through all detected landmarks and record only those not in the ignored list.
    for idx, landmark in enumerate(landmarks):
        # Retrieve the landmark's name from the MediaPipe Pose enumeration.
        landmark_name = mp_pose.PoseLandmark(idx).name
        if landmark_name not in ignored_landmarks:
            csv_data.append([
                frame_number,
                landmark_name,
                landmark.x,
                landmark.y,
                landmark.z,
            ])


def check_duplicates_in_csv(file_path):
    """
    Reads a CSV file to detect duplicate entries based on 'frame' and 'landmark'.
    If duplicates are found, they are logged in 'log.txt' for further review.
    """
    try:
        df = pd.read_csv(file_path)
        if "frame" not in df.columns or "landmark" not in df.columns:
            print("Error: The CSV file must contain 'frame' and 'landmark' columns.")
            return

        # Identify rows that are duplicated in terms of frame and landmark.
        duplicates = df[df.duplicated(subset=["frame", "landmark"], keep=False)]
        if not duplicates.empty:
            log_message = (
                "\n\n\n-------------------------------------------------------------\n"
                "Duplicates found in the CSV file:\n\n"
                f"{duplicates}\n"
                "-------------------------------------------------------------\n\n\n"
            )
            print(log_message)
            # Log the duplicate entries to the log file.
            with open("log.txt", "a+") as file:
                file.write(log_message)
        else:
            print("\nNo duplicates found in the CSV file.\n")

    except Exception as e:
        # Print any errors encountered during the duplicate check.
        print(f"An error occurred while processing the file: {e}")


def process_mp4(mp4_file: str):
    """
    Processes a single video file (MP4 or AVI):
      - Extracts pose landmarks using MediaPipe's Pose solution.
      - Creates two output videos for visual inspection: one with landmark overlay and one as a stick-figure.
      - Writes the extracted landmarks to a CSV file, checks for duplicates, applies KNN imputation, and updates
        a global frame count variable.
    
    Parameters:
      mp4_file: The path to the video file to process.
    """
    global min_num_of_frames
    print(f"Processing video: {mp4_file}")

    frame_number = 0  # Counter for frames that successfully contain landmark data.
    csv_data = []     # List to store landmark data for each frame.

    # Initialize MediaPipe's Pose solution with specified confidence thresholds.
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(mp4_file)

    # Ensure the video file was opened properly.
    if not cap.isOpened():
        raise IOError(f"Error opening video file {mp4_file}")

    # Retrieve the video frame dimensions.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Derive output directories based on the input file's path by replacing 'raw' with 'video' and 'tabular'.
    outdir = mp4_file[: mp4_file.rfind("/") + 1]
    input_filename = mp4_file[mp4_file.rfind("/") + 1:]
    videooutdir = outdir.replace("raw", "video")
    dataoutdir = outdir.replace("raw", "tabular")

    # Create output directories if they do not already exist.
    os.makedirs(videooutdir, exist_ok=True)
    os.makedirs(dataoutdir, exist_ok=True)

    # Generate output file names for stick-figure video, overlay video, and CSV files.
    inflnm, inflext = input_filename.split(".")
    out_stick_mp4 = f"{videooutdir}{inflnm}_stick.{inflext}"
    out_overlay_mp4 = f"{videooutdir}{inflnm}_overlay.{inflext}"
    out_pre_csv = f"{dataoutdir}{inflnm}_original.csv"
    out_imputed_csv = f"{dataoutdir}{inflnm}_imputed.csv"

    # Initialize video writers to save the output videos.
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_stick = cv2.VideoWriter(out_stick_mp4, fourcc, 10, (frame_width, frame_height))
    out_overlay = cv2.VideoWriter(out_overlay_mp4, fourcc, 10, (frame_width, frame_height))

    # Process each frame from the video.
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break  # End processing when there are no more frames.

        # Flip the frame horizontally for a mirror effect and convert it from BGR to RGB for MediaPipe.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Optimize by marking the image as non-writeable during processing.

        results = pose.process(image)
        if results.pose_landmarks:
            # If landmarks are detected, append them to our CSV data.
            write_landmarks_to_csv(results.pose_landmarks.landmark, frame_number, csv_data)

            # Prepare two versions of the frame for output: one with overlays and one as a blank stick-figure.
            image.flags.writeable = True
            image_overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_stick = np.zeros_like(image_overlay)

            # Draw the landmarks and their connections on both output images.
            mp_drawing.draw_landmarks(image_overlay, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image_stick, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Write the processed frames to their respective output video files.
            out_overlay.write(image_overlay)
            out_stick.write(image_stick)

            # Increment the frame counter only when valid landmark data is captured.
            frame_number += 1

    # Release all resources: close the pose solution, video capture, and video writers.
    pose.close()
    cap.release()
    out_overlay.release()
    out_stick.release()

    # Check if the video had a sufficient number of frames with valid landmark data.
    if frame_number < 50:
        log_message = (
            f"\n\n\n--------------------------------- "
            f"Not enough landmarks found in {mp4_file} "
            "---------------------------------\n\n\n"
        )
        print(log_message)
        with open("log.txt", "a+") as file:
            file.write(log_message)

        # Remove the generated videos if the frame count is insufficient.
        if os.path.exists(out_stick_mp4):
            os.remove(out_stick_mp4)
        if os.path.exists(out_overlay_mp4):
            os.remove(out_overlay_mp4)
        return

    # Write the raw landmark data to a CSV file with a header.
    with open(out_pre_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "landmark", "x", "y", "z"])
        csv_writer.writerows(csv_data)

    # Check the CSV for any duplicate entries.
    check_duplicates_in_csv(out_pre_csv)

    # Apply KNN imputation to fill in any missing data and save the final CSV.
    knn_impute_csv(out_pre_csv, out_imputed_csv)
    print(f"{mp4_file} frame count: {frame_number}")

    # Update the global minimum frame counter if this video has fewer valid frames.
    if min_num_of_frames == 0:
        min_num_of_frames = frame_number
        print(f"Initial min_num_of_frames: {min_num_of_frames}")
    elif frame_number < min_num_of_frames:
        min_num_of_frames = frame_number
        print(f"New min_num_of_frames: {min_num_of_frames}")
    else:
        print(f"min_num_of_frames remains: {min_num_of_frames}")


def process_all_mp4(input_dir: str):
    """
    Recursively processes all video files (MP4 or AVI) within the provided directory.
    
    Parameters:
      input_dir: The root directory that contains the video files.
    """
    for entry in os.listdir(input_dir):
        full_path = os.path.join(input_dir, entry)
        if os.path.isdir(full_path):
            # Recursively process directories.
            process_all_mp4(full_path)
        elif entry.lower().endswith((".mp4", ".avi")):
            # Process video files that match the extensions.
            process_mp4(full_path)


if __name__ == "__main__":
    # Set the default input directory to 'raw', unless overridden via command-line arguments.
    input_dir = "raw"
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        input_dir = sys.argv[1]

    # Begin processing all video files in the specified directory.
    process_all_mp4(input_dir)
    print(f"final min_num_of_frames: {min_num_of_frames}")
