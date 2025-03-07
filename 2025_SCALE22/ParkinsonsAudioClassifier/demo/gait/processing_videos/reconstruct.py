import os
import sys

import cv2


def reconstruct_video(frames_dir, output_mp4, fps=24):
    """Reconstructs a video from individual frames."""

    # Create the base directory for the output path if it doesn't exist
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)

    # Get the list of image files in the directory
    image_files = [
        img
        for img in os.listdir(frames_dir)
        if img.endswith(".jpg") or img.endswith(".png")
    ]
    image_files.sort()

    # Read the first frame to get the video dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, image_files[0]))
    height, width, _ = first_frame.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    # Write each frame to the video
    for image_file in image_files:
        frame = cv2.imread(os.path.join(frames_dir, image_file))
        video.write(frame)

    # Release the VideoWriter object
    video.release()


# iterate over all mp4 files in the raw directory
def process_all_mp4(inputdir: str):
    for item in os.listdir(inputdir):
        if item == ".DS_Store":
            continue

        root_dir = f"{inputdir}/{item}"
        # print(f"Processing item: {root_dir}")

        at_least_one_file = False
        for file in os.listdir(root_dir):
            check_item = f"{root_dir}/{file}"
            if not os.path.isdir(check_item) and file != ".DS_Store":
                at_least_one_file = True
                # print(f"File found: {check_item}")
                break

        if at_least_one_file:
            input = root_dir
            output = root_dir.replace("raw", "video")
            output = output.replace("/normalization", "") + ".mp4"
            print("\n")
            print(f"Processing dir: {input}")
            print(f"Processing mp4: {output}")
            print("\n")
            reconstruct_video(input, output, fps=24)
        else:
            # print(f"Processing directory: {root_dir}")
            process_all_mp4(root_dir)


if __name__ == "__main__":
    inputdir = "raw"
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        inputdir = sys.argv[1]
    process_all_mp4(inputdir)
