import cv2
import imutils
import numpy as np
import argparse

# Global variable for the HOG + SVM model (instantiated in main)
HOGCV = None

def detect(frame, threshold):
    """
    Takes a single frame (image) and detects persons in it using detectMultiScale.
    Only draws bounding boxes around each detection if the confidence 'weight' is
    above the provided 'threshold'. Returns the processed frame.
    """
    # Detect persons in the frame
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(
        frame,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.03
    )

    person_count = 0
    # Zip bounding boxes with their associated weights (confidence)
    for (x, y, w, h), weight in zip(bounding_box_cordinates, weights):
        # Only consider detections above the threshold
        if weight < threshold:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person_count+1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
        person_count += 1

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX,
                0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person_count}', (40, 70),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('output', frame)
    return frame

def detectByCamera(writer, threshold):
    """
    Captures video from the system's default webcam (device index 0).
    Applies the detect() function on each frame (filtered by 'threshold'),
    optionally writes the processed frames to a video file, and displays them in a window.
    Press 'q' to quit the capture loop.
    """
    video = cv2.VideoCapture(0)
    print('[INFO] Detecting people via webcam...')

    while True:
        check, frame = video.read()
        if not check:
            break

        frame = detect(frame, threshold)
        
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def detectByPathVideo(path, writer, threshold):
    """
    Reads a video from a file path. For each frame, it resizes,
    detects persons above the 'threshold', writes the processed frame to a video file
    (if writer is provided), and displays in a window.
    Press 'q' to quit the loop.
    """
    video = cv2.VideoCapture(path)
    check, frame = video.read()

    if not check:
        print('Video not found. Please provide a valid path (full path to the video file).')
        return

    print('[INFO] Detecting people in video...')

    while video.isOpened():
        check, frame = video.read()
        if not check:
            break

        # Resize frame to a max width of 800 for faster processing
        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        frame = detect(frame, threshold)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def detectByPathImage(path, output_path, threshold):
    """
    Reads a single image from a file path, resizes it,
    detects persons above the 'threshold', and optionally saves the output image to disk.
    Displays the output window until a key is pressed.
    """
    image = cv2.imread(path)
    if image is None:
        print('Image not found. Please provide a valid image path.')
        return

    image = imutils.resize(image, width=min(800, image.shape[1]))
    result_image = detect(image, threshold)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def humanDetector(args):
    """
    Decides whether to open the webcam, read a video file, or read an image
    based on the argparse arguments. Sets up a video writer if specified.
    Passes the 'confidence_threshold' to detection methods to filter out low-confidence detections.
    """
    image_path = args["image"]
    video_path = args['video']
    camera_arg = str(args["camera"]).lower()  # handle 'True' or 'true'
    camera = (camera_arg == 'true')
    confidence_threshold = float(args["confidence"])

    writer = None
    # Create the video writer only if we're dealing with a video/camera scenario
    # (not a single image) and an output path was provided
    if args['output'] is not None and image_path is None:
        # Adjust resolution for your output (width=600, height=600 as in the original sample)
        writer = cv2.VideoWriter(
            args['output'],
            cv2.VideoWriter_fourcc(*'MJPG'),
            10,
            (600, 600)
        )

    if camera:
        print('[INFO] Opening webcam...')
        detectByCamera(writer, confidence_threshold)
    elif video_path is not None:
        print('[INFO] Opening video from path...')
        detectByPathVideo(video_path, writer, confidence_threshold)
    elif image_path is not None:
        print('[INFO] Opening image from path...')
        detectByPathImage(image_path, args['output'], confidence_threshold)

def argsParser():
    """
    Parses command-line arguments for controlling input sources (image, video, webcam),
    optional output video file path, and a confidence threshold.
    """
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None,
                           help="Path to the video file.")
    arg_parse.add_argument("-i", "--image", default=None,
                           help="Path to the image file.")
    arg_parse.add_argument("-c", "--camera", default=False,
                           help="Set to true if you want to use the webcam.")
    arg_parse.add_argument("-o", "--output", type=str,
                           help="Path to an optional output video file.")
    arg_parse.add_argument("-conf", "--confidence", default=0.5, type=float,
                           help="Confidence threshold for the detection (0 to 1).")
    args = vars(arg_parse.parse_args())
    return args

if __name__ == "__main__":
    # Initialize the HOG descriptor and SVM detector
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Parse the arguments
    args = argsParser()

    # Run detection logic
    humanDetector(args)
