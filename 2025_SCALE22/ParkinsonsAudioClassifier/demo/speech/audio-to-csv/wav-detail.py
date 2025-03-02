import os
import wave

def get_wav_info(directory):
    """
    Recursively search for WAV files in the given directory and print their sample rates.
    :param directory: The root directory to start the search from.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    with wave.open(file_path, 'r') as wav_file:
                        sample_rate = wav_file.getframerate()
                        print(f"File: {file}, Sample Rate: {sample_rate} Hz")
                except wave.Error as e:
                    print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    folder_path = "./clips"
    if os.path.isdir(folder_path):
        get_wav_info(folder_path)
    else:
        print("Invalid directory path.")