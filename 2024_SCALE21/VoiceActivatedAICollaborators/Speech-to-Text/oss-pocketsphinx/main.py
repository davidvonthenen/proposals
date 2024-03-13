from pocketsphinx import LiveSpeech


def main():
    for phrase in LiveSpeech():
        print(phrase)


if __name__ == "__main__":
    main()
