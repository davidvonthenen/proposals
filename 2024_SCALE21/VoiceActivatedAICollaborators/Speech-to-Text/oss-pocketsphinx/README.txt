For MacOS/Linux assuming you have already installed xcode developer tools, this also requires brew installing:
- jq
- cmake
- sox


For Command Line (and C-style so libraries): Steps to run pocketsphinx against a microphone:
1. cd [install-location]/bin
2. sox -qd $(./pocketsphinx soxflags) | ./pocketsphinx - | jq -r '.t'

For Python:
1. [recommended but not required] Create a Python Environment using something like Conda
2. pip install -r requirements.txt
3. python main.py





-------

Steps to build pocketsphinx to a location of your choosing and install it there:
1. git clone the repo: https://github.com/cmusphinx/pocketsphinx
2. cd pocketsphinx
3. cmake -S . -B build
4. cmake --build build -DCMAKE_INSTALL_PREFIX=/Users/vonthd/Documents/pocketsphinx
5. cmake --build build --target install


Steps to run pocketsphinx against a WAV file (provided a WAV file in the root of the repo):
1. cd [install-location]/bin
2. sox rec.wav $(./pocketsphinx soxflags) | ./pocketsphinx single - | jq -r '.t'