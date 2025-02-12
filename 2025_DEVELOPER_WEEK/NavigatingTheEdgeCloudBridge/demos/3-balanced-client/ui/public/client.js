// main.js

const PLAY_STATES = {
    NO_AUDIO: "no_audio",
    LOADING: "loading",
    PLAYING: "playing",
};

let playState = PLAY_STATES.NO_AUDIO;
let audioChunks = [];
let socket;
let microphone;

const playButton = document.getElementById("play-button");
const recordButton = document.getElementById("record");

function initWebSocket() {
    if (!socket) {
        // socket = new WebSocket("ws://localhost:3000");
        socket = new WebSocket("ws://127.0.0.1:3000");

        socket.addEventListener("open", () => {
            console.log("WebSocket opened");
            playState = PLAY_STATES.PLAYING;
            updatePlayButton();
        });

        socket.addEventListener("message", handleWebSocketMessage);

        socket.addEventListener("close", () => {
            console.log("WebSocket closed");
            playState = PLAY_STATES.NO_AUDIO;
            updatePlayButton();
        });

        socket.addEventListener("error", (error) => {
            console.error("WebSocket error:", error);
            playState = PLAY_STATES.NO_AUDIO;
            updatePlayButton();
        });
    }
}

function handleWebSocketMessage(event) {
    if (typeof event.data === "string") {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === "Flushed") {
                console.log("Audio Flushed");
                playAudioChunks();
            } else {
                console.log("Message from server:", msg);
            }
        } catch (e) {
            // Not JSON, might be other server text messages
            console.log("Text message from server:", event.data);
        }
    } else if (event.data instanceof Blob) {
        console.log("Incoming TTS audio blob");
        audioChunks.push(event.data);
    }
}

function updatePlayButton() {
    if (!playButton) return;
    const icon = playButton.querySelector(".button-icon");
    if (!icon) return;
    switch (playState) {
        case PLAY_STATES.NO_AUDIO:
            icon.className = "button-icon fa-solid fa-play";
            break;
        case PLAY_STATES.LOADING:
            icon.className = "button-icon fa-solid fa-circle-notch";
            break;
        case PLAY_STATES.PLAYING:
            icon.className = "button-icon fa-solid fa-stop";
            break;
    }
}

function stopMicrophone() {
    // No traditional audio element used, we rely on AudioContext
    // Just set state to NO_AUDIO since decoding is done in memory
    playState = PLAY_STATES.NO_AUDIO;
    updatePlayButton();

    // stop microphone
    stopRecording().catch((error) => console.error("Error stopping:", error));

    // stop websocket connection
    if (socket && socket.readyState === WebSocket.OPEN) {
        const stopData = {
            type: "transcription_control",
            action: "stop",
        };
        socket.send(JSON.stringify(stopData));
    }
}

function playAudioChunks() {
    const blob = new Blob(audioChunks, { type: "audio/wav" });
    if (window.MediaSource) {
        const audioContext = new AudioContext();
        const reader = new FileReader();
        reader.onload = function () {
            const arrayBuffer = this.result;
            audioContext.decodeAudioData(arrayBuffer, (buffer) => {
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                source.start();
                playState = PLAY_STATES.PLAYING;
                updatePlayButton();

                source.onended = () => {
                    audioChunks = [];
                    playState = PLAY_STATES.NO_AUDIO;
                    updatePlayButton();
                };
            });
        };
        reader.readAsArrayBuffer(blob);
    } else {
        console.error("MP4 audio NOT supported");
    }
    audioChunks = [];
}

async function getMicrophone() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    return new MediaRecorder(stream, { mimeType: "audio/webm" });
}

async function openMicrophone(mic) {
    return new Promise((resolve) => {
        mic.onstart = () => {
            console.log("Microphone opened");
            document.body.classList.add("recording");
            resolve();
        };
        mic.ondataavailable = (event) => {
            if (event.data.size > 0 && socket && socket.readyState === WebSocket.OPEN) {
                // Send raw binary audio data to server for transcription
                socket.send(event.data);
            }
        };
        mic.start(1000);
    });
}

async function startRecording() {
    microphone = await getMicrophone();
    await openMicrophone(microphone);
}

async function stopRecording() {
    microphone.stop();
    microphone.stream.getTracks().forEach((track) => track.stop());
    const stopData = {
        type: "transcription_control",
        action: "stop",
    };
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(stopData));
    }
    microphone = null;
    document.body.classList.remove("recording");
    console.log("Microphone closed");
}

// Use "change" event to detect when the checkbox is toggled
recordButton.addEventListener("change", async () => {
    initWebSocket();

    // log value of recordButton.checked
    console.log("recordButton.checked:", recordButton.checked);

    if (recordButton.checked) {
        playState = PLAY_STATES.PLAYING;
        updatePlayButton();

        const startData = {
            type: "transcription_control",
            action: "start",
        };

        // poll until the socket is open before sending start message
        while (socket.readyState !== WebSocket.OPEN) {
            // sleep for 100ms
            await new Promise((resolve) => setTimeout(resolve, 100));
        }

        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(startData));
        }
        console.log("Start recording");
        startRecording().catch((error) => console.error("Error starting:", error));
    } else if (!recordButton.checked) {
        console.log("Stop recording");
        stopRecording().catch((error) => console.error("Error stopping:", error));
    }
});

if (playButton) {
    playButton.addEventListener("click", playButtonClick);
}
