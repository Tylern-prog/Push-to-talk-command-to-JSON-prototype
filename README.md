# Push-to-talk command to JSON prototype
## Overview
This project is a prototype that demonstrates an end-to-end workflow for voice-driven simulator command entry"

1. Push-to-talk audio capture
2. Local speech-to-text processing using faster-whisper
3. Local LLM inference using llama.cpp (via llama-cpp-python library)
4. sound JSON output roughly representing a simulator configuration command

This prototype is focused on establishing an integration pattern and being a "proof-of-concept". It is not production ready and does not include any safety validation or command library integration

## What this prototype does

- Records audio while user holds the "hold to talk" button
- Transcribes audio to text via Whisper
- converts command transcript into a JSON object using local llm
- Displays the following:
- the transcript
- the JSON command

This app can also be adapter to send the JSON to a c# receiver app via HTTP

## Technologies used

### Audio Input

- ***sounddevice*** for microphone input

### Speech-to-text

- ***faster-whisper:*** fast and efficient implementation of whisper using CTranslate2
- ***Model:*** whisper-small.en

### Local LLM Inference

- ***llama-cpp-python*** provides python bindings for llama.cpp and chat completion functionality for model interaction and inference
- ***Model:*** Microsoft's Phi-3.5 Mini Instruct quantitized and converted to GGUF

### UI
- minimalist Tkinter UI for push-to-talk button and output

## Setup (for Ubuntu on AMDGPU)

### System installations

sudo apt-get update

sudo apt-get install python3-venv libportaudio2 portaudio19-dev portaudio19-doc ffmpeg python3-numpy (if below pip install throws error) python3-tk

### Python installations

python3 -m virtenv virtenv

source .virtend/bin/activate

pip install -U pip


pip install sounddevice

pip install numpy

pip install faster-whisper

export FORCE_CMAKE=1

export CMAKE_ARGS="-DGGML_VULKAN=on"

pip install llama-cpp-python

### Model download (for inference)

Link: https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF

NOTE: whisper model will download at runtime

### Running the app
1. In terminal run: 'export LLM_GGUF_PATH="/path/to/model.gguf"'
2. execute: python3 main.py
3. hold the the "hold to talk" button and give a command


