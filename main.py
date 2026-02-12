import json, re, threading, time
import numpy as np
import sounddevice as sd
import tkinter as tk
from faster_whisper import WhisperModel
from llama_cpp import Llama
import os
# Sample Rate and audio channels (mono)
SR = 16000
CH = 1

WHISPER_MODEL = "small.en"
LLM_GGUF = os.path.expanduser(os.environ["LLM_GGUF_PATH"])  # change to model of choice

audio_chunks = []
recording = False
lock = threading.Lock()

# callback function used by sd
def cb(indata, frames, time_info, status):
    global audio_chunks
    if status:
        pass
    with lock:
        if recording:
            audio_chunks.append(indata[:, 0].copy())

def start_rec():
    global recording, audio_chunks
    with lock:
        audio_chunks = []
        recording = True

def stop_rec():
    global recording
    with lock:
        recording = False
        # if no audio
        if not audio_chunks:
            return np.zeros((0,), dtype=np.float32)
        x = np.concatenate(audio_chunks).astype(np.float32)
    return x

def extract_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1:
        raise ValueError("non-JSON object encountered")
    return t[i:j+1]

SYSTEM = (
    "You are a command parser. Output EXACTLY ONE JSON object and NO OTHER TEXT. "
    "NO CODE FENCES. NO MARKDOWN. First char { last char }. "
    "KEYS: debug_heard,intent,parameters,confirmation_required,missing,clarification_question. "
    "If required info missing, intent=needs_clarification and ask one short question"
)

def llm_json(llm, transcript):
    prompt = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":transcript}
    ]
    response = llm.create_chat_completion(messages=prompt, temperature=0.0, max_tokens=180)
    content = response["choices"][0]["message"]["content"]
    return json.loads(extract_json(content))

def main():
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    llm = Llama(model_path=LLM_GGUF, n_gpu_layers=-1, n_ctx=1024, n_batch=512, verbose=False)

    root = tk.Tk()
    root.title("STT Parser")

    status = tk.StringVar(value="Idle")
    transcript_var = tk.StringVar(value="")
    json_var = tk.StringVar(value="")

    tk.Label(root, textvariable=status, font=("Arial", 14)).pack(pady=8)
    tk.Label(root, text="Transcript:").pack(anchor="w", padx=10)
    tk.Label(root, textvariable=transcript_var, wraplength=700, justify="left").pack(padx=10, pady=6)
    tk.Label(root, text="JSON:").pack(anchor="w", padx=10)
    tk.Label(root, textvariable=json_var, wraplength=700, justify="left").pack(padx=10, pady=6)
    
    def transcribe_and_run(audio):
        # don't take audio <200ms
        if audio.size < SR * 0.2:
            status.set("Too short.")
            return
        status.set("Transcribing...")
        # faster whisper transcribe(), two return vars: segments and info (not needed)
        segments, _ = whisper.transcribe(audio, language="en", vad_filter=True)
        text = "".join(s.text for s in segments).strip()
        transcript_var.set(text)

        if not text:
            status.set("No speech detected.")
            return

        status.set("LLM => JSON...")
        try:
            out = llm_json(llm, text)
            json_var.set(json.dumps(out, indent=2))
            status.set("Done")
        except Exception as e:
            json_var.set(f"Error: {e}")
            status.set("Error")

    def on_press(_):
        start_rec()
        status.set("Recording... (hold down)")
        transcript_var.set("")
        json_var.set("")
    # transcribe audio and set transcriptoin variable on release
    def on_release(_):
        status.set("Processing...")
        audio = stop_rec()
        threading.Thread(target=transcribe_and_run, args=(audio,), daemon=True).start()

    btn = tk.Button(root, text="Hold to Talk", width=25, height=3)
    btn.pack(pady=15)
    btn.bind("<ButtonPress-1>", on_press)
    btn.bind("<ButtonRelease-1>", on_release)
    # main loop with sound device
    with sd.InputStream(samplerate=SR, channels=CH, dtype="float32", callback=cb, blocksize=0):
        root.mainloop()

if __name__ == "__main__":
    main()
