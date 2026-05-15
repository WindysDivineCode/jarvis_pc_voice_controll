import sounddevice as sd
import keyboard
from scipy.io.wavfile import write
import whisper
import pyttsx3
from ollama import chat
import time
import numpy as np

PUSH_TO_TALK_KEY = "space"
STOP_KEY = "esc"

# ---------- TTS ----------

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ---------- audio record ----------
def record_audio(filename="input.wav", fs=16000):
    print(f"Hold {PUSH_TO_TALK_KEY} to talk. Release it when done.")

    while not keyboard.is_pressed(PUSH_TO_TALK_KEY):
        if keyboard.is_pressed(STOP_KEY):
            return False
        time.sleep(0.01)

    print("Listening...")
    chunks = []

    with sd.InputStream(samplerate=fs, channels=1, dtype="float32") as stream:
        while keyboard.is_pressed(PUSH_TO_TALK_KEY):
            chunk, overflowed = stream.read(int(fs * 0.1))
            if overflowed:
                print("Audio overflowed. Keep the recording a little shorter.")
            chunks.append(chunk)

    if not chunks:
        return False

    audio = np.concatenate(chunks)
    write(filename, fs, audio)
    print("Recorded.")
    return True

# ---------- whisper ----------
model = whisper.load_model("medium.en")

def transcribe(file):
    result = model.transcribe(file, language="en")
    return result["text"]
# ---------- AI ----------
def ask_ai(prompt):
    response = chat(
        model="qwen2.5:14b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ---------- MAIN LOOP ----------
print(f"Jarvis started. Hold {PUSH_TO_TALK_KEY} to talk. Press {STOP_KEY} to stop.")

while True:
    if keyboard.is_pressed(STOP_KEY):
        print("Stopping Jarvis...")
        break

    if not record_audio():
        break

    text = transcribe("input.wav")
    print("You said:", text)

    answer = ask_ai(text)
    print("Jarvis:", answer)

    speak(answer)
    time.sleep(0.01)
print("Jarvis stopped.")
