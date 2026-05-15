import sounddevice as sd
import keyboard
from scipy.io.wavfile import write
import whisper
import pyttsx3
from ollama import chat
import time

# ---------- TTS ----------

def speak(text):
               
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ---------- audio record ----------
def record_audio(filename="input.wav", duration=4, fs=44100):
    print("Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("Recorded.")

# ---------- whisper ----------
model = whisper.load_model("base.en")

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
print("Jarvis started.")
print("Press esc to stop.")

while True:
    if keyboard.is_pressed("esc"):
        print("Stopping Jarvis...")
        break
    record_audio()
    
    text = transcribe("input.wav")
    print("You said:", text)

    answer = ask_ai(text)
    print("Jarvis:", answer)

    speak(answer)
    time.sleep(0.01)
print("Jarvis stopped.")