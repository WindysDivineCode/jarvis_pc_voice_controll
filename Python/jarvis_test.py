from faster_whisper import WhisperModel
import time

model_size = "small"  # or "large-v2" if v3 isn't available

# device options: "cpu", "cuda"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

start = time.time()

segments, info = model.transcribe("test2.m4a")

print("Detected language:", info.language)

text = ""
for segment in segments:
    text += segment.text

end = time.time()

print("\nTRANSCRIPTION:\n", text)
print("\nTIME TAKEN:", round(end - start, 2), "seconds")