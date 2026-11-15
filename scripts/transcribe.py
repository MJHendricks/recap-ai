from faster_whisper import WhisperModel

audio_path = "../samples/meeting1.m4a"
output_path = "../output/output.txt"
model = WhisperModel("small", device="cpu", compute_type="int8")

print("Transcribing...")

segments, info = model.transcribe(
    audio_path,
    language="en",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
    word_timestamps=True
)

# store all lines into an array
output = []
for segment in segments:
    output.append(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

print("Saving File...")
# save the array into a file, line by line.
with open(output_path,"w") as file:
    for line in output:
        file.write(line)

print("File Saved.")