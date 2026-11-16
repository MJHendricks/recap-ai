from faster_whisper import WhisperModel
import sys
import json

output_path = ""

def main(path = "../samples/meeting1.m4a",size="small"):
    model = WhisperModel(size, device="cpu", compute_type="int8")
    print("Transcribing...")

    segments, info = model.transcribe(
        path,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True
    )

    # output
    json_lines = []
    plaintext_lines = []
    for segment in segments:
        clean_text = segment.text.strip()

        # without timestamps for txt file
        plaintext_lines.append(clean_text)
        # with timestamps for json file
        json_lines.append({"start":segment.start,"end":segment.end,"text":clean_text})

    print("Saving File...")
    # save json
    with open("../output/output.json","w",encoding="utf-8") as json_file:
        json.dump(json_lines,json_file,indent=4,ensure_ascii=False)

    # save txt
    with open("../output/output.txt","w") as text_file:
        text_file.write(" ".join(plaintext_lines))

    print("Files Saved.")

main()


