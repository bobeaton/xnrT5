from dataclasses import dataclass
import os
import sys
from typing import Optional

if os.name == 'nt':
    import msvcrt
    for stream in (sys.stdout, sys.stderr):
        try:
            msvcrt.setmode(stream.fileno(), os.O_TEXT)
        except Exception:
            pass

# ...rest of your imports and code...
# from logging import config
import sys
from xnrTts import TTSEngine
import uroman as ur
uroman = ur.Uroman()

@dataclass
class Config:
    model_name: str
    dataset_name: str
    speaker_index: int
    num_samples: int = 10
    clip_seconds: float = 3.0
    clips: int = 3
    commit: Optional[str] = None
    max_words: int = 40
    sr: int = 16000
    text_col: str = "text"
    speaker_model_type: str="xvect"
    folder: str = None
    prefix: str = None
    input_path: str = None

# LANG_MAP = {
#     "ENG": {
#         "folder": r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\EnglishDownloads",
#         "model": "facebook/mms-tts-en",
#         "prefix": "ENG-",
#         "snapshot": r"C:\Users\pete_\.cache\huggingface\hub\models--facebook--mms-tts-en\snapshots\<your-snapshot-id>"
#     },
#     "XNR": {
#         "folder": r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads",
#         "model": "facebook/mms-tts-xnr",
#         "prefix": "XNR-",
#         "snapshot": r"C:\Users\pete_\.cache\huggingface\hub\models--facebook--mms-tts-xnr\snapshots\7aa0e29aa0a30d2abf5553581c237bf3573b50fd",
#         "model_name": "sil-ai/xnr-tts-training-data-speecht5",
#         "dataset_name": "sil-ai/xnr-tts-training-data",
#         "commit": "a3d5c09d8a85d49a14a594623c205dd07f13650a",
#         "speaker_index": 6
#     },
#     "DOG": {
#         "folder": r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\DogriWordDownloads",
#         "model": "facebook/mms-tts-dgo",
#         "prefix": "DOG-",
#         "snapshot": r"C:\Users\pete_\.cache\huggingface\hub\models--facebook--mms-tts-dgo\snapshots\<your-snapshot-id>",
#         "model_name": "sil-ai/dgo-tts-training-data-speecht5",
#         "dataset_name": "sil-ai/dgo-tts-training-data",
#         "commit": "503eb0135e9c88d2f187dc4311b4134d04d0aae7",
#         "speaker_index": 285
#     },
#     "DGOHIN": {
#         "folder": r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\HindustaniWordDownloads",
#         "model": "facebook/mms-tts-hin",
#         "prefix": "DGOHIN-",
#         "snapshot": r"C:\Users\pete_\.cache\huggingface\hub\models--facebook--mms-tts-hin\snapshots\<your-snapshot-id>"
#     },
#     "HIN": {
#         "folder": r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\HindiWordDownloads",
#         "model": "facebook/mms-tts-hin",
#         "prefix": "HIN-",
#         "snapshot": r"C:\Users\pete_\.cache\huggingface\hub\models--facebook--mms-tts-hin\snapshots\<your-snapshot-id>"
#     }
# }

xnr_a = Config(
    model_name="sil-ai/xnr-tts-training-data-a-speecht5",
    dataset_name="sil-ai/xnr-tts-training-data",
    speaker_index=6,
    clip_seconds=3.0,
    clips=3,
    commit="7689c758a9d1300dddd97e2bed75f0047650871c",
    text_col="text_a",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads",
    prefix="XNR-",
    input_path=r"C:\btmp\SpeecheloCleanInLines.txt"
)

xnr_b = Config(
    model_name="sil-ai/xnr-tts-training-data-b-speecht5",
    dataset_name="sil-ai/xnr-tts-training-data",
    speaker_index=6,
    clip_seconds=3.0,
    clips=3,
    commit="c3606074764ba6cc6ef51a2e46076af2a47b62b4",
    text_col="text_b",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads",
    prefix="XNR-",
    input_path=r"C:\btmp\SpeecheloCleanInLinesNormalized.txt"
)

dgo_a = Config(
    model_name="sil-ai/dgo-tts-training-data-a-speecht5",
    dataset_name="sil-ai/dgo-tts-training-data",
    speaker_index=285,
    clip_seconds=3.0,
    clips=3,
    commit="8ebc0aa1003b602a3b648802602334c734b33ae9",
    text_col="text_a",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\DogriWordDownloads",
    prefix="DOG-",
    input_path=r"C:\btmp\SpeecheloCleanInLines.txt"
)

dgo_b = Config(
    model_name="sil-ai/dgo-tts-training-data-b-speecht5",
    dataset_name="sil-ai/dgo-tts-training-data",
    speaker_index=285,
    clip_seconds=3.0,
    clips=3,
    commit="3cf20df9bbc61fde0737ab34cd10de0012c90898",
    text_col="text_b",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\DogriWordDownloads",
    prefix="DOG-",
    input_path=r"C:\btmp\SpeecheloCleanInLinesNormalized.txt"
)

CONFIG_MAP = {
    "xnr_a": xnr_a,
    "xnr_b": xnr_b,
    "dgo_a": dgo_a,
    "dgo_b": dgo_b
}

def main(max_words = 20):  # max is 600, but the join adds spaces, so use something less
    import logging
    import traceback
    import numpy as np
    import time
    import re
    import glob
    import scipy.io.wavfile as wavfile 
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
    print("Starting generate_wav.py (print statement)")

    try:
        if len(sys.argv) != 2:
            print("Usage: python generate_wav.py <config_name> (i.e. xnr_a or xnr_b)")
            sys.exit(1)
        config_name = sys.argv[1]  # This is a string like "xnr_a" or "xnr_b"

        config = CONFIG_MAP.get(config_name)
        if not config:
            print(f"Unknown language configuration: {config_name}. Available configs: {', '.join(CONFIG_MAP.keys())}")
            sys.exit(1)
        print(f"Config: {config}")

        # print("Initializing TTSEngine...")
        tts = TTSEngine(config)
        print("TTSEngine initialized.")

        input_path = config.input_path
        processed_marker = r"c:\btmp\SpeecheloCleanInLinesProcessed"
        done_processing_marker = r"c:\btmp\SpeecheloCleanInLinesProcessDone"

        file_stat = os.stat(input_path)
        last_processed_time = None

        while True:

            # exit if we see the done processing marker
            if os.path.exists(done_processing_marker):
                return
        
            # 1. Locate the trigger file matching the pattern:
            process_pattern = r"c:\btmp\GenerateWavProcess_*-*-*-*"
            process_files = glob.glob(process_pattern)

            if not process_files:
                print("No process instruction file found, skipping.")
                time.sleep(2)
                continue

            # 2. Pick the first matching file (or handle multiples if needed)
            process_file = process_files[0]
            match = re.match(r".*GenerateWavProcess_([^_]+)-([^_]+)-([^_]+)-([^\\]+)$", process_file)
            if not match:
                print("Could not parse process file name format:", process_file)
                time.sleep(2)
                continue

            lang, book_num, book_name, chapter_num = match.groups()

            # Proceed with the language map and filename as usual:
            # if lang not in LANG_MAP:
            #     print(f"Unknown language code: {lang}")
            #     time.sleep(2)
            #     continue

            # config = LANG_MAP[lang]
            filename = f"{config.folder}\\{config.prefix}{book_num}-{book_name}-{int(chapter_num):03d}.wav"

            print(f"Parsed process instruction for: lang={lang}, book_num={book_num}, book_name={book_name}, chapter_num={chapter_num}")
            print(f"Input filename: {input_path}, Output filename: {filename}")

            # create array to accumulate audio segments
            all_audio = []  # Will hold all audio segments

            if not os.path.exists(input_path):
                time.sleep(2)
                continue

            file_stat = os.stat(input_path)
            mtime = file_stat.st_mtime
            now = time.time()

            # Only proceed if file has a new timestamp, is at least 1 second old, and processed marker does not exist
            if (last_processed_time is None or mtime != last_processed_time) \
                and (now - mtime > 1) \
                and not os.path.exists(processed_marker):

                print("Detected new or updated input file, processing...")
                # --- Processing Logic ---
                # Read input text
                with open(input_path , "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                for text in lines:
                    text = re.sub(r'\s+', ' ', text.strip())

                    # Split into sentences
                    sentences = re.split(r'([.!?]+)', text)

                    # Recombine sentences with their punctuation
                    sentence_list = []
                    for i in range(0, len(sentences) - 1, 2):
                        if i + 1 < len(sentences):
                            sentence = (sentences[i] + sentences[i + 1]).strip()
                            if sentence:
                                sentence_list.append(sentence)

                    # Handle case where text doesn't end with punctuation
                    if len(sentences) % 2 == 1 and sentences[-1].strip():
                        sentence_list.append(sentences[-1].strip())

                    # Group sentences by word count
                    text_chunks = []
                    current_chunk = []
                    current_word_count = 0

                    for sentence in sentence_list:
                        sentence_word_count = len(sentence.split())
                        if current_word_count + sentence_word_count > config.max_words and current_chunk:
                            text_chunks.append(' '.join(current_chunk))
                            current_chunk = [sentence]
                            current_word_count = sentence_word_count
                        else:
                            current_chunk.append(sentence)
                            current_word_count += sentence_word_count
                    if current_chunk:
                        text_chunks.append(' '.join(current_chunk))
                    for chunk in text_chunks:
                        speech = tts.run_inference(chunk)
                        all_audio.append(speech)
                        
                # Combine all audio and save
                full_audio = np.concatenate(all_audio)
                audio_int16 = np.int16(full_audio / np.max(np.abs(full_audio)) * 32767)
                wavfile.write(filename, config.sr, audio_int16)
                print(f"Saved combined audio to: {filename}")

                # --- Processing Logic End ---

                # Create marker file
                with open(processed_marker, "w") as proc_f:
                    proc_f.write("processed")

                last_processed_time = mtime

            time.sleep(2)  # Check every 2 seconds

    except Exception as e:
        print("An exception occurred:")
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    main()