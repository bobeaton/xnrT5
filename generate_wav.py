from dataclasses import dataclass
import os
import sys
from typing import Optional, List

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
    tts_type: str  # "speecht5" or "generic_hf" or "vitsmodel"
    dataset_name: str = None
    speaker_index: int = None
    speaker_indices: Optional[List[int]] = None
    num_samples: int = 1
    clip_seconds: float = 3.0
    clips: int = 3
    commit: Optional[str] = None
    max_words: int = 20
    sr: int = 16000
    text_col: str = "text"
    speaker_model_type: str="xvect"
    folder: str = None
    prefix: str = None
    input_path: str = None

xnr_a = Config(
    tts_type="speecht5",
    model_name="sil-ai/xnr-tts-training-data-speecht5-a",
    commit="afb4ef469bca8093ff4b974a54824e9b05ab1dd3",
    dataset_name="sil-ai/xnr-tts-training-data",
    speaker_index=6,
    clip_seconds=3.0,
    clips=3,
    text_col="text_a",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads",
    prefix="XNR-",
    input_path=r"C:\btmp\SpeecheloCleanInLines.txt"
)

xnr_b = Config(
    tts_type="speecht5",
    model_name="sil-ai/xnr-tts-training-data-speecht5-b",
    commit="795e6ed4b0b995b3ea064b648a6164521c8b75a7",
    dataset_name="sil-ai/xnr-tts-training-data",
    speaker_index=6,
    clip_seconds=3.0,
    clips=3,
    text_col="text_b",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads",
    prefix="XNR-",
    input_path=r"C:\btmp\SpeecheloCleanInLinesNormalized.txt"
)

dgo_a = Config(
    tts_type="speecht5",
    model_name="sil-ai/dgo-tts-training-data-speecht5-a",
    commit="f07447080bee98d1d4cb5036191fd2f145fb3fe5",
    dataset_name="sil-ai/dgo-tts-training-data",
    speaker_index=1000,
    clip_seconds=5.0,
    clips=1,
    text_col="text_a",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\DogriWordDownloads",
    prefix="DOG-",
    input_path=r"C:\btmp\SpeecheloCleanInLines.txt"
)

dgo_b = Config(
    tts_type="speecht5",
    model_name="sil-ai/dgo-tts-training-data-speecht5-b",
    commit="dcda3ac3a4d0a66eb3b6fd86b9986e81644e13f4",
    dataset_name="sil-ai/dgo-tts-training-data",
    speaker_index=1000,
    clip_seconds=5.0,
    clips=1,
    text_col="text_b",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\DogriWordDownloads",
    prefix="DOG-",
    input_path=r"C:\btmp\SpeecheloCleanInLinesNormalized.txt"
)

dgohin = Config(
    tts_type="vitsmodel",
    model_name="facebook/mms-tts-hin",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\HindustaniWordDownloads",
    prefix="DGOHIN-",
    input_path=r"C:\btmp\SpeecheloCleanInLines.txt"
)

dgohinai = Config(
    tts_type="vitsmodel",
    model_name="facebook/mms-tts-hin",
    folder=r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\HindustaniWordDownloads",
    prefix="DGOHINAI-",
    input_path=r"C:\btmp\SpeecheloCleanInLines.txt"
)

CONFIG_MAP = {
    "xnr_a": xnr_a,
    "xnr_b": xnr_b,
    "dgo_a": dgo_a,
    "dgo_b": dgo_b,
    "dgohin": dgohin,
    "dgohinai": dgohinai
}

def main():
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
        if len(sys.argv) < 2:
            print("Usage: python generate_wav.py <config_name> [--save_chunks]")
            sys.exit(1)
        config_name = sys.argv[1]  # This is a string like "xnr_a" or "xnr_b"
        # optional flag to save each inferred chunk as a separate wav
        save_chunks = any(a in ("--save_chunks", "save_chunks") for a in sys.argv[2:])

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
                time.sleep(10)
                continue

            # 2. Pick the first matching file (or handle multiples if needed)
            process_file = process_files[0]
            match = re.match(r".*GenerateWavProcess_([^_]+)-([^_]+)-([^_]+)-([^\\]+)$", process_file)
            if not match:
                print("Could not parse process file name format:", process_file)
                time.sleep(2)
                continue

            lang, book_num, book_name, chapter_num = match.groups()

            if lang not in config.prefix:
                print(f"Language code in process file '{lang}' does not match the expected config prefix '{config.prefix}'")
                exit(1)

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

                # Build chunks for the entire file first (so we can process
                # the whole file for each speaker before switching speakers).
                all_text_chunks = []
                for text in lines:
                    text = re.sub(r'\s+', ' ', text.strip())

                    # Split into sentences (include Devanagari danda '।')
                    sentences = re.split(r'([.!?।]+)', text)

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

                    # Group sentences by word/phrase count, but when a sentence
                    # itself exceeds max_words, split it only at punctuation
                    # clause boundaries (commas, semicolons, colons, danda).
                    text_chunks = []
                    current_chunk = []
                    current_word_count = 0

                    def emit_current_chunk():
                        nonlocal current_chunk, current_word_count, text_chunks
                        if current_chunk:
                            text_chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_word_count = 0

                    for sentence in sentence_list:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        words = sentence.split()
                        sentence_word_count = len(words)

                        # If sentence fits within max_words, try to add to current chunk
                        if sentence_word_count <= config.max_words:
                            if current_word_count + sentence_word_count > config.max_words and current_chunk:
                                emit_current_chunk()
                            current_chunk.append(sentence)
                            current_word_count += sentence_word_count
                            continue

                        # Sentence is too long: split on clause punctuation (commas, semicolons, colons, danda)
                        clauses = re.split(r'([,;:।]+)', sentence)
                        # join clause text with following punctuation where present
                        joined_clauses = []
                        i = 0
                        while i < len(clauses):
                            if i + 1 < len(clauses) and re.match(r'[,;:।]+', clauses[i + 1]):
                                joined_clauses.append((clauses[i] + clauses[i + 1]).strip())
                                i += 2
                            else:
                                joined_clauses.append(clauses[i].strip())
                                i += 1

                        for clause in joined_clauses:
                            if not clause:
                                continue
                            cwords = clause.split()
                            c_wc = len(cwords)
                            if c_wc <= config.max_words:
                                if current_word_count + c_wc > config.max_words and current_chunk:
                                    emit_current_chunk()
                                current_chunk.append(clause)
                                current_word_count += c_wc
                            else:
                                # Clause still longer than max. Emit current chunk first,
                                # then emit the long clause as its own chunk WITHOUT
                                # splitting on raw word count (preserve punctuation boundaries).
                                emit_current_chunk()
                                text_chunks.append(clause)

                    # flush remaining
                    emit_current_chunk()

                    all_text_chunks.extend(text_chunks)

                # If multiple speaker indices specified, process the entire
                # collection of chunks for each speaker before moving on.
                # Determine speaker list: either explicit indices or single config value
                if config.speaker_indices:
                    speaker_list = config.speaker_indices
                else:
                    speaker_list = [config.speaker_index]

                for spk in speaker_list:
                    print(f"Processing speaker index: {spk}")
                    # If there is only one speaker in the list, the engine
                    # was already initialized for that speaker; avoid
                    # repeatedly calling set_speaker_index in that case.
                    if len(speaker_list) > 1:
                        tts.set_speaker_index(spk)
                    spk_audio = []
                    for ci, chunk in enumerate(all_text_chunks):
                        speech = tts.run_inference(chunk)
                        if speech is not None:
                            spk_audio.append(speech)
                        # Optionally save each chunk as an individual wav
                        if save_chunks and speech is not None:
                            # speech may be shaped (1, N) or (N,), normalize and save
                            sp = np.asarray(speech).squeeze()
                            if sp.size == 0:
                                continue
                            maxv = np.max(np.abs(sp))
                            if maxv == 0:
                                audio_int16 = np.int16(sp)
                            else:
                                audio_int16 = np.int16(sp / maxv * 32767)
                            # 1-based chunk index
                            chunk_idx = ci + 1
                            filename_chunk = filename[:-4] + f"_{spk}_{chunk_idx:02d}.wav"
                            wavfile.write(filename_chunk, config.sr, audio_int16)
                            print(f"Saved chunk {chunk_idx} to: {filename_chunk}")

                    # After all chunks for this speaker, combine and write full file
                    if config.tts_type == "vitsmodel":
                        # Flatten for vitsmodel due to variable output lengths
                        full_audio_spk = np.concatenate([audio.flatten() for audio in spk_audio])
                    else:
                        # Keep original behavior for speecht5 and generic_hf
                        full_audio_spk = np.concatenate(spk_audio)
                    audio_int16 = np.int16(full_audio_spk / np.max(np.abs(full_audio_spk)) * 32767)
                    # If multiple speakers, append speaker index to filename
                    if len(speaker_list) > 1:
                        filename_spk = filename[:-4] + f"_{spk}.wav"
                    else:
                        filename_spk = filename
                    wavfile.write(filename_spk, config.sr, audio_int16)
                    print(f"Saved combined audio to: {filename_spk}")
                
                # --- Processing Logic End ---

                # Create marker file
                with open(processed_marker, "w") as proc_f:
                    proc_f.write("processed")

                last_processed_time = mtime

            time.sleep(10)  # Check every 10 seconds

    except Exception as e:
        print("An exception occurred:")
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    main()