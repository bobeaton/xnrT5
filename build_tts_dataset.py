import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def detect_delimiter(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "|", "\t"])
        return dialect.delimiter
    except csv.Error:
        return ","


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    delimiter = "|" #detect_delimiter(metadata_path)
    df = pd.read_csv(metadata_path, sep=delimiter, dtype=str, keep_default_na=False)

    # if df.shape[1] == 1 and delimiter != "|":
    #     # Fall back to pipe-splitting if the file appears to be pipe-delimited data.
    #     raw_lines = metadata_path.read_text(encoding="utf-8").splitlines()
    #     rows = [line.split("|") for line in raw_lines if line.strip()]
    #     if len(rows) > 0 and len(rows[0]) >= 5:
    #         header = rows[0]
    #         data = rows[1:]
    #         df = pd.DataFrame(data, columns=header[: len(data[0])])

    expected_cols = {"audio_file", "character", "speakerName", "transcriptionA", "transcriptionB"}
    lower_cols = {c.lower(): c for c in df.columns}
    if not expected_cols.issubset(set(df.columns)) and expected_cols.issubset({c.lower() for c in df.columns}):
        df.rename(columns={lower_cols[k]: k for k in expected_cols if k.lower() in lower_cols}, inplace=True)

    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata CSV is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def normalize_audio_path(audio_file: str, audio_root: Optional[Path]) -> str:
    audio_file = audio_file.strip()
    if not audio_file:
        return ""
    if os.path.isabs(audio_file):
        return audio_file
    if audio_root is None:
        return audio_file
    return str((audio_root / audio_file).resolve())


def build_dataset_rows(
    metadata: pd.DataFrame,
    text_field: str,
    audio_root: Optional[Path],
    include_fields: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    include_fields = include_fields or ["audio", "text", "text_a", "text_b", "character", "speakerName", "audio_file"]
    rows = []
    for idx, row in metadata.iterrows():
        audio_file = row["audio_file"]
        text_a = row.get("transcriptionA", "").strip()
        text_b = row.get("transcriptionB", "").strip()
        text = text_a if text_field == "transcriptionA" else text_b
        if not text:
            continue

        entry = {
            "audio": normalize_audio_path(audio_file, audio_root),
            "text": text,
            "text_a": text_a,
            "text_b": text_b,
            "character": row.get("character", "").strip(),
            "speakerName": row.get("speakerName", "").strip(),
            "audio_file": audio_file,
        }
        rows.append({k: entry[k] for k in include_fields if k in entry})
    return rows


def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {out_path}")
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows(rows: List[Dict[str, str]], val_ratio: float) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if val_ratio <= 0.0:
        return rows, []
    split_index = max(1, int(len(rows) * (1.0 - val_ratio)))
    return rows[:split_index], rows[split_index:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SpeechT5-compatible TTS dataset from a metadata CSV.")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV file")
    parser.add_argument("--audio-root", required=False, default=None, help="Audio root directory for relative audio file names")
    parser.add_argument("--text-field", choices=["transcriptionA", "transcriptionB"], default="transcriptionB",
                        help="Which transcription column to use for model text")
    parser.add_argument("--output-dir", required=True, help="Destination directory for generated dataset files")
    parser.add_argument("--valid-ratio", type=float, default=0.0,
                        help="Fraction of examples to write into validation set")
    parser.add_argument("--output-format", choices=["csv", "jsonl"], default="csv",
                        help="Output format for dataset files")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    audio_root = Path(args.audio_root) if args.audio_root else None
    out_dir = Path(args.output_dir)

    metadata = load_metadata(metadata_path)
    rows = build_dataset_rows(metadata, args.text_field, audio_root)
    if not rows:
        raise ValueError("No dataset rows were built from metadata. Check metadata file and text field selection.")

    train_rows, val_rows = split_rows(rows, args.valid_ratio)

    if args.output_format == "csv":
        write_csv(train_rows, out_dir / "train.csv")
        if val_rows:
            write_csv(val_rows, out_dir / "validation.csv")
    else:
        import json
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
            for row in train_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if val_rows:
            with (out_dir / "validation.jsonl").open("w", encoding="utf-8") as f:
                for row in val_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_rows)} train examples")
    if val_rows:
        print(f"Wrote {len(val_rows)} validation examples")
    print(f"Dataset files created under: {out_dir}")
    print("Each row includes audio path and text, plus speaker/character metadata.")

# cd c:\vscode\xnrT5
# python build_tts_dataset.py `
#   --metadata "C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\metadata.csv" `
#   --audio-root "C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\wavs" `
#   --output-dir "C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH" `
#   --text-field transcriptionB `
#   --valid-ratio 0.05 `
#   --output-format csv

if __name__ == "__main__":
    main()
