"""
Fine-tune SpeechT5 TTS model on a custom dataset with speaker embeddings.

This script:
1. Loads train/validation datasets from CSVs (created by build_tts_dataset.py)
2. Loads audio files and extracts speaker embeddings
3. Fine-tunes SpeechT5 with speaker-conditional synthesis
4. Saves the fine-tuned model for inference with speaker_index control

Usage (PowerShell, based on RTX 4060 Laptop GPU):
    python finetune_speecht5.py `
      --model-name "microsoft/speecht5_tts" `
      --vocoder-name "microsoft/speecht5_hifigan" `
      --train-file "C:\\Users\\pete_\\Dropbox\\NTprogress\\PahariAudio\\KangriWordDownloads\\FCBH\\train.csv" `
      --validation-file "C:\\Users\\pete_\\Dropbox\\NTprogress\\PahariAudio\\KangriWordDownloads\\FCBH\\validation.csv" `
      --output-dir "C:\\vscode\\Models\\TTS\\Kangri" `
      --text-column text_b `
      --audio-column audio `
      --speaker-column character `
      --num-epochs 3 `
      --batch-size 8 `
      --learning-rate 5e-5 `
      --gradient-accumulation-steps 2 `
      --max-target-length 16000 `
      --audio-root "C:\\Users\\pete_\\Dropbox\\NTprogress\\PahariAudio\\KangriWordDownloads\\FCBH\\wavs"

Usage (bash, based on RTX 4060 Laptop GPU):
    python finetune_speecht5.py \
      --model-name "microsoft/speecht5_tts" \
      --vocoder-name "microsoft/speecht5_hifigan" \
      --train-file "C:\\Users\\pete_\\Dropbox\\NTprogress\\PahariAudio\\KangriWordDownloads\\FCBH\\train.csv" \
      --validation-file "C:\\Users\\pete_\\Dropbox\\NTprogress\\PahariAudio\\KangriWordDownloads\\FCBH\\validation.csv" \
      --output-dir "C:\\vscode\\Models\\TTS\\Kangri" \
      --text-column text_b \
      --audio-column audio \
      --speaker-column character \
      --num-epochs 3 \
      --batch-size 8 \
      --learning-rate 5e-5 \
      --gradient-accumulation-steps 2 \
      --max-target-length 16000 \
      --audio-root "C:\\Users\\pete_\\Dropbox\\NTprogress\\PahariAudio\\KangriWordDownloads\\FCBH\\wavs"
"""

# Recommended settings for NVIDIA GeForce RTX 4060 Laptop GPU:
#   --batch-size 8
#   --gradient-accumulation-steps 2
#   --learning-rate 5e-5
#   --eval-batch-size 16
#   --max-target-length 16000
#   --eval-steps 250
#   --save-steps 250

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset, Audio as HFAudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    Trainer,
    TrainingArguments,
)
from speechbrain.inference import EncoderClassifier
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def trim_silence_energy(x, sr, frame_ms=30, hop_ms=10, rel_db=-45.0, pad_ms=0):
    """Trim silence from audio using energy-based detection."""
    x = x.float()
    if x.ndim > 1:
        x = x.mean(dim=0)
    if x.numel() == 0:
        return x
    m = x.abs().max()
    if m > 0:
        x = x / m
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if x.numel() < frame:
        return x
    f = x.unfold(0, frame, hop)
    e = 10.0 * torch.log10(f.pow(2).mean(dim=1) + 1e-12)
    thr = e.median() + rel_db
    idx = (e > thr).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return x
    a, b = idx[0].item(), idx[-1].item()
    s = max(0, a * hop - int(sr * pad_ms / 1000))
    t = min(x.numel(), b * hop + frame + int(sr * pad_ms / 1000))
    return x[s:t]


def create_speaker_embeddings(
    audio_paths: List[str],
    speaker_labels: List[str],
    sr: int = 16000,
    clip_seconds: float = 3.0,
    crops: int = 3,
    aggregate: str = "mean",
    speaker_model_type: str = "xvect",
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Create speaker embeddings from audio files using SpeechBrain.
    
    Args:
        audio_paths: List of paths to audio files
        speaker_labels: List of speaker labels corresponding to audio files
        sr: Sample rate
        clip_seconds: Duration of each clip for embedding
        crops: Number of crops per audio file
        aggregate: How to aggregate embeddings ("mean" or "median")
        speaker_model_type: "xvect" or "ecapa"
        device: "cuda" or "cpu"
    
    Returns:
        Dictionary mapping speaker_label -> speaker_embedding (numpy array)
    """
    logger.info(f"Loading speaker recognition model: speechbrain/spkrec-{speaker_model_type}-voxceleb")
    model = EncoderClassifier.from_hparams(
        source=f"speechbrain/spkrec-{speaker_model_type}-voxceleb",
        run_opts={"device": device},
        savedir=None,
    )

    speaker_embeddings = {}
    L = int(sr * clip_seconds)

    # Group audio files by speaker
    speaker_to_audios = {}
    for audio_path, speaker_label in zip(audio_paths, speaker_labels):
        if speaker_label not in speaker_to_audios:
            speaker_to_audios[speaker_label] = []
        speaker_to_audios[speaker_label].append(audio_path)

    logger.info(f"Creating embeddings for {len(speaker_to_audios)} speakers...")

    for speaker_label, audio_files in tqdm(speaker_to_audios.items(), desc="Creating speaker embeddings"):
        waves, lengths, per_counts = [], [], []

        for audio_path in audio_files:
            try:
                # Load audio
                if not os.path.exists(audio_path):
                    logger.warning(f"Audio file not found: {audio_path}, skipping")
                    continue

                waveform, file_sr = torchaudio.load(audio_path)
                
                # Resample if needed
                if file_sr != sr:
                    resampler = torchaudio.transforms.Resample(file_sr, sr)
                    waveform = resampler(waveform)

                a = waveform.float()
                if a.ndim > 1:
                    a = a.mean(dim=0)

                # Trim silence
                a = trim_silence_energy(a, sr)

                # Create crops
                segs = []
                if a.numel() <= L:
                    segs = [a]
                else:
                    k = max(1, crops)
                    idxs = torch.linspace(0, a.numel() - L, steps=k, dtype=torch.long)
                    segs = [trim_silence_energy(a[i : i + L], sr) for i in idxs]

                per_counts.append(len(segs))
                waves.extend(segs)
                lengths.extend([s.numel() for s in segs])

            except Exception as e:
                logger.warning(f"Error loading audio {audio_path}: {e}, skipping")
                continue

        if not waves:
            logger.warning(f"No valid audio for speaker {speaker_label}, skipping")
            continue

        # Pad and create batch
        batch = pad_sequence(waves, batch_first=True)
        rel = torch.tensor(lengths, dtype=torch.float32)
        rel = rel / rel.max().clamp_min(1)

        # Compute embeddings
        with torch.no_grad():
            emb = model.encode_batch(batch.to(device), rel.to(device)).squeeze(1).cpu()

        # L2 normalization
        emb = torch.nn.functional.normalize(emb, dim=-1)

        # Aggregate embeddings by clip group
        outs, p = [], 0
        for c in per_counts:
            e = emb[p : p + c]
            v = e.mean(dim=0) if aggregate == "mean" else e.median(dim=0).values
            outs.append(v)
            p += c

        # Average across clips
        speaker_emb = torch.stack(outs).mean(dim=0).numpy()
        speaker_embeddings[speaker_label] = speaker_emb

    logger.info(f"Created embeddings for {len(speaker_embeddings)} speakers")
    return speaker_embeddings


def load_csv_dataset(
    csv_path: Path,
    audio_root: Optional[Path],
    text_column: str,
    audio_column: str,
    speaker_column: str,
) -> Dataset:
    """Load dataset from CSV and add audio paths."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    # Normalize audio paths
    if audio_root:
        df[audio_column] = df[audio_column].apply(
            lambda x: str((audio_root / x).resolve()) if x and not os.path.isabs(x) else x
        )

    # Convert to HF Dataset
    dataset = Dataset.from_pandas(df)
    return dataset


def preprocess_function(
    examples,
    processor,
    text_column: str,
    audio_column: str,
    speaker_column: str,
    speaker_embeddings: Dict[str, np.ndarray],
    sr: int = 16000,
    max_target_length: int = 8000,
):
    """Preprocess examples for SpeechT5 training."""
    # Get texts and speakers
    texts = examples[text_column]
    speakers = examples[speaker_column]

    valid_texts = []
    valid_speech = []
    valid_embeddings = []

    for text, audiopath, speakerlabel in zip(texts, examples[audio_column], speakers):
        try:
            if not os.path.exists(audiopath):
                logger.warning(f"Audio not found: {audiopath}")
                continue

            waveform, file_sr = torchaudio.load(audiopath)
            if file_sr != sr:
                waveform = torchaudio.transforms.Resample(file_sr, sr)(waveform)

            speech = waveform.squeeze(0).numpy()
            if len(speech) > max_target_length:
                speech = speech[:max_target_length]

            if speakerlabel not in speaker_embeddings:
                logger.warning(f"No embedding for speaker: {speakerlabel}")
                continue

            valid_texts.append(text)
            valid_speech.append(speech)
            valid_embeddings.append(speaker_embeddings[speakerlabel])

        except Exception as e:
            logger.warning(f"Error processing {audiopath}: {e}")
            continue

    if not valid_speech:
        return None

    # Process texts through tokenizer
    inputs = processor(text=valid_texts, sampling_rate=sr, return_tensors="pt", padding=True)

    # Load and process audio
    batch_speech_values = []
    batch_speaker_embeddings = []

    for audio_path, speaker_label in zip(examples[audio_column], speakers):
        try:
            # Load audio
            if not os.path.exists(audio_path):
                logger.warning(f"Audio not found: {audio_path}")
                continue

            waveform, file_sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if file_sr != sr:
                resampler = torchaudio.transforms.Resample(file_sr, sr)
                waveform = resampler(waveform)

            speech = waveform.squeeze(0).numpy()

            # Trim to max length
            if len(speech) > max_target_length:
                speech = speech[:max_target_length]

            batch_speech_values.append(speech)

            # Get speaker embedding
            if speaker_label in speaker_embeddings:
                batch_speaker_embeddings.append(speaker_embeddings[speaker_label])
            else:
                logger.warning(f"No embedding for speaker: {speaker_label}")

        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")
            continue

    if not batch_speech_values:
        return None

    # Pad speech values
    speech_values = processor(
        audio=batch_speech_values,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        max_length=max_target_length,
        truncation=True,
    )

    speech_key = "input_features" if "input_features" in speech_values else "input_values"

    # Stack speaker embeddings
    if batch_speaker_embeddings:
        speaker_embeddings_tensor = torch.tensor(np.array(batch_speaker_embeddings), dtype=torch.float32)
    else:
        return None

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "speaker_embeddings": speaker_embeddings_tensor,
        "speech_values": speech_values[speech_key],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SpeechT5 TTS on a custom dataset with speaker embeddings."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/speecht5_tts",
        help="HuggingFace model identifier for SpeechT5 TTS",
    )
    parser.add_argument(
        "--vocoder-name",
        type=str,
        default="microsoft/speecht5_hifigan",
        help="HuggingFace model identifier for vocoder",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--validation-file",
        type=str,
        default=None,
        help="Path to validation CSV file (optional)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in CSV",
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default="audio",
        help="Name of audio file path column in CSV",
    )
    parser.add_argument(
        "--speaker-column",
        type=str,
        default="speakerName",
        help="Name of speaker label column in CSV",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default=None,
        help="Root directory for relative audio paths",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save fine-tuned model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (optimized for RTX 4060 with fp16)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=300,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=250,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=16000,
        help="Maximum audio length in samples for each training example (16000 = 1 second)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="fp16" if torch.cuda.is_available() else "no",
        help="Mixed precision training",
    )
    parser.add_argument(
        "--speaker-model-type",
        type=str,
        choices=["xvect", "ecapa"],
        default="xvect",
        help="Speaker embedding model type",
    )

    args = parser.parse_args()

    # Validate inputs
    train_path = Path(args.train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    audio_root = Path(args.audio_root) if args.audio_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading SpeechT5 model: {args.model_name}")
    processor = SpeechT5Processor.from_pretrained(args.model_name)
    model = SpeechT5ForTextToSpeech.from_pretrained(args.model_name, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32)
    vocoder = SpeechT5HifiGan.from_pretrained(args.vocoder_name)

    logger.info(f"Loading training dataset from: {train_path}")
    train_dataset = load_csv_dataset(
        train_path,
        audio_root,
        args.text_column,
        args.audio_column,
        args.speaker_column,
    )

    if args.validation_file:
        val_path = Path(args.validation_file)
        if val_path.exists():
            logger.info(f"Loading validation dataset from: {val_path}")
            val_dataset = load_csv_dataset(
                val_path,
                audio_root,
                args.text_column,
                args.audio_column,
                args.speaker_column,
            )
        else:
            logger.warning(f"Validation file not found: {val_path}")
            val_dataset = None
    else:
        val_dataset = None

    logger.info(f"Train set size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation set size: {len(val_dataset)}")

    # Extract audio paths and speaker labels for embedding creation
    logger.info("Extracting audio and speaker information...")
    train_audio_paths = [path for path in train_dataset[args.audio_column]]
    train_speaker_labels = [label for label in train_dataset[args.speaker_column]]

    # Create speaker embeddings
    speaker_embeddings = create_speaker_embeddings(
        train_audio_paths,
        train_speaker_labels,
        sr=16000,
        clip_seconds=3.0,
        crops=3,
        aggregate="mean",
        speaker_model_type=args.speaker_model_type,
        device=args.device,
    )

    # Save speaker embeddings mapping for later use
    embeddings_meta = {
        speaker: emb.tolist() for speaker, emb in speaker_embeddings.items()
    }
    with open(output_dir / "speaker_embeddings.json", "w") as f:
        json.dump(embeddings_meta, f, indent=2)
    logger.info(f"Saved speaker embeddings mapping to {output_dir / 'speaker_embeddings.json'}")

    # Preprocess datasets
    logger.info("Preprocessing datasets...")

    def preprocess_fn(batch):
        return preprocess_function(
            batch,
            processor,
            args.text_column,
            args.audio_column,
            args.speaker_column,
            speaker_embeddings,
            max_target_length=args.max_target_length,
        )

    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=8,
        desc="Processing training data",
    )

    if val_dataset:
        val_dataset = val_dataset.map(
            preprocess_fn,
            batched=True,
            batch_size=8,
            desc="Processing validation data",
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        seed=args.seed,
        fp16=args.mixed_precision == "fp16",
        bf16=args.mixed_precision == "bf16",
        optim="adamw_8bit" if args.device == "cuda" else "adamw_torch",
        report_to="tensorboard",
    )

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving fine-tuned model to: {output_dir}")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    vocoder.save_pretrained(str(output_dir / "vocoder"))

    logger.info("Fine-tuning complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Speaker embeddings saved to: {output_dir / 'speaker_embeddings.json'}")
    logger.info("You can now use this model with different speaker embeddings for inference!")


if __name__ == "__main__":
    main()
