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

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Debugging helpers: dump a compact summary of the first few collator batches
from pathlib import Path

_COLLATOR_DEBUG_FILE: Path = Path("collator_debug_batches.jsonl")
_COLLATOR_DEBUG_COUNT: int = 0
_COLLATOR_DEBUG_MAX: int = 3

def _safe_shape(obj):
    try:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return tuple(obj.shape)
        if isinstance(obj, np.ndarray):
            return tuple(obj.shape)
        # list-like
        if hasattr(obj, "__len__"):
            if len(obj) == 0:
                return (0,)
            first = obj[0]
            if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
                return (len(obj), len(first))
            return (len(obj),)
    except Exception:
        return "unknown"
    return None

@dataclass
class SpeechT5DataCollator:
    processor: SpeechT5Processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Drop any examples that ended up with empty input_ids
        features = [f for f in features if f.get("input_ids")]
        if not features:
            # Nothing valid in this batch; let Trainer skip it
            return {}

        # Drop examples missing audio/spectrogram targets (`speech_values`) —
        # these can appear if preprocessing produced inconsistent records.
        missing_speech_count = sum(1 for f in features if not f.get("speech_values"))
        if missing_speech_count:
            logger.warning("Dropping %d examples from batch: missing speech_values", missing_speech_count)
            # Write an immediate debug summary before filtering so we capture the
            # original batch contents that triggered the drop (this can happen
            # before the later debug write location when the batch becomes empty).
            try:
                import os
                proc_info = {"pid": os.getpid()}
                summaries = []
                for f in features:
                    summaries.append(
                        {
                            "keys": sorted(list(f.keys())),
                            "has_input_ids": bool(f.get("input_ids")),
                            "input_ids_len": len(f.get("input_ids")) if f.get("input_ids") else None,
                            "has_speech_values": bool(f.get("speech_values")),
                            "speech_values_shape": _safe_shape(f.get("speech_values")),
                            "has_speaker_embeddings": f.get("speaker_embeddings") is not None,
                            "speaker_embeddings_len": len(f.get("speaker_embeddings")) if f.get("speaker_embeddings") is not None else None,
                        }
                    )
                with open(_COLLATOR_DEBUG_FILE, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"drop_dump": True, "proc": proc_info, "batch_size": len(features), "summaries": summaries}) + "\n")
            except Exception as e:
                logger.debug("Failed to write immediate collator drop summary: %s", e)
            features = [f for f in features if f.get("speech_values")]

        if not features:
            # Nothing valid after removing missing speech_values; skip batch
            return {}

        # Log the batch structure before padding so we can diagnose missing fields
        first_feature = features[0]
        logger.info(
            "Collator batch: size=%d, keys=%s, input_ids_type=%s, input_ids_len=%s",
            len(features),
            sorted(first_feature.keys()),
            type(first_feature.get("input_ids")).__name__,
            len(first_feature.get("input_ids")) if first_feature.get("input_ids") is not None else None,
        )

        # Transient debug: write a compact summary of the first few batches to disk
        global _COLLATOR_DEBUG_COUNT
        try:
            if _COLLATOR_DEBUG_COUNT < _COLLATOR_DEBUG_MAX:
                # input("Paused at collator; press Enter to continue...")
                summaries = []
                for f in features:
                    s = {
                        "keys": sorted(list(f.keys())),
                        "has_input_ids": bool(f.get("input_ids")),
                        "input_ids_len": len(f.get("input_ids")) if f.get("input_ids") else None,
                        "has_speech_values": bool(f.get("speech_values")),
                        "speech_values_shape": _safe_shape(f.get("speech_values")),
                        "has_speaker_embeddings": f.get("speaker_embeddings") is not None,
                        "speaker_embeddings_len": len(f.get("speaker_embeddings")) if f.get("speaker_embeddings") is not None else None,
                    }
                    summaries.append(s)
                with open(_COLLATOR_DEBUG_FILE, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"batch_size": len(features), "summaries": summaries}) + "\n")
                _COLLATOR_DEBUG_COUNT += 1
                logger.info(
                    "Wrote collator debug batch summary to %s (%d/%d)",
                    str(_COLLATOR_DEBUG_FILE),
                    _COLLATOR_DEBUG_COUNT,
                    _COLLATOR_DEBUG_MAX,
                )
        except Exception as e:
            logger.warning("Failed to write collator debug summary: %s", e)

        # Let the processor handle text padding; it expects input_ids/attention_mask in each feature
        try:
            # Only pad the text-related fields with the tokenizer.
            # Don't pass `speech_values` here — those are audio arrays and
            # will break the tokenizer's tensor conversion.
            text_batch = self.processor.tokenizer.pad(
                [
                    {"input_ids": f["input_ids"], "attention_mask": f.get("attention_mask")}
                    for f in features
                ],
                padding=True,
                return_tensors="pt",
            )
        except Exception as exc:
            logger.error(
                "processor.pad failed: first_feature=%s, feature_count=%d",
                {k: type(v).__name__ for k, v in first_feature.items()},
                len(features),
            )
            raise

        # Speaker embeddings: list[list[float]] -> tensor [B, emb_dim]
        speaker_embs = torch.tensor(
            [f["speaker_embeddings"] for f in features],
            dtype=torch.float32,
        )

        # Speech values (mel-spectrograms): all should be same size [n_mels=80, n_frames]
        # Convert to [batch, n_frames, n_mels] for SpeechT5 prenet
        speech_seqs = []
        for f in features:
            # f["speech_values"] is a list of lists (80, frames)
            # Convert to tensor and ensure shape
            mel_spec = torch.as_tensor(f["speech_values"], dtype=torch.float32)
            # Verify it's 2D
            if mel_spec.ndim != 2:
                raise ValueError(f"Expected 2D mel-spec, got {mel_spec.ndim}D with shape {mel_spec.shape}")
            speech_seqs.append(mel_spec)

        # All mel-specs should have same shape (80, n_frames) since preprocessing pads uniformly
        # Stack and transpose: (batch, 80, frames) -> (batch, frames, 80)
        if len(speech_seqs) > 0:
            stacked_speech = torch.stack(speech_seqs, dim=0)  # (batch, 80, frames)
            padded_speech = stacked_speech.transpose(1, 2)  # (batch, frames, 80)
        else:
            # Empty batch
            padded_speech = torch.zeros((len(speech_seqs), 102, 80), dtype=torch.float32)

        batch: Dict[str, Union[torch.Tensor, Any]] = {
            "input_ids": text_batch["input_ids"],
            "speaker_embeddings": speaker_embs,
            "labels": padded_speech,
        }

        if "attention_mask" in text_batch:
            batch["attention_mask"] = text_batch["attention_mask"]

        return batch
        
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_mel_spectrogram(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Compute mel-spectrogram from waveform using librosa.
    Returns mel-spectrogram with shape (n_mels=80, n_frames).
    
    Args:
        waveform: audio waveform (numpy array, shape [n_samples])
        sr: sampling rate
    
    Returns:
        mel-spectrogram (numpy array, shape [80, time_steps])
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa is required for mel-spectrogram computation. Install it with: pip install librosa")
    
    # Compute mel-spectrogram with SpeechT5 defaults
    # n_mels=80, n_fft=400, hop_length=160 (for 16kHz)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        fmin=0,
        fmax=8000,
    )
    
    # Convert to log scale
    mel_spec = np.log(np.maximum(mel_spec, 1e-9))
    
    return mel_spec


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
    # We need to preserve the input batch length when using batched=True with
    # `datasets.Dataset.map`. Return lists that align with the original examples
    # so downstream components won't receive misaligned/missing fields.
    n = len(examples[text_column])

    out_input_ids = [[] for _ in range(n)]
    out_attention_mask = [[] for _ in range(n)]
    out_speaker_embeddings = [None for _ in range(n)]
    out_speech_values = [None for _ in range(n)]

    valid_texts = []
    valid_speeches = []
    valid_speaker_embeddings = []
    valid_indices = []

    skipped_audio = 0
    skipped_speaker = 0
    skipped_text = 0
    skipped_error = 0

    for i, (text, audio_path, speaker_label) in enumerate(
        zip(examples[text_column], examples[audio_column], examples[speaker_column])
    ):
        try:
            if not text or str(text).strip() == "":
                skipped_text += 1
                continue

            if not os.path.exists(audio_path):
                skipped_audio += 1
                logger.warning(f"Audio not found: {audio_path}")
                continue

            if speaker_label not in speaker_embeddings:
                skipped_speaker += 1
                logger.warning(f"No embedding for speaker: {speaker_label}")
                continue

            waveform, file_sr = torchaudio.load(audio_path)

            if file_sr != sr:
                resampler = torchaudio.transforms.Resample(file_sr, sr)
                waveform = resampler(waveform)

            if waveform.ndim > 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            speech = waveform.squeeze(0).numpy()

            if len(speech) > max_target_length:
                speech = speech[:max_target_length]

            valid_indices.append(i)
            valid_texts.append(text)
            valid_speeches.append(speech)
            valid_speaker_embeddings.append(speaker_embeddings[speaker_label])

        except Exception as e:
            skipped_error += 1
            logger.warning(f"Error processing {audio_path}: {e}")
            continue

    total_examples = n
    logger.info(
        "Preprocess batch: total=%d, kept=%d, skipped_text=%d, skipped_audio=%d, skipped_speaker=%d, skipped_error=%d",
        total_examples,
        len(valid_texts),
        skipped_text,
        skipped_audio,
        skipped_speaker,
        skipped_error,
    )

    if valid_texts:
        # Processor returns plain lists when return_tensors is omitted
        text_inputs = processor(
            text=valid_texts,
            padding=True,
            truncation=True,
        )

        # Convert waveforms to mel-spectrograms
        mel_specs = []
        for speech in valid_speeches:
            mel_spec = compute_mel_spectrogram(speech, sr=sr)
            mel_specs.append(mel_spec)

        # Pad mel-spectrograms to a fixed expected length
        # With hop_length=160, max_target_length samples -> max_mel_frames
        # Add buffer to account for librosa computation variations
        expected_mel_frames = (max_target_length // 160) + 2
        padded_mel_specs = []
        for mel_spec in mel_specs:
            # mel_spec shape is (80, time_steps)
            if mel_spec.shape[1] < expected_mel_frames:
                pad_width = ((0, 0), (0, expected_mel_frames - mel_spec.shape[1]))
                mel_spec = np.pad(mel_spec, pad_width, mode='constant', constant_values=-11.0)
            elif mel_spec.shape[1] > expected_mel_frames:
                # Truncate if longer
                mel_spec = mel_spec[:, :expected_mel_frames]
            padded_mel_specs.append(mel_spec)

        # Map processed valid entries back to their original positions
        for j, orig_idx in enumerate(valid_indices):
            # Ensure returned fields are plain Python lists (JSON/Arrow-friendly)
            iid = text_inputs["input_ids"][j]
            out_input_ids[orig_idx] = iid.tolist() if hasattr(iid, "tolist") else list(iid)
            if "attention_mask" in text_inputs:
                am = text_inputs["attention_mask"][j]
                out_attention_mask[orig_idx] = am.tolist() if hasattr(am, "tolist") else list(am)

            # Use mel-spectrogram instead of waveform
            mel_spec = padded_mel_specs[j]
            out_speech_values[orig_idx] = mel_spec.tolist()

            out_speaker_embeddings[orig_idx] = (
                valid_speaker_embeddings[j].tolist()
                if hasattr(valid_speaker_embeddings[j], "tolist")
                else list(valid_speaker_embeddings[j])
            )

    # Return aligned lists; invalid positions will have empty/None placeholders
    result = {
        "input_ids": out_input_ids,
        "speaker_embeddings": out_speaker_embeddings,
        "speech_values": out_speech_values,
    }

    if any(out_attention_mask):
        result["attention_mask"] = out_attention_mask

    return result

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
    model = SpeechT5ForTextToSpeech.from_pretrained(args.model_name, dtype=torch.float32)
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
    # Include validation speakers as well so embeddings exist during eval
    logger.info("Extracting audio and speaker information...")
    train_audio_paths = [path for path in train_dataset[args.audio_column]]
    train_speaker_labels = [label for label in train_dataset[args.speaker_column]]
    if val_dataset is not None:
        try:
            val_audio_paths = [path for path in val_dataset[args.audio_column]]
            val_speaker_labels = [label for label in val_dataset[args.speaker_column]]
            # Extend the lists so speaker embeddings are created for all speakers
            train_audio_paths = train_audio_paths + val_audio_paths
            train_speaker_labels = train_speaker_labels + val_speaker_labels
        except Exception:
            # If validation dataset doesn't have expected columns, ignore
            logger.debug("Validation dataset missing expected audio/speaker columns; skipping extension of embedding list")

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
        remove_columns=train_dataset.column_names,
    )

    def log_dataset_health(dataset, split_name: str):
        missing_input_ids = 0
        missing_speech_values = 0
        missing_speaker_embeddings = 0
        missing_attention_mask = 0
        for row in dataset:
            if not row.get("input_ids"):
                missing_input_ids += 1
            if not row.get("speech_values"):
                missing_speech_values += 1
            if row.get("speaker_embeddings") is None:
                missing_speaker_embeddings += 1
            if "attention_mask" in row and not row.get("attention_mask"):
                missing_attention_mask += 1

        logger.info(
            "%s dataset health: rows=%d, missing_input_ids=%d, missing_speech_values=%d, missing_speaker_embeddings=%d, missing_attention_mask=%d",
            split_name,
            len(dataset),
            missing_input_ids,
            missing_speech_values,
            missing_speaker_embeddings,
            missing_attention_mask,
        )

        if missing_input_ids or missing_speech_values or missing_speaker_embeddings:
            logger.warning(
                "%s has %d problematic rows; inspect preprocessing or source CSV values.",
                split_name,
                missing_input_ids + missing_speech_values + missing_speaker_embeddings,
            )

    log_dataset_health(train_dataset, "Train")

    def filter_valid_rows(example):
        return bool(example.get("input_ids")) and bool(example.get("speech_values")) and example.get("speaker_embeddings") is not None

    train_valid_count = train_dataset.filter(filter_valid_rows).num_rows
    if train_valid_count != len(train_dataset):
        logger.warning(
            "Filtered train dataset: kept %d/%d valid rows after preprocessing.",
            train_valid_count,
            len(train_dataset),
        )
    train_dataset = train_dataset.filter(filter_valid_rows)

    if val_dataset:
        val_dataset = val_dataset.map(
            preprocess_fn,
            batched=True,
            batch_size=8,
            desc="Processing validation data",
            remove_columns=val_dataset.column_names,
        )
        log_dataset_health(val_dataset, "Validation")
        val_valid_count = val_dataset.filter(filter_valid_rows).num_rows
        if val_valid_count != len(val_dataset):
            logger.warning(
                "Filtered validation dataset: kept %d/%d valid rows after preprocessing.",
                val_valid_count,
                len(val_dataset),
            )
        val_dataset = val_dataset.filter(filter_valid_rows)

    # TrainingArguments in some transformers versions may not accept
    # `evaluation_strategy` as a constructor kwarg. Try to set it and
    # fall back if the installed transformers is older.
    try:
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
            fp16=False,
            bf16=False,
            optim="adamw_8bit" if args.device == "cuda" else "adamw_torch",
            report_to="tensorboard",
        )
    except TypeError:
        logger.warning("TrainingArguments() does not accept 'evaluation_strategy' in this transformers version; using fallback parameters.")
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
            save_strategy="steps",
            seed=args.seed,
            fp16=False,
            bf16=False,
            optim="adamw_8bit" if args.device == "cuda" else "adamw_torch",
            report_to="tensorboard",
        )

    # For debugging, allow forcing dataloader workers to 0 so the collator runs
    # in the main process (helps with breakpoints/input/pdb). Set the
    # environment variable `COLLATOR_DEBUG=1` to enable.
    try:
        import os

        if os.environ.get("COLLATOR_DEBUG"):
            training_args.dataloader_num_workers = 0
            logger.info("COLLATOR_DEBUG enabled — set training_args.dataloader_num_workers=0 for deterministic collator debugging")
    except Exception:
        pass

    logger.info("Initializing Trainer...")
    # Write a compact summary of dataset columns and first rows so we can
    # verify which keys survive into the Trainer/Dataloader.
    try:
        cols = list(train_dataset.column_names)
        sample_summaries = []
        for i in range(min(3, len(train_dataset))):
            row = train_dataset[i]
            s = {"idx": i, "keys": sorted(list(row.keys()))}
            for k in s["keys"]:
                s[f"shape_{k}"] = _safe_shape(row.get(k))
            sample_summaries.append(s)
        with open(_COLLATOR_DEBUG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"dataset_columns": cols, "samples": sample_summaries}) + "\n")
        logger.info("Wrote dataset column/row summary to %s", str(_COLLATOR_DEBUG_FILE))
    except Exception as e:
        logger.warning("Failed to write train dataset sample summary: %s", e)
    data_collator = SpeechT5DataCollator(processor=processor)

    # Wrap HF `datasets.Dataset` in a lightweight torch `Dataset` adapter so
    # the Trainer/DataLoader will receive dicts unchanged and won't drop
    # columns via internal dataset formatting/selection logic.
    from torch.utils.data import Dataset as _TorchDataset

    class _HFDatasetAdapter(_TorchDataset):
        def __init__(self, hf_ds):
            self._ds = hf_ds
            self._getitem_count = 0
            logger.info("_HFDatasetAdapter init called for dataset of length %d", len(hf_ds))
            # Write to JSONL immediately to confirm instantiation
            try:
                with open(_COLLATOR_DEBUG_FILE, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"adapter_initialized": True, "ds_len": len(hf_ds)}) + "\n")
            except Exception:
                pass

        def __len__(self):
            return len(self._ds)

        def __getitem__(self, idx):
            self._getitem_count += 1
            row = self._ds[int(idx)]
            # Log EVERY call for the first 20 items
            if self._getitem_count <= 20:
                try:
                    summary = {
                        "adapter_getitem": {
                            "call_count": self._getitem_count,
                            "idx": int(idx),
                            "has_speech_values": "speech_values" in row,
                            "keys": sorted(list(row.keys())) if hasattr(row, 'keys') else str(type(row)),
                        }
                    }
                    with open(_COLLATOR_DEBUG_FILE, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(summary) + "\n")
                except Exception as e:
                    try:
                        with open(_COLLATOR_DEBUG_FILE, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"adapter_getitem_error": str(e)}) + "\n")
                    except:
                        pass
            return row
            
            
    train_dataset_wrapped = _HFDatasetAdapter(train_dataset)
    val_dataset_wrapped = _HFDatasetAdapter(val_dataset) if val_dataset is not None else None
    # Build Trainer kwargs dynamically to remain compatible with older
    # transformers versions that may not accept `remove_unused_columns`.
    import inspect

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "processing_class": processor,  # instead of tokenizer=...
    }

    try:
        # In transformers 4.57.0 the Trainer constructor does not accept
        # remove_unused_columns, so we need to disable it on TrainingArguments.
        training_args.remove_unused_columns = False
    except Exception:
        logger.warning("Could not set training_args.remove_unused_columns=False; this may cause Trainer to drop unused batch columns.")

    try:
        sig = inspect.signature(Trainer.__init__)
        if "remove_unused_columns" in sig.parameters:
            trainer_kwargs["remove_unused_columns"] = False
    except Exception:
        # Fallback: don't add the kwarg if inspection fails
        pass

    logger.info("Trainer will use remove_unused_columns=%s", getattr(training_args, "remove_unused_columns", None))

    # Pass the wrapped datasets to Trainer to avoid HF Dataset column pruning.
    trainer = Trainer(**{**trainer_kwargs, "train_dataset": train_dataset_wrapped, "eval_dataset": val_dataset_wrapped})

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
