

import os
import re

from transformers import SpeechT5Tokenizer
import uroman as ur
uroman = ur.Uroman()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
import scipy.signal
import torch

class TTSEngine:
    def  __init__(self, config):
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
        import torch
        
        # print("Initializing TTS Engine...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            print(f"⚠ WARNING: No CUDA GPU found. The process will run on the CPU.")

        self.config = config
        if config.speaker_indices is None:  # create the speaker_embedding for the single index once here
            self.speaker_embedding = self.create_speaker_embedding(dataset_name=config.dataset_name, 
                                        num_samples=config.num_samples, 
                                        aggregate="mean", 
                                        index=config.speaker_index, 
                                        clip_seconds=config.clip_seconds, 
                                        speaker_model_type=config.speaker_model_type, 
                                        crops=config.clips)

        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        if config.commit:
            self.model = SpeechT5ForTextToSpeech.from_pretrained(config.model_name, revision=config.commit).to(self.device)
        else:
            self.model = SpeechT5ForTextToSpeech.from_pretrained(config.model_name).to(self.device)
        self.processor = SpeechT5Processor.from_pretrained(config.model_name)
        self.tokenizer = self.processor.tokenizer
        
    @staticmethod
    def trim_silence_energy(x, sr, frame_ms=30, hop_ms=10, rel_db=-45.0, pad_ms=0):
            x = x.float();
            if x.ndim > 1: x = x.mean(dim=0)
            if x.numel() == 0: return x
            m = x.abs().max()
            if m > 0: x = x / m
            frame = max(1, int(sr*frame_ms/1000)); hop = max(1, int(sr*hop_ms/1000))
            if x.numel() < frame: return x
            f = x.unfold(0, frame, hop); e = 10.0*torch.log10(f.pow(2).mean(dim=1)+1e-12)
            thr = e.median() + rel_db
            idx = (e > thr).nonzero(as_tuple=True)[0]
            if idx.numel() == 0: return x
            a, b = idx[0].item(), idx[-1].item()
            s = max(0, a*hop - int(sr*pad_ms/1000)); t = min(x.numel(), b*hop + frame + int(sr*pad_ms/1000))
            return x[s:t]
    
    def create_speaker_embedding(
            self,
            dataset_name,
            index=0,
            num_samples=1,
            hf_token=None,
            sr=16000,
            clip_seconds=3.0,
            center=False,
            crops=3,
            aggregate="median",
            speaker_model_type="xvect"
        ):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from datasets import load_dataset, Audio
        from speechbrain.inference import EncoderClassifier

        ds = load_dataset(
            dataset_name,
            split=f"train[{index}:{index+num_samples}]",
            token=hf_token,
        )
        ds = ds.cast_column("audio", Audio(decode=True, sampling_rate=sr))
        
        # Ensure we only select as many examples as were actually loaded
        loaded_count = len(ds)
        if loaded_count == 0:
            raise ValueError(f"Loaded dataset slice contains no rows (requested index {index} num_samples {num_samples}).")
        take_n = min(num_samples, loaded_count)
        seg = ds.select(range(0, take_n))

        waves, lengths, per_counts = [], [], []
        L = int(sr*clip_seconds)

        for ex in seg:
            a = torch.tensor(ex["audio"]["array"]).float()
            if a.ndim > 1: a = a.mean(dim=0)
            a = self.trim_silence_energy(a, sr)
            segs = []
            if a.numel() <= L:
                segs = [a]
            else:
                if center:
                    s = (a.numel()-L)//2; segs = [a[s:s+L]]
                else:
                    k = max(1, crops)
                    idxs = torch.linspace(0, a.numel()-L, steps=k, dtype=torch.long)
                    segs = [self.trim_silence_energy(a[i:i+L], sr) for i in idxs]
            per_counts.append(len(segs))
            waves.extend(segs)
            lengths.extend([s.numel() for s in segs])

        batch = pad_sequence(waves, batch_first=True)
        rel = torch.tensor(lengths, dtype=torch.float32); rel = rel/rel.max().clamp_min(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EncoderClassifier.from_hparams(
            source=f"speechbrain/spkrec-{speaker_model_type}-voxceleb",
            run_opts={"device": device},
            savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "spkrec"),
        )

        with torch.no_grad():
            emb = model.encode_batch(batch.to(device), rel.to(device)).squeeze(1).cpu()

        emb = torch.nn.functional.normalize(emb, dim=-1)    # L2-norm across D

        outs, p = [], 0
        for c in per_counts:
            e = emb[p:p+c]
            v = e.median(dim=0).values if aggregate == "median" else e.mean(dim=0)
            outs.append(v); p += c

        spk = torch.stack(outs).mean(dim=0).numpy()
        return spk
        
    def run_inference(self, text_prompt):
        print(f"Running TTS inference on device: {self.device} for: {text_prompt}")

        # If the sentence isn't roman script, print the romanizedversion
        romanized = uroman.romanize_string(text_prompt.strip())
        print(f"Romanized: {romanized}")

        inputs = self.processor(text=text_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        speaker_embedding = torch.tensor(self.speaker_embedding).unsqueeze(0).to(self.device)

        spectrogram = self.model.generate_speech(input_ids, speaker_embedding, threshold=0.01)
        # print(f"Generated spectrogram shape: {spectrogram.shape}")
        with torch.no_grad():
            speech = self.vocoder(spectrogram)
        # print(f"Generated speech shape: {speech.shape}")
        return speech.cpu().numpy()

    def set_speaker_index(self, index: int):
        """Recompute the speaker embedding for a different speaker index.

        This avoids reloading the model/vocoder and only recomputes the
        speaker embedding used for inference.
        """
        print(f"Setting speaker index to: {index} and recomputing embedding")
        self.speaker_embedding = self.create_speaker_embedding(
            dataset_name=self.config.dataset_name,
            num_samples=self.config.num_samples,
            aggregate="mean",
            index=index,
            clip_seconds=self.config.clip_seconds,
            speaker_model_type=self.config.speaker_model_type,
            crops=self.config.clips,
        )

    def check_tokenization(self, text):
        print(self.tokenizer.tokenize(text))


def export_dataset_metadata(
    dataset_name: str,
    split: str = "train",
    out_csv: str = "dataset_metadata.csv",
    fields: list | None = None,
    hf_token: str | None = None,
    max_rows: int | None = None,
):
    """Export a CSV mapping dataset indices -> selected fields.

    This is a lightweight helper to inspect what examples live at which
    zero-based indices in a HuggingFace `datasets` dataset split. It does
    NOT initialize or depend on the TTSEngine and so avoids loading models.

    Parameters
    - dataset_name: HF dataset id (e.g. "sil-ai/xnr-tts-training-data")
    - split: split string, typically "train"
    - out_csv: destination CSV file path
    - fields: list of field names to include. If None, attempts to include
      a reasonable default (text, audio, id if present).
    - hf_token: optional HF token
    - max_rows: optional limit to number of rows to export (useful for large datasets)

    The CSV will contain a leading `index` column (zero-based) followed by
    the requested fields. For audio columns that are dict-like, the function
    will try to write a path/filename if present; otherwise it will write a
    short repr.
    """
    import csv
    from datasets import load_dataset

    print(f"Loading dataset {dataset_name} split={split} (this may take some time)")
    ds = load_dataset(dataset_name, split=split, token=hf_token)

    # Decide which fields to write
    available = ds.column_names
    if fields is None:
        cand = []
        for f in ("text", "sentence", "transcript", "utterance"):
            if f in available:
                cand.append(f)
                break
        # audio/id fallbacks
        if "audio" in available:
            cand.append("audio")
        if "id" in available:
            cand.append("id")
        fields = cand if cand else available[:3]

    print(f"Exporting fields: index + {fields} -> {out_csv}")

    def _field_value(example, fname):
        v = example.get(fname, None)
        if v is None:
            return ""
        # if Audio column (dict-like with path)
        if isinstance(v, dict):
            # common keys: 'path', 'audio', 'array', 'file'
            for k in ("path", "file", "filename", "audio"):
                if k in v:
                    return v[k]
            # if it contains an 'array', return length
            if "array" in v:
                return f"<audio_array len={len(v['array'])}>"
            return str(v)
        # fallback: primitive or list
        try:
            return str(v)
        except Exception:
            return repr(v)

    with open(out_csv, "w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["index"] + fields)
        for i, ex in enumerate(ds):
            if max_rows is not None and i >= max_rows:
                break
            row = [i]
            for f in fields:
                row.append(_field_value(ex, f))
            writer.writerow(row)

    print(f"Wrote {out_csv}")

# Put whatever text you want here
# xnr text sample
# text = """
# कने सुणा, ध़रतिआ पर जितणैं भ़ी बीआं बाल़िआं सब्जिआं कने हरे भ़रे बूट्टे, कने बीआं बाल़े फल़दार रुक्ख भ़ी मैं तुसां दे खाणैं ताईं दित्ते ह़न।
# कने ध़रतिआ पर सारे जानबर, पंछी, कने रगड़ोई नैं चलणैं बाल़े जितणैं भ़ी जीब जंतु ह़न, मैं तिह़नां सारेआं जो घ़ा कने हरे भ़रे बूट्टे खाणैं ताईं दित्ते ह़न। 
# कने इंह़आं ई होई गेआ।
# """
# # dgo text sample
# # text = """
# # पर उत्‍थैं ओह़ जादा दिन नेई रुके, ते कफरनह़ूम शैह़रे च जाईऐ रेह़, जित्‍थैं पैह़लैं जबूलून ते नफताली गोत्तरें दे लोग गलील झ़ीला दे बखियें रौंह़दे हे।
# # """

# speech = run_inference(text)
# sf.write("output.wav", speech.squeeze(), 16000)

# # Speed up the audio by resampling to a shorter length
# speedup = 1.0  # 20% faster
# output_path = "output_fast.wav"
# if speedup == 1.0:
#     sf.write(output_path, speech.squeeze(), 16000)
#     print(f"Audio saved to {output_path}")
# else:
#     n_samples = int(speech.shape[-1] / speedup)
#     speech_fast = scipy.signal.resample(speech, n_samples, axis=-1)
#     sf.write(output_path, speech_fast.squeeze(), 16000)
#     print(f"Faster audio saved to {output_path}")