from xnrTts import export_dataset_metadata

# export first 200 rows of the train split, writing text and id/audio where available
export_dataset_metadata(
    "sil-ai/dgo-tts-training-data",
    split="train",
    out_csv="C:\\Users\\pete_\\Dropbox\\NTprogress\\PahariAudio\\DogriWordDownloads\\TTSData\\dgo_trained_metadata.csv",
    fields=["text_b", "id", "audio"],
    max_rows=20000
)