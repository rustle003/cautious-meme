import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from transformers.models.whisper import modeling_whisper
from datasets import load_dataset
from pathlib import Path
# from coremltools.models.compute_device import *

# from ane_transformers.huggingface import distilbert



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "."

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=torch_dtype,
    device=device,
    # language="en"
)

dataset = load_dataset("audiofolder", data_dir = Path.cwd().as_posix() + "/input_audio", split="test")
# sample = dataset[0]["audio"]

for sample in dataset:
    result = pipe(sample["audio"])
    print(result["text"])

# result = pipe(sample)
# print(result["text"])