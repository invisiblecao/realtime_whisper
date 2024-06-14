import os
import sys
import gc
import time
import pyaudio
import numpy as np
import torch
import itertools
import logging
import traceback
import rx.operators as ops
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import rich
from termcolor import colored
from threading import Thread
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment
import diart
import diart.operators as dops
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import MicrophoneAudioSource
from diart.models import SegmentationModel, EmbeddingModel
from huggingface_hub import login, hf_hub_download

# Suppress ALSA warnings
os.environ['PYTHONWARNINGS'] = 'ignore:.*:UserWarning'

# Set up device and model parameters
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

# Load Whisper model and processor
print("Loading Whisper model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device
)

segmentation_model_id = 'pyannote/segmentation'
embedding_model_id = 'pyannote/embedding'

DIA_CONFIG = {
    'max_speakers': 3,
    'duration': 3,  # Reduce duration for lower latency
    'step': 0.3,    # Reduce step for lower latency
    'latency': 'min',
    'tau_active': 0.5,
    'rho_update': 0.422,
    'delta_new': 0.4
}

print(f"diart_whisper.py\n\n\
    Device: {device}\n\
    Segmentation: {segmentation_model_id} \n\
    Embedding: {embedding_model_id} \n\
    diarization: {DIA_CONFIG}\n"
)

# Suppress unwanted warnings for a clean output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# Login to Hugging Face Hub
with open(os.devnull, 'w') as f, redirect_stdout(f):
    login(token='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', add_to_git_credential=True)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def concat(chunks, collar=0.05):
    first_annotation = chunks[0][0]
    first_waveform = chunks[0][1]
    annotation = Annotation(uri=first_annotation.uri)
    data = []
    for ann, wav in chunks:
        annotation.update(ann)
        data.append(wav.data)
    annotation = annotation.support(collar)
    window = SlidingWindow(
        first_waveform.sliding_window.duration,
        first_waveform.sliding_window.step,
        first_waveform.sliding_window.start,
    )
    data = np.concatenate(data, axis=0)
    return annotation, SlidingWindowFeature(data, window)

current_speaker = ''

def colorize_transcription(transcription):
    global current_speaker
    colors = 3 * [
        "bright_red", "bright_blue", "bright_green", "orange3", "deep_pink1",
        "yellow2", "magenta", "cyan", "bright_magenta", "dodger_blue2"
    ]
    result = []
    for speaker, text in transcription:
        if speaker == -1:
            speaker = current_speaker
        if speaker == '':
            speaker = 0
        if current_speaker != speaker:
            result.append("\n")
            current_speaker = speaker
        result.append(f"[{colors[speaker]}] {text}")
    return "".join(result)

class WhisperTranscriber:
    def __init__(self, pipe):
        self.pipe = pipe
        self._buffer = ""

    def transcribe(self, audio):
        with suppress_stdout():
            result = self.pipe(audio, return_timestamps=True)
        return result

    def identify_speakers(self, transcription, diarization, time_shift):
        # Debug print to inspect the transcription structure
        # print("Transcription result structure:", transcription)
        
        speaker_captions = []
        for segment in transcription["chunks"]:
            if ("timestamp" not in segment) or len(segment["timestamp"]) == 0:
                continue
            start, end = segment["timestamp"]
            start += time_shift
            end += time_shift
            dia = diarization.crop(Segment(start, end))
            speakers = dia.labels()
            num_speakers = len(speakers)
            if num_speakers == 0:
                caption = (-1, segment["text"])
            elif num_speakers == 1:
                spk_id = int(speakers[0].split("speaker")[1])
                caption = (spk_id, segment["text"])
            else:
                max_speaker = int(np.argmax([
                    dia.label_duration(spk) for spk in speakers
                ]))
                caption = (max_speaker, segment["text"])
            speaker_captions.append(caption)
        return speaker_captions

    def __call__(self, diarization, waveform):
        audio = waveform.data.astype("float32").reshape(-1)
        transcription = self.transcribe(audio)
        self._buffer += transcription["text"]
        time_shift = waveform.sliding_window.start
        speaker_transcriptions = self.identify_speakers(transcription, diarization, time_shift)
        return speaker_transcriptions

try:
    segmentation_path = hf_hub_download(repo_id=segmentation_model_id, filename="pytorch_model.bin", use_auth_token='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')
    segmentation = SegmentationModel.from_pretrained(segmentation_path)
    segmentation.to(device)
except Exception as e:
    print(f"Error loading segmentation model: {e}")
    segmentation = None

try:
    embedding_path = hf_hub_download(repo_id=embedding_model_id, filename="pytorch_model.bin", use_auth_token='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')
    embedding = EmbeddingModel.from_pretrained(embedding_path)
    embedding.to(device)
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding = None

config = SpeakerDiarizationConfig(
    max_speakers=DIA_CONFIG['max_speakers'],
    duration=DIA_CONFIG['duration'],
    step=DIA_CONFIG['step'],
    latency=DIA_CONFIG['latency'],
    tau_active=DIA_CONFIG['tau_active'],
    rho_update=DIA_CONFIG['rho_update'],
    delta_new=DIA_CONFIG['delta_new'],
    device=device,
    segmentation=segmentation,
    embedding=embedding
)

with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
    dia = SpeakerDiarization(config)

source = MicrophoneAudioSource(config.step)
asr = WhisperTranscriber(pipe)

transcription_duration = 3
batch_size = int(transcription_duration // config.step)

source.stream.pipe(
    dops.rearrange_audio_stream(
        config.duration, config.step, config.sample_rate
    ),
    ops.buffer_with_count(count=batch_size),
    ops.map(dia),
    ops.map(concat),
    ops.filter(lambda ann_wav: ann_wav[0].get_timeline().duration() > 0),
    ops.starmap(asr),
    ops.map(colorize_transcription),
).subscribe(
    on_next=rich.print,
    on_error=lambda _: traceback.print_exc()
)

print("Listening...")
source.read()