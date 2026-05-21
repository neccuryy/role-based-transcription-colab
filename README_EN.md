# 🎙️ Role-Based Transcription using Colab

A step-by-step guide and a ready-to-run script for audio/video transcription with automatic speaker diarization (role assignment) using **WhisperX** and **PyAnnote 3.1** in Google Colab.

> 🛠️ **Python 3.12 Fix:** This script is fully adapted to the latest Google Colab environment updates. The code includes a built-in "absolute protection" against popular library compatibility errors (such as `AttributeError: np.NaN`, `transformers` version conflicts, and deprecated `torchaudio` backends).

---

## 📂 Repository Structure

* **`README.md`** — Detailed step-by-step documentation and guide.
* **`role-based-transcription-colab.ipynb`** — Ready-to-use interactive notebook for Google Colab. You can download and upload it directly into your Colab workspace to avoid creating cells manually.

---

## 🚀 Quick Start Guide

### Step 1: Prepare the Colab Environment
1. Open [Google Colab](https://colab.research.google.com/).
2. You can either create a new notebook and copy the code from this guide, or upload the `role-based-transcription-colab.ipynb` file from this repository.
3. In the upper right corner, click the arrow **▼** next to the *«Connect»* button.
4. Click **«Change runtime type»** -> select **T4 GPU** -> click *«Save»*.
5. Click the **«Connect»** button.

### Step 2: Configure Hugging Face (Token & Models)
1. Go to the token settings on [Hugging Face Tokens](https://huggingface.co/settings/tokens).
2. Sign up/Log in and make sure to verify your email.
3. Click the **`+ Create new token`** button. Select the **Read** type, enter any name, and click *«Create token»*. Copy the generated token.
4. Return to your Colab notebook: on the left panel, click the key icon (**Secrets**).
5. Click *«Add new secret»*: Name — `HF_TOKEN`, Value — paste your copied token. Toggle the **«Notebook access»** switch to the active position.
6. **Important:** Visit the two links below and accept the user terms for the models (click the **Accept** button on Hugging Face pages):
   * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Step 3: Upload your Media File
1. On the left panel in Colab, click the folder icon (**Files**).
2. Drag and drop your video or audio file into this window and wait until the circular upload indicator is completely filled.

---

## 💻 Code Execution Order

If you are using a blank notebook, create 4 code cells using the `Ctrl + M + B` shortcut. 

> ⚠️ *During intermediate installations (Cells 1, 2, 3), warnings or red dependency error messages may appear in the console. **Kindly ignore them**, as they are artifacts of building the specific environment constraints.*

### Cell 1: Ultimate Package Installation
```python
# CELL 1: ULTIMATE INSTALLATION (PYTHON 3.12 FIX)

print("⏳ 1. Removing incompatible PyTorch 2.5...")
!pip uninstall -y torch torchvision torchaudio > /dev/null

print("⏳ 2. Installing ideal PyTorch 2.3.1 (compatible with Python 3.12 and WhisperX)...")
!pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

print("⏳ 3. Installing correct Numpy and HuggingFace versions...")
!pip install -q "numpy<2.0" "huggingface_hub<0.24.0"

print("⏳ 4. Installing Pyannote and system utilities...")
!pip install -q pyannote.audio==3.1.1
!apt-get install -y ffmpeg > /dev/null

print("⏳ 5. Installing WhisperX (bypassing legacy requirements)...")
!pip install -q --no-deps git+https://github.com/m-bain/whisperx.git
!pip install -q faster-whisper ctranslate2

print("\n" + "="*50)
print("✅ PERFECT ENVIRONMENT CREATED!")
print("👉 NOW REQUIRED: Runtime -> Restart session.")
```

### 🔄 Session Restart #1
In the top menu of Colab, select: **Runtime** -> **Restart session**. This will apply the newly installed packages.

### Cell 2: Transformers Synchronization
```python
# CELL 2: TRANSFORMERS SYNCHRONIZATION
print("⏳ Downgrading Transformers to a stable version...")
!pip install -q transformers==4.38.2 huggingface_hub==0.22.2 tokenizers==0.15.2
print("✅ Done! Now do: Runtime -> Restart session.")
```

### 🔄 Session Restart #2
Select once again: **Runtime** -> **Restart session**.

### Cell 3: Updating WhisperX to a Clean Release
```python
# CELL 3: UPDATE TO FRESH VERSION
print("⏳ Restoring the freshest version of WhisperX...")
!pip install -q --force-reinstall --no-deps git+https://github.com/m-bain/whisperx.git

print("⏳ Updating Transformers and FFmpeg...")
!pip install -q ffmpeg-python transformers huggingface_hub

print("✅ DONE! Do: Runtime -> Restart session.")
```

### 🔄 Session Restart #3
For the last time, select: **Runtime** -> **Restart session**.

### Cell 4: Final Role-Based Transcription
> 📝 **Configuration before running:** In the `═══ SETTINGS ═══` block, replace the `FILE_NAME` variable value with the exact name of the file you uploaded to the files panel.

```python
# CELL 4: FINAL ROLE-BASED TRANSCRIPTION (ABSOLUTE PROTECTION)

import os
import gc
import inspect

import numpy as np
# 🔥 1. NUMPY ERROR FIX (Must run STRICTLY BEFORE importing pyannote and other libs!)
if not hasattr(np, 'NaN'): np.NaN = np.nan
if not hasattr(np, 'NAN'): np.NAN = np.nan
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import pandas as pd
import torch
import torchaudio

# 🪄 2. TORCHAUDIO LEGACY BACKEND FIX
if not hasattr(torchaudio, 'set_audio_backend'): torchaudio.set_audio_backend = lambda backend: None
if not hasattr(torchaudio, 'get_audio_backend'): torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, 'list_audio_backends'): torchaudio.list_audio_backends = lambda: ["soundfile", "sox"]

# NOW import pyannote safely (the np.NaN error will not trigger)
from pyannote.audio import Pipeline

# 🪄 3. PYANNOTE TYPO INTERCEPTOR
import pyannote.audio.core.inference
if not getattr(pyannote.audio.core.inference.Inference.__init__, '_is_patched', False):
    _orig_inference_init = pyannote.audio.core.inference.Inference.__init__
    def _safe_inference_init(self, model, *args, **kwargs):
        if 'token' in kwargs: kwargs['use_auth_token'] = kwargs.pop('token')
        _orig_inference_init(self, model, *args, **kwargs)
    _safe_inference_init._is_patched = True
    pyannote.audio.core.inference.Inference.__init__ = _safe_inference_init

# 🪄 4. FASTER-WHISPER OPTIONS INTERCEPTOR
import faster_whisper
if not getattr(faster_whisper.transcribe.TranscriptionOptions, '_is_patched', False):
    _orig_transcription_options = faster_whisper.transcribe.TranscriptionOptions
    def _patched_options(*args, **kwargs):
        sig = inspect.signature(_orig_transcription_options)
        defaults = {'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0, 'prompt_reset_on_temperature': 0.5,
                    'multilingual': False, 'max_new_tokens': None, 'clip_timestamps': None,
                    'hallucination_silence_threshold': None, 'hotwords': None}
        for k, v in defaults.items():
            if k in sig.parameters and k not in kwargs:
                kwargs[k] = v
        return _orig_transcription_options(*args, **kwargs)
    _patched_options._is_patched = True
    faster_whisper.transcribe.TranscriptionOptions = _patched_options

# 🪄 5. DIRECT WHISPERX IMPORT
import whisperx
from whisperx.diarize import assign_word_speakers
from google.colab import userdata

# ═══ SETTINGS ═══
FILE_NAME = "2026-03-25_11-05.mkv"       # ← Make sure the file name is correct
LANGUAGE = "en"                             # Language
# ═════════════════

torch.cuda.empty_cache()
gc.collect()

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    raise ValueError("❌ HF_TOKEN not found in secrets (🔑 on the left panel)!")

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"🖥️ GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
assert os.path.exists(FILE_NAME), f"❌ File '{FILE_NAME}' is not uploaded to Colab!"

print("\n[1/5] Loading Whisper Model...")
model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=LANGUAGE)

print("[2/5] Preparing Audio...")
audio = whisperx.load_audio(FILE_NAME)

print("[3/5] Transcribing Speech...")
result = model.transcribe(audio, batch_size=16)
lang = result.get("language", LANGUAGE)

del model
torch.cuda.empty_cache()

print("[4/5] Aligning words to timestamps...")
align_model, meta = whisperx.load_align_model(language_code=lang, device=device)
result = whisperx.align(result["segments"], align_model, meta, audio, device, return_char_alignments=False)

del align_model
torch.cuda.empty_cache()

print("[5/5] Separating Speakers (Diarization bypassing bugs)...")
pyannote_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN).to(torch.device(device))
audio_data = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}
diarization = pyannote_pipeline(audio_data)

# Manually building the speaker table
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({
        "segment": f"{turn.start:.3f}-{turn.end:.3f}",
        "label": speaker,
        "speaker": speaker,
        "start": turn.start,
        "end": turn.end
    })

if len(segments) > 0:
    diarize_df = pd.DataFrame(segments)
    result = assign_word_speakers(diarize_df, result)
else:
    print("⚠️ No voices detected, transcript will be speakerless.")

print("\n" + "="*60 + "\n🔥 RESULT:\n")
lines = []
for segment in result.get("segments", []):
    speaker = segment.get("speaker", "SPEAKER_??")
    text = segment.get('text', '').strip()
    start = segment.get('start', 0)
    end = segment.get('end', 0)
    
    line = f"[{start:07.2f} → {end:07.2f}] {speaker}: {text}"
    print(line)
    lines.append(line)

output_filename = os.path.splitext(FILE_NAME)[0] + "_transcript.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n✅ Success! Transcript saved to file: {output_filename}")
from google.colab import files
files.download(output_filename)
```

---

## 🎯 Output Execution

* The transcribed text with timestamps and speaker labels will be printed directly into the cell console.
* The final `.txt` output file will be **downloaded automatically** to your computer.
* The output file is also saved on the left panel ("Files") right next to your original video file.