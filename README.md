> 🌍 **Looking for the English version?** Check out the [README_EN.md](./README_EN.md) for the English guide.
# 🎙️ Role-Based Transcription using Colab

Пошаговое руководство и готовый скрипт для транскрибации аудио/видео с автоматическим разделением по ролям (спикерам) с помощью **WhisperX** и **PyAnnote 3.1** в среде Google Colab.

> 🛠️ **Python 3.12 Fix:** Скрипт полностью адаптирован под новые обновления сред Colab. В коде реализована "абсолютная защита" от популярных ошибок совместимости библиотек (таких как `AttributeError: np.NaN`, конфликты версий `transformers` и устаревшие бэкенды `torchaudio`).

---

## 📂 Что внутри репозитория?

* **`README.md`** — Подробная текстовая инструкция со всеми шагами настройки на русском языке.
* **`README_EN.md`** — Подробная текстовая инструкция со всеми шагами настройки на английском языке.
* **`role-based-transcription-colab.ipynb`** — Готовый интерактивный блокнот для Google Colab. Вы можете скачать его и загрузить прямо в свой Colab, чтобы не создавать ячейки вручную.

---

## 🚀 Быстрый старт (Инструкция)

### Шаг 1: Подготовка среды Colab
1. Перейдите в [Google Colab](https://colab.research.google.com/).
2. Вы можете создать новый блокнот и перенести код из этой инструкции, либо открыть файл `role-based-transcription-colab.ipynb` из этого репозитория.
3. В верхнем правом углу нажмите на стрелочку **▼** рядом с кнопкой *«Подключиться»*.
4. Нажмите **«Сменить среду выполнения»** -> выберите **Графический процессор T4** -> нажмите *«Сохранить»*.
5. Нажмите кнопку **«Подключиться»**.

### Шаг 2: Настройка Hugging Face (Токен и модели)
1. Зайдите в настройки токенов на [Hugging Face Tokens](https://huggingface.co/settings/tokens).
2. Зарегистрируйтесь (если еще нет) и обязательно подтвердите почту.
3. Нажмите кнопку **`+ Create new token`**. Выберите тип **Read**, введите любое имя и нажмите *«Create token»*. Копируйте полученный токен.
4. Вернитесь в свой блокнот Colab: на панели слева нажмите на иконку ключа (**Секреты / Secrets**).
5. Нажмите *«Добавить секрет»*: Название — `HF_TOKEN`, Значение — вставьте скопированный токен. Переключатель **«Доступ из блокнотов»** переведите в активное положение.
6. **Важно:** Перейдите по двум ссылкам ниже и примите условия использования моделей (нажмите кнопку **Accept** на страницах Hugging Face):
   * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Шаг 3: Загрузка медиафайла
1. На панели Colab слева нажмите на иконку папки (**Файлы**).
2. Перетащите в это окно ваш видео- или аудиофайл и дождитесь, пока круговой индикатор загрузки полностью заполнится.

---

## 💻 Порядок выполнения кода

Если вы используете чистый блокнот, создайте 4 кодовых ячейки с помощью комбинации клавиш `Ctrl + M + B`. 

> ⚠️ *Во время промежуточных установок (Ячейки 1, 2, 3) в консоли могут появляться предупреждения или красные сообщения об ошибках зависимостей. **Не обращайте на них внимания**, это особенности сборки окружения.*

### Ячейка 1: Ультимативная установка пакетов
```python
# ЯЧЕЙКА 1: УЛЬТИМАТИВНАЯ УСТАНОВКА (РЕШЕНИЕ ПРОБЛЕМЫ PYTHON 3.12)

print("⏳ 1. Удаляем несовместимый PyTorch 2.5...")
!pip uninstall -y torch torchvision torchaudio > /dev/null

print("⏳ 2. Ставим идеальную версию PyTorch 2.3.1 (совместимую с Python 3.12 и WhisperX)...")
!pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

print("⏳ 3. Ставим правильные версии Numpy и HuggingFace...")
!pip install -q "numpy<2.0" "huggingface_hub<0.24.0"

print("⏳ 4. Ставим Pyannote и системные утилиты...")
!pip install -q pyannote.audio==3.1.1
!apt-get install -y ffmpeg > /dev/null

print("⏳ 5. Ставим WhisperX (в обход старых требований)...")
!pip install -q --no-deps git+[https://github.com/m-bain/whisperx.git](https://github.com/m-bain/whisperx.git)
!pip install -q faster-whisper ctranslate2

print("\n" + "="*50)
print("✅ ИДЕАЛЬНАЯ СРЕДА СОЗДАНА!")
print("👉 ТЕПЕРЬ ОБЯЗАТЕЛЬНО: Среда выполнения -> Перезапустить сеанс (Restart session).")

```

### 🔄 Перезапуск сессии №1

В верхнем меню Colab выберите: **Среда выполнения** -> **Перезапустить сеанс** *(Restart session)*. Это применит установленные пакеты.

### Ячейка 2: Синхронизация Трансформеров

```python
# ЯЧЕЙКА 2: СИНХРОНИЗАЦИЯ ТРАНСФОРМЕРОВ
print("⏳ Опускаем Transformers до стабильной версии...")
!pip install -q transformers==4.38.2 huggingface_hub==0.22.2 tokenizers==0.15.2
print("✅ Готово! Теперь сделайте: Среда выполнения -> Перезапустить сеанс.")

```

### 🔄 Перезапуск сессии №2

Снова выберите: **Среда выполнения** -> **Перезапустить сеанс**.

### Ячейка 3: Обновление WhisperX до чистой версии

```python
# ЯЧЕЙКА 3: ОБНОВЛЕНИЕ ДО СВЕЖЕЙ ВЕРСИИ
print("⏳ Возвращаем самую свежую версию WhisperX...")
!pip install -q --force-reinstall --no-deps git+[https://github.com/m-bain/whisperx.git](https://github.com/m-bain/whisperx.git)

print("⏳ Обновляем Трансформеры и FFmpeg...")
!pip install -q ffmpeg-python transformers huggingface_hub

print("✅ ГОТОВО! Сделайте: Среда выполнения -> Перезапустить сеанс.")

```

### 🔄 Перезапуск сессии №3

Последний раз выберите: **Среда выполнения** -> **Перезапустить сеанс**.

### Ячейка 4: Финальная транскрибация по ролям

> 📝 **Настройка перед запуском:** В блоке `═══ НАСТРОЙКИ ═══` замените значение переменной `FILE_NAME` на точное имя файла, который вы загрузили в панель файлов.

```python
# ЯЧЕЙКА 4: ФИНАЛЬНАЯ ТРАНСКРИБАЦИЯ ПО РОЛЯМ

import os
import gc
import inspect

import numpy as np
# 🔥 1. ФИКС ОШИБКИ NUMPY (Делаем ЭТО СТРОГО ДО импорта pyannote и других библиотек!)
if not hasattr(np, 'NaN'): np.NaN = np.nan
if not hasattr(np, 'NAN'): np.NAN = np.nan
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import pandas as pd
import torch
import torchaudio

# 🪄 2. БРОНЕЖИЛЕТ ОТ СТАРЫХ ОШИБОК TORCHAUDIO
if not hasattr(torchaudio, 'set_audio_backend'): torchaudio.set_audio_backend = lambda backend: None
if not hasattr(torchaudio, 'get_audio_backend'): torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, 'list_audio_backends'): torchaudio.list_audio_backends = lambda: ["soundfile", "sox"]

# ТЕПЕРЬ импортируем pyannote (ошибка с np.NaN больше не появится)
from pyannote.audio import Pipeline

# 🪄 3. ПЕРЕХВАТЧИК ОПЕЧАТОК PYANNOTE
import pyannote.audio.core.inference
if not getattr(pyannote.audio.core.inference.Inference.__init__, '_is_patched', False):
    _orig_inference_init = pyannote.audio.core.inference.Inference.__init__
    def _safe_inference_init(self, model, *args, **kwargs):
        if 'token' in kwargs: kwargs['use_auth_token'] = kwargs.pop('token')
        _orig_inference_init(self, model, *args, **kwargs)
    _safe_inference_init._is_patched = True
    pyannote.audio.core.inference.Inference.__init__ = _safe_inference_init

# 🪄 4. УМНЫЙ ПЕРЕХВАТЧИК НАСТРОЕК FASTER-WHISPER
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

# 🪄 5. ПРЯМОЙ ИМПОРТ WHISPERX
import whisperx
from whisperx.diarize import assign_word_speakers
from google.colab import userdata

# ═══ НАСТРОЙКИ ═══
FILE_NAME = "2026-03-25_11-05.mkv"       # ← Убедитесь, что имя файла верное
LANGUAGE = "ru"                             # Язык
# ═════════════════

torch.cuda.empty_cache()
gc.collect()

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    raise ValueError("❌ Токен HF_TOKEN не найден в секретах (🔑 слева)!")

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"🖥️ Видеокарта: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
assert os.path.exists(FILE_NAME), f"❌ Файл '{FILE_NAME}' не загружен в Colab!"

print("\n[1/5] Загрузка модели Whisper...")
model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=LANGUAGE)

print("[2/5] Подготовка аудио...")
audio = whisperx.load_audio(FILE_NAME)

print("[3/5] Распознавание текста...")
result = model.transcribe(audio, batch_size=16)
lang = result.get("language", LANGUAGE)

del model
torch.cuda.empty_cache()

print("[4/5] Привязка слов к таймингам...")
align_model, meta = whisperx.load_align_model(language_code=lang, device=device)
result = whisperx.align(result["segments"], align_model, meta, audio, device, return_char_alignments=False)

del align_model
torch.cuda.empty_cache()

print("[5/5] Разделение по голосам (Диаризация в обход багов)...")
pyannote_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN).to(torch.device(device))
audio_data = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}
diarization = pyannote_pipeline(audio_data)

# Вручную собираем таблицу с голосами
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
    print("⚠️ Голоса не обнаружены, текст будет без спикеров.")

print("\n" + "="*60 + "\n🔥 РЕЗУЛЬТАТ:\n")
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

print(f"\n✅ Готово! Текст сохранен в файл: {output_filename}")
from google.colab import files
files.download(output_filename)

```

---

## 🎯 Результат работы

* Транскрибированный текст с метками времени и именами спикеров (например: `[0005.20 → 0012.45] SPEAKER_01: Привет, как дела?`) выведется прямо в консоль блокнота.
* Итоговый файл формата `.txt` **скачается автоматически** на ваш компьютер.
* Также готовый файл сохранится в левой панели Colab («Файлы») рядом с исходным видео.

## 🛠️ Решение возможных проблем (FAQ)

* **Ошибка `CUDA out of memory` в Ячейке 4**
  > **Решение:** Аудиофайл слишком большой или память Colab перегружена. Попробуйте уменьшить `batch_size=16` до `batch_size=8` или `4` в коде ячейки 4, либо выберите в меню *Среда выполнения -> Очистить все результаты* и перезапустите сессию.
* **Скрипт зависает на этапе `[4/5] Привязка слов`**
  > **Решение:** Проверьте, правильно ли указан язык в `LANGUAGE`. Если язык аудио английский, а в настройках стоит `"ru"`, модель выравнивания зависнет.
* **Ошибка `ValueError: HF_TOKEN not found`**
  > **Решение:** Вы забыли добавить токен в Секреты (иконка ключа 🔑 слева) или не включили тумблер «Доступ из блокнотов».
