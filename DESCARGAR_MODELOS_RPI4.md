# Guía de descarga de modelos en RPi4

Todos los modelos que necesitan los tres benchmarks, paso a paso.  
Pensada para ejecutarse en la **Raspberry Pi 4** (Raspberry Pi OS 64-bit / Ubuntu 22.04 ARM).

---

## Estructura de carpetas objetivo

```
benchmark-comparativo-turtlebot4/
└── models/
    ├── qwen2.5-3b-instruct-q4_k_m.gguf        ← LLM
    ├── Llama-3.2-3B-Instruct-Q4_K_M.gguf      ← LLM
    ├── Phi-3.5-mini-instruct-Q4_K_M.gguf      ← LLM
    ├── vosk-model-small-es-0.42/               ← STT (Vosk pequeño)
    ├── vosk-model-es-0.42/                     ← STT (Vosk grande)
    ├── whisper/
    │   ├── tiny/                               ← STT (faster-whisper tiny)
    │   └── base/                               ← STT (faster-whisper base)
    └── piper/
        ├── es_ES-davefx-medium.onnx            ← TTS
        ├── es_ES-davefx-medium.onnx.json
        ├── es_ES-mls_10246-low.onnx            ← TTS
        ├── es_ES-mls_10246-low.onnx.json
        ├── es_ES-sharvard-medium.onnx          ← TTS
        └── es_ES-sharvard-medium.onnx.json
```

---

## Paso 0 — Requisitos previos

```bash
sudo apt update && sudo apt install -y wget curl unzip git python3-pip
```

Verifica espacio libre (necesitas ~8 GB):

```bash
df -h ~
```

Sitúate en el proyecto:

```bash
cd ~/benchmark-comparativo-turtlebot4   # ajusta la ruta si es distinta
mkdir -p models/piper models/whisper/tiny models/whisper/base
```

---

## Paso 1 — Modelos LLM (GGUF, ~2 GB cada uno)

Los tres ficheros `.gguf` van directamente en `models/`.

### 1a. Qwen2.5-3B-Instruct Q4_K_M (~2.0 GB)

```bash
wget -c -P models/ \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

**Si falla (sin huggingface-cli):**

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF',
    filename='qwen2.5-3b-instruct-q4_k_m.gguf',
    local_dir='models/'
)
"
```

---

### 1b. Llama-3.2-3B-Instruct Q4_K_M (~2.0 GB)

```bash
wget -c -P models/ \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**Si falla:**

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='bartowski/Llama-3.2-3B-Instruct-GGUF',
    filename='Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    local_dir='models/'
)
"
```

---

### 1c. Phi-3.5-mini-Instruct Q4_K_M (~2.3 GB)

```bash
wget -c -P models/ \
  https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

**Si falla:**

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='bartowski/Phi-3.5-mini-instruct-GGUF',
    filename='Phi-3.5-mini-instruct-Q4_K_M.gguf',
    local_dir='models/'
)
"
```

---

### Decirle al benchmark LLM dónde están los modelos

El script usa por defecto `C:\Users\Usuario1\llm_models` (Windows).  
En RPi4 pásale la ruta con el flag o con variable de entorno:

```bash
# Opción A — flag
python3 bench_llm_escenarios.py --models-dir ~/benchmark-comparativo-turtlebot4/models

# Opción B — variable de entorno (persiste en la sesión)
export LLM_MODELS_DIR=~/benchmark-comparativo-turtlebot4/models
python3 bench_llm_escenarios.py
```

Para que sea permanente, añade la línea `export` a `~/.bashrc`.

---

## Paso 2 — Modelos STT Vosk

### 2a. Vosk pequeño (vosk-model-small-es-0.42, ~39 MB)

```bash
wget -c -P models/ \
  https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip

unzip models/vosk-model-small-es-0.42.zip -d models/
rm models/vosk-model-small-es-0.42.zip
```

**Si falla el unzip (sin memoria suficiente):**

```bash
python3 -c "
import zipfile, os
with zipfile.ZipFile('models/vosk-model-small-es-0.42.zip') as z:
    z.extractall('models/')
os.remove('models/vosk-model-small-es-0.42.zip')
"
```

---

### 2b. Vosk grande (vosk-model-es-0.42, ~1.4 GB)

```bash
wget -c -P models/ \
  https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip

unzip models/vosk-model-es-0.42.zip -d models/
rm models/vosk-model-es-0.42.zip
```

**Si la descarga se interrumpe** (`-c` retoma donde lo dejó):

```bash
wget -c -P models/ https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip
```

---

## Paso 3 — Modelos STT Whisper (faster-whisper)

faster-whisper descarga los modelos automáticamente la primera vez.  
Para **pre-descargarlos** en la carpeta correcta (sin internet en el bench):

### 3a. Whisper tiny (~78 MB)

```bash
python3 -c "
from faster_whisper import WhisperModel
WhisperModel('tiny', compute_type='int8', download_root='models/whisper/tiny')
print('tiny OK')
"
```

**Si falla (sin faster-whisper instalado):**

```bash
pip install faster-whisper
# Luego vuelve a ejecutar el comando anterior
```

**Alternativa con huggingface_hub:**

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Systran/faster-whisper-tiny',
    local_dir='models/whisper/tiny'
)
"
```

---

### 3b. Whisper base (~148 MB)

```bash
python3 -c "
from faster_whisper import WhisperModel
WhisperModel('base', compute_type='int8', download_root='models/whisper/base')
print('base OK')
"
```

**Alternativa con huggingface_hub:**

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Systran/faster-whisper-base',
    local_dir='models/whisper/base'
)
"
```

---

## Paso 4 — Modelos TTS Piper (.onnx)

Cada voz son dos ficheros: `.onnx` (modelo) y `.onnx.json` (config).  
Van en `models/piper/`.

### 4a. es_ES-davefx-medium (~63 MB)

```bash
wget -c -P models/piper/ \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx

wget -c -P models/piper/ \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json
```

---

### 4b. es_ES-mls_10246-low (~28 MB)

```bash
wget -c -P models/piper/ \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx

wget -c -P models/piper/ \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx.json
```

---

### 4c. es_ES-sharvard-medium (~63 MB)

```bash
wget -c -P models/piper/ \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx

wget -c -P models/piper/ \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json
```

**Si falla wget con Hugging Face (problema de redireccionamiento):**

```bash
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os

voces = [
    ('davefx', 'medium'),
    ('mls_10246', 'low'),
    ('sharvard', 'medium'),
]
os.makedirs('models/piper', exist_ok=True)
for voz, calidad in voces:
    nombre = f'es_ES-{voz}-{calidad}'
    for ext in ['.onnx', '.onnx.json']:
        src = hf_hub_download(
            repo_id='rhasspy/piper-voices',
            filename=f'es/es_ES/{voz}/{calidad}/{nombre}{ext}',
        )
        shutil.copy(src, f'models/piper/{nombre}{ext}')
        print(f'  OK: {nombre}{ext}')
"
```

---

## Paso 5 — Modelos TTS Coqui (descarga automática)

Coqui TTS descarga sus modelos automáticamente al primer uso.  
La caché se guarda en `~/.local/share/tts/`.

Para **pre-descargar** sin ejecutar el benchmark:

```bash
python3 -c "
from TTS.api import TTS
TTS('tts_models/es/css10/vits', progress_bar=True, gpu=False)
print('VITS OK')
"
```

> **XTTS-v2** (~1.8 GB) solo se descarga si hay más de 2.5 GB de RAM libre.  
> Si la RPi4 tiene 4 GB de RAM puede intentarlo:

```bash
python3 -c "
from TTS.api import TTS
TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=True, gpu=False)
print('XTTS-v2 OK')
"
```

**Si Coqui TTS no está instalado:**

```bash
pip install TTS
```

---

## Paso 6 — KittenTTS (sin modelos locales)

KittenTTS usa una API local; no requiere ficheros en `models/`.

```bash
pip install kittentts
```

Si no está disponible en pip, el benchmark lo omite automáticamente con `[SKIP]`.

---

## Verificación final

Comprueba que la estructura es correcta:

```bash
find models/ -maxdepth 2 -type f | sort
```

Resultado esperado (mínimo para que los tres benchmarks arranquen):

```
models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
models/Phi-3.5-mini-instruct-Q4_K_M.gguf
models/piper/es_ES-davefx-medium.onnx
models/piper/es_ES-davefx-medium.onnx.json
models/piper/es_ES-mls_10246-low.onnx
models/piper/es_ES-mls_10246-low.onnx.json
models/piper/es_ES-sharvard-medium.onnx
models/piper/es_ES-sharvard-medium.onnx.json
models/qwen2.5-3b-instruct-q4_k_m.gguf
models/vosk-model-es-0.42/am/final.mdl
models/vosk-model-small-es-0.42/am/final.mdl
models/whisper/base/model.bin
models/whisper/tiny/model.bin
```

---

## Script todo-en-uno (descarga automatizada)

Si quieres lanzarlo todo de una vez guarda este bloque como `download_models.sh`  
y ejecútalo con `bash download_models.sh` desde la raíz del proyecto:

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p models/piper models/whisper/tiny models/whisper/base

echo "=== LLM ==="
wget -c -P models/ https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
wget -c -P models/ https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
wget -c -P models/ https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf

echo "=== Vosk ==="
wget -c -P models/ https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
wget -c -P models/ https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip
unzip -n models/vosk-model-small-es-0.42.zip -d models/ && rm -f models/vosk-model-small-es-0.42.zip
unzip -n models/vosk-model-es-0.42.zip      -d models/ && rm -f models/vosk-model-es-0.42.zip

echo "=== Whisper ==="
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Systran/faster-whisper-tiny', local_dir='models/whisper/tiny')
snapshot_download(repo_id='Systran/faster-whisper-base', local_dir='models/whisper/base')
EOF

echo "=== Piper ==="
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES"
wget -c -P models/piper/ $BASE/davefx/medium/es_ES-davefx-medium.onnx
wget -c -P models/piper/ $BASE/davefx/medium/es_ES-davefx-medium.onnx.json
wget -c -P models/piper/ $BASE/mls_10246/low/es_ES-mls_10246-low.onnx
wget -c -P models/piper/ $BASE/mls_10246/low/es_ES-mls_10246-low.onnx.json
wget -c -P models/piper/ $BASE/sharvard/medium/es_ES-sharvard-medium.onnx
wget -c -P models/piper/ $BASE/sharvard/medium/es_ES-sharvard-medium.onnx.json

echo "=== Coqui (auto-descarga al primer uso) ==="
python3 -c "from TTS.api import TTS; TTS('tts_models/es/css10/vits', progress_bar=True, gpu=False)"

echo ""
echo "Descarga completada. Ejecuta: find models/ -maxdepth 2 -type f | sort"
```

---

## Problemas comunes

| Síntoma | Solución |
|---|---|
| `wget: unable to resolve host` | Sin internet — usa un router/hotspot y reintenta |
| Descarga cortada a medias | Vuelve a lanzar el mismo `wget -c`, retoma desde donde paró |
| `unzip: cannot allocate memory` | Usa la alternativa Python del paso 2 |
| `ModuleNotFoundError: huggingface_hub` | `pip install huggingface_hub` |
| Hugging Face pide login (modelos privados) | `huggingface-cli login` con token de HF |
| RPi4 se queda sin RAM al descargar LLM | Descarga uno a uno y cierra otros procesos |
| Piper falla con `SKIP` al ejecutar STT | El modelo `.onnx.json` puede faltar — descárgalo del paso 4 |
| faster-whisper descarga igualmente aunque exista el directorio | Verifica que `models/whisper/tiny/model.bin` existe |
