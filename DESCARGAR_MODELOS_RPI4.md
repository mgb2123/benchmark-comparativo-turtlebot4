# Guía de entorno y modelos — RPi4 / PC

Preparación completa del entorno para ejecutar los benchmarks STT y TTS.  
Validado en **Raspberry Pi 4** (Ubuntu 22.04 ARM) y **PC** (Ubuntu 24.04, Python 3.12, x86-64).

---

## Estructura de carpetas objetivo

```
benchmark-comparativo-turtlebot4/
├── .venv/                              ← entorno virtual Python
└── models/
    ├── vosk-model-small-es-0.42/       ← STT (Vosk pequeño)
    ├── vosk-model-es-0.42/             ← STT (Vosk grande, opcional)
    ├── whisper/
    │   ├── tiny/                       ← STT (faster-whisper tiny)
    │   └── base/                       ← STT (faster-whisper base)
    └── piper/
        ├── es_ES-davefx-medium.onnx    ← TTS
        ├── es_ES-davefx-medium.onnx.json
        ├── es_ES-mls_10246-low.onnx    ← TTS (opcional)
        ├── es_ES-mls_10246-low.onnx.json
        ├── es_ES-sharvard-medium.onnx  ← TTS (opcional)
        └── es_ES-sharvard-medium.onnx.json
```

---

## Paso 0 — Requisitos previos del sistema

```bash
sudo apt update && sudo apt install -y wget curl unzip git python3-pip python3.12-venv libespeak-ng1
```

> En RPi4 con Ubuntu 22.04, sustituye `python3.12-venv` por `python3.10-venv` según la versión instalada.
> `libespeak-ng1` es necesario para piper-tts.

Verifica espacio libre (necesitas ~3 GB mínimo, ~8 GB con Coqui XTTS-v2):

```bash
df -h ~
```

Sitúate en el proyecto y crea la estructura de modelos:

```bash
cd ~/benchmark-comparativo-turtlebot4   # ajusta la ruta si es distinta
mkdir -p models/piper models/whisper/tiny models/whisper/base
```

---

## Paso 0.5 — Entorno virtual Python

Los benchmarks requieren varios paquetes Python que pueden entrar en conflicto
con el sistema. Se recomienda un **venv** con `--system-site-packages` para
heredar los paquetes ya instalados (numpy, matplotlib…) y añadir los que faltan.

### Crear el venv

```bash
python3 -m venv .venv --system-site-packages
```

### Instalar dependencias

```bash
.venv/bin/pip install \
    seaborn pandas \
    piper-tts \
    openai-whisper \
    "coqui-tts[codec]"
```

> **IMPORTANTE — Python 3.12:** el paquete `TTS` (Coqui original) solo soporta
> hasta Python 3.11. En Python 3.12 usa el fork **`coqui-tts`**, que mantiene
> la misma API (`from TTS.api import TTS`). La opción `[codec]` instala
> `torchcodec`, requerido desde PyTorch 2.9.

Si `faster-whisper` o `vosk` no están en el sistema, instálalos también:

```bash
.venv/bin/pip install faster-whisper vosk
```

### Parches de compatibilidad (Python 3.12 + librerías recientes)

Tras instalar, aplica los tres parches necesarios en el venv:

**1. numba 0.65.x no reconoce la API de coverage 7.x**

El módulo `coverage_support.py` de numba usa `coverage.types.Tracer` y
`coverage.types.TShouldTraceFn` que coverage 7.x eliminó. La integración
coverage-numba no se usa en los benchmarks, así que se desactiva condicionalmente:

```bash
NUMBA_COVERAGE_SUPPORT=".venv/lib/$(ls .venv/lib)/site-packages/numba/misc/coverage_support.py"

# Reemplaza el bloque try/except que activa coverage_available
python3 - <<'EOF'
import re, sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = """\
try:
    import coverage
except ImportError:
    coverage_available = False
else:
    coverage_available = True"""

new = """\
try:
    import coverage
    # Disable if coverage API is incompatible (coverage 7.x renamed/removed Tracer)
    coverage_available = (
        hasattr(coverage.types, 'Tracer') or hasattr(coverage.types, 'TracerCore')
    ) and hasattr(coverage.types, 'TShouldTraceFn')
except ImportError:
    coverage_available = False"""

if old not in src:
    print("SKIP: bloque no encontrado (ya parcheado o versión diferente)")
else:
    with open(path, 'w') as f:
        f.write(src.replace(old, new, 1))
    print(f"OK: {path}")
EOF
"$NUMBA_COVERAGE_SUPPORT"
```

**2. transformers 5.x eliminó `isin_mps_friendly` (usada por coqui-tts/XTTS)**

```bash
AUTOREGRESSIVE=".venv/lib/$(ls .venv/lib)/site-packages/TTS/tts/layers/tortoise/autoregressive.py"

python3 - <<'EOF'
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = "# TODO: use torch.isin from Pytorch 2.4\nfrom transformers.pytorch_utils import isin_mps_friendly as isin"
new = "isin = __import__('torch').isin  # isin_mps_friendly removed in transformers 5.x"

if old not in src:
    # Fallback: buscar solo la línea de importación
    old2 = "from transformers.pytorch_utils import isin_mps_friendly as isin"
    if old2 in src:
        src = src.replace(old2, new, 1)
        with open(path, 'w') as f:
            f.write(src)
        print(f"OK (fallback): {path}")
    else:
        print("SKIP: línea no encontrada (ya parcheado o versión diferente)")
else:
    with open(path, 'w') as f:
        f.write(src.replace(old, new, 1))
    print(f"OK: {path}")
EOF
"$AUTOREGRESSIVE"
```

**3. (Ya resuelto por `[codec]`) torchcodec requerido desde PyTorch 2.9**

Si en algún momento el import de TTS falla con:
```
ImportError: the torchcodec library is required for audio IO
```
instala: `pip install "coqui-tts[codec]"` (ya incluido en el paso anterior).

### Verificar que todo importa correctamente

```bash
.venv/bin/python3 -c "
import psutil, numpy, matplotlib, seaborn, pandas
import faster_whisper, vosk
from piper import PiperVoice
from TTS.api import TTS
import whisper
print('Todas las importaciones OK')
"
```

---

## Paso 1 — Modelos STT Vosk

### 1a. Vosk pequeño (vosk-model-small-es-0.42, ~39 MB)

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

### 1b. Vosk grande (vosk-model-es-0.42, ~1.4 GB) — opcional

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

## Paso 2 — Modelos STT Whisper (faster-whisper)

**Usa `snapshot_download` como método principal.** El argumento `download_root`
de `WhisperModel` crea una estructura de caché de Hugging Face dentro del
directorio destino (`models--Systran--faster-whisper-tiny/blobs/…`) y
el benchmark falla porque espera `models/whisper/tiny/model.bin` directamente.

### 2a. Whisper tiny (~78 MB) y base (~148 MB)

```bash
.venv/bin/python3 - <<'EOF'
from huggingface_hub import snapshot_download
print('Descargando whisper tiny...')
snapshot_download(repo_id='Systran/faster-whisper-tiny', local_dir='models/whisper/tiny')
print('tiny OK')
print('Descargando whisper base...')
snapshot_download(repo_id='Systran/faster-whisper-base', local_dir='models/whisper/base')
print('base OK')
EOF
```

> Si `huggingface_hub` no está instalado: `.venv/bin/pip install huggingface_hub`

**Por qué NO usar `WhisperModel(..., download_root=...)`:**

```bash
# ❌ NO: coloca los ficheros bajo models--Systran--faster-whisper-tiny/blobs/<hash>
python3 -c "
from faster_whisper import WhisperModel
WhisperModel('tiny', compute_type='int8', download_root='models/whisper/tiny')
"
# El benchmark busca models/whisper/tiny/model.bin y falla con:
# [ERROR] Unable to open file 'model.bin' in model '.../models/whisper/tiny'
```

---

## Paso 3 — Modelos TTS Piper (.onnx)

Cada voz son dos ficheros: `.onnx` (modelo) y `.onnx.json` (config).  
Van en `models/piper/`.

### 3a. es_ES-davefx-medium (~63 MB) — recomendada

```bash
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES"
wget -c -P models/piper/ $BASE/davefx/medium/es_ES-davefx-medium.onnx
wget -c -P models/piper/ $BASE/davefx/medium/es_ES-davefx-medium.onnx.json
```

---

### 3b. es_ES-mls_10246-low (~28 MB) — opcional

```bash
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES"
wget -c -P models/piper/ $BASE/mls_10246/low/es_ES-mls_10246-low.onnx
wget -c -P models/piper/ $BASE/mls_10246/low/es_ES-mls_10246-low.onnx.json
```

---

### 3c. es_ES-sharvard-medium (~63 MB) — opcional

```bash
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES"
wget -c -P models/piper/ $BASE/sharvard/medium/es_ES-sharvard-medium.onnx
wget -c -P models/piper/ $BASE/sharvard/medium/es_ES-sharvard-medium.onnx.json
```

**Si falla wget con Hugging Face (problema de redireccionamiento):**

```bash
.venv/bin/python3 - <<'EOF'
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
EOF
```

---

## Paso 4 — Modelos TTS Coqui (descarga automática)

Coqui TTS descarga sus modelos automáticamente al primer uso.  
La caché se guarda en `~/.local/share/tts/`.

Para **pre-descargar** sin ejecutar el benchmark:

```bash
.venv/bin/python3 -c "
from TTS.api import TTS
TTS('tts_models/es/css10/vits', progress_bar=True, gpu=False)
print('VITS OK')
"
```

> **XTTS-v2** (~1.8 GB) solo se descarga si hay más de 2.5 GB de RAM libre.  
> Si la RPi4 tiene 4 GB de RAM puede intentarlo:

```bash
.venv/bin/python3 -c "
from TTS.api import TTS
TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=True, gpu=False)
print('XTTS-v2 OK')
"
```

> **Instalación:** ya cubierta en el Paso 0.5 con `pip install "coqui-tts[codec]"`.  
> No uses `pip install TTS` — ese paquete solo soporta hasta Python 3.11.

---

## Paso 5 — KittenTTS (sin modelos locales)

KittenTTS usa una API local; no requiere ficheros en `models/`.

```bash
.venv/bin/pip install kittentts
```

Si no está disponible en pip, el benchmark lo omite automáticamente con `[SKIP]`.

---

## Verificación final

Comprueba que la estructura de modelos es correcta:

```bash
find models/ -maxdepth 3 \( -name "*.bin" -o -name "*.onnx" -o -name "final.mdl" \) | sort
```

Resultado esperado (mínimo para que los benchmarks arranquen):

```
models/piper/es_ES-davefx-medium.onnx
models/vosk-model-small-es-0.42/am/final.mdl
models/whisper/base/model.bin
models/whisper/tiny/model.bin
```

Comprueba que el venv tiene todo lo necesario:

```bash
.venv/bin/python3 -c "
import psutil, numpy, matplotlib, seaborn, pandas
import faster_whisper, vosk
from piper import PiperVoice
from TTS.api import TTS
import whisper
print('Todas las importaciones OK')
"
```

---

## Ejecutar los benchmarks

Usa siempre el Python del venv, o actívalo previamente:

```bash
# Activar (opcional, más cómodo)
source .venv/bin/activate

# STT: faster-whisper (tiny/base, int8/fp32, beam 1-5, VAD) vs Vosk
.venv/bin/python3 bench_stt_exhaustivo.py --quick   # rápido (2 clips/config)
.venv/bin/python3 bench_stt_exhaustivo.py           # completo

# TTS: Piper vs Coqui TTS
.venv/bin/python3 bench_tts_exhaustivo.py

# TTS edge v2: Piper + Coqui + KittenTTS con métricas extendidas
.venv/bin/python3 bench_tts_edgetts_v2.py --quick
.venv/bin/python3 bench_tts_edgetts_v2.py

# Generar gráficas PNG a partir de los JSON de resultados
.venv/bin/python3 generar_informe.py
```

---

## Script todo-en-uno (modelos)

Guarda este bloque como `download_models.sh` y ejecútalo con  
`bash download_models.sh` desde la raíz del proyecto:

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p models/piper models/whisper/tiny models/whisper/base

echo "=== Vosk ==="
wget -c -P models/ https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
wget -c -P models/ https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip
unzip -n models/vosk-model-small-es-0.42.zip -d models/ && rm -f models/vosk-model-small-es-0.42.zip
unzip -n models/vosk-model-es-0.42.zip       -d models/ && rm -f models/vosk-model-es-0.42.zip

echo "=== Whisper (snapshot_download — método correcto) ==="
.venv/bin/python3 - <<'EOF'
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
.venv/bin/python3 -c "from TTS.api import TTS; TTS('tts_models/es/css10/vits', progress_bar=True, gpu=False)"

echo ""
echo "Descarga completada. Ejecuta: find models/ -maxdepth 3 -name '*.bin' -o -name '*.onnx' -o -name 'final.mdl' | sort"
```

---

## Problemas comunes

| Síntoma | Solución |
|---|---|
| `ensurepip is not available` al crear el venv | `sudo apt install python3.12-venv` (ajusta la versión) |
| `externally-managed-environment` al hacer pip install | Usa el venv: `.venv/bin/pip install …` |
| `No matching distribution found for TTS` en Python 3.12 | Usa `pip install "coqui-tts[codec]"` en lugar de `TTS` |
| `AttributeError: module 'coverage.types' has no attribute 'Tracer'` | Aplica el parche de numba del Paso 0.5 |
| `ImportError: cannot import name 'isin_mps_friendly'` | Aplica el parche de autoregressive.py del Paso 0.5 |
| `ImportError: torchcodec library is required` | `pip install "coqui-tts[codec]"` |
| `wget: unable to resolve host` | Sin internet — usa un router/hotspot y reintenta |
| Descarga cortada a medias | Vuelve a lanzar el mismo `wget -c`, retoma desde donde paró |
| `unzip: cannot allocate memory` | Usa la alternativa Python del paso 1 |
| `ModuleNotFoundError: huggingface_hub` | `.venv/bin/pip install huggingface_hub` |
| Hugging Face pide login (modelos privados) | `huggingface-cli login` con token de HF |
| `[ERROR] Unable to open file 'model.bin'` en Whisper | Usa `snapshot_download` (no `WhisperModel(..., download_root=…)`) |
| Piper falla con `[SKIP]` al ejecutar benchmark | Verifica que existe el `.onnx.json` además del `.onnx` |
| faster-whisper re-descarga aunque exista el directorio | Verifica que `models/whisper/tiny/model.bin` existe directamente en esa ruta |
| Warning `Unable to import Axes3D` al importar TTS | Inofensivo — conflicto entre matplotlib del sistema y del venv |
