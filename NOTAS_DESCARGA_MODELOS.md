# Notas sobre la descarga de modelos — desviaciones y problemas encontrados

Registro de lo que se hizo diferente al ejecutar la configuración, tanto en RPi4
como en PC (Ubuntu 24.04, Python 3.12). Las correcciones marcadas como
**[INCORPORADO]** ya están reflejadas en `DESCARGAR_MODELOS_RPI4.md`.

---

## Paso 1 — Vosk: SIN CAMBIOS

Descarga e instalación idéntica a lo indicado en el documento:

```bash
wget -c -P models/ https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip models/vosk-model-small-es-0.42.zip -d models/ && rm models/vosk-model-small-es-0.42.zip

wget -c -P models/ https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip
unzip models/vosk-model-es-0.42.zip -d models/ && rm models/vosk-model-es-0.42.zip
```

No hubo problemas. Velocidad media ~8 MB/s.

---

## Paso 2 — Whisper: PRIMER INTENTO INCORRECTO → CORRECCIÓN

### Primer intento (según el paso primario del documento):

```python
from faster_whisper import WhisperModel
WhisperModel('tiny', compute_type='int8', download_root='models/whisper/tiny')
```

**Problema:** `download_root` en `faster_whisper` NO coloca los ficheros
directamente en la carpeta destino. En su lugar crea la estructura de caché
de HuggingFace:

```
models/whisper/tiny/
└── models--Systran--faster-whisper-tiny/
    ├── blobs/
    │   ├── dcb76c6586fc0... ← model.bin (pero con nombre hash)
    │   └── ...
    └── refs/main
```

El benchmark (`bench_stt_exhaustivo.py`) busca `models/whisper/tiny/model.bin`
directamente, y falla con:

```
[ERROR] Unable to open file 'model.bin' in model '.../models/whisper/tiny'
```

### Solución aplicada **[INCORPORADO]**

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Systran/faster-whisper-tiny', local_dir='models/whisper/tiny')
snapshot_download(repo_id='Systran/faster-whisper-base', local_dir='models/whisper/base')
```

`snapshot_download` con `local_dir` coloca los ficheros planos directamente
en la carpeta destino, incluido `model.bin`. Esto es lo que espera el benchmark.

El documento principal ya usa este método como primario. `WhisperModel(..., download_root=...)`
queda solo como nota negativa en "Problemas comunes".

---

## Paso 3 — Piper: SIN CAMBIOS

`wget` directo a HuggingFace funcionó sin problemas (no fue necesario el
fallback con `hf_hub_download`). Las 3 voces descargadas correctamente:

- `es_ES-davefx-medium` (~63 MB)
- `es_ES-mls_10246-low` (~28 MB)  
- `es_ES-sharvard-medium` (~77 MB)

---

## Paso 4 — Coqui TTS

### En RPi4: PROBLEMA DE ESPACIO EN DISCO

```bash
pip install TTS        # falla con: ERROR: [Errno 28] No space left on device
```

Causa: el caché de pip ocupaba **1.35 GB**. Solución:

```bash
pip cache purge
pip install TTS --no-cache-dir
```

### En PC (Python 3.12): `TTS` NO COMPATIBLE → usar `coqui-tts` **[INCORPORADO]**

El paquete `TTS` (Coqui original) solo soporta Python ≤ 3.11. En Python 3.12
falla con `ERROR: No matching distribution found for TTS`.

Solución: instalar el fork `coqui-tts` que mantiene la misma API:

```bash
pip install "coqui-tts[codec]"
```

La opción `[codec]` instala `torchcodec`, requerido desde PyTorch 2.9.  
Sin ella falla con: `ImportError: the torchcodec library is required for audio IO`.

Tras instalar, se necesitan tres parches en el venv (ver Paso 0.5 del documento principal):

1. **numba 0.65.x + coverage 7.x**: `coverage.types.Tracer` fue eliminado en coverage 7.x.
   El módulo `coverage_support.py` de numba falla al importarse a través de librosa → TTS.
   Fix: desactivar condicionalmente `coverage_available` cuando la API no es compatible.

2. **transformers 5.x eliminó `isin_mps_friendly`**: usada por la capa tortoise/autoregressive
   de XTTS. Fix: reemplazar la importación con `torch.isin` (nativo desde PyTorch 2.4).

3. **torchcodec**: ya resuelto por la opción `[codec]` arriba.

---

## Paso 5 — KittenTTS: NO VERIFICADO

No se instaló ni comprobó. El benchmark lo omite automáticamente con `[SKIP]`
si no está disponible, según el propio documento.

---

## Resumen de desviaciones

### RPi4 (Ubuntu 22.04, Python 3.10)

| Paso | Estado | Desviación |
|------|--------|-----------|
| 0.5 — venv / deps | OK | — |
| 1 — Vosk | OK | Ninguna |
| 2 — Whisper | OK (2.º intento) | `download_root` no funciona → `snapshot_download` [INCORPORADO] |
| 3 — Piper | OK | Ninguna |
| 4 — Coqui TTS | OK (2.º intento) | `pip cache purge` + `--no-cache-dir` por falta de espacio |
| 5 — KittenTTS | NO VERIFICADO | Sin modelos locales, el bench lo skipea |

### PC (Ubuntu 24.04, Python 3.12)

| Componente | Estado | Desviación |
|------------|--------|-----------|
| venv | OK | Requirió `sudo apt install python3.12-venv` |
| `TTS` → `coqui-tts` | OK | `TTS` no soporta Python 3.12 → `coqui-tts[codec]` [INCORPORADO] |
| numba + coverage | OK (parcheado) | `coverage_support.py` usa API eliminada en coverage 7.x [INCORPORADO] |
| transformers + XTTS | OK (parcheado) | `isin_mps_friendly` eliminado en transformers 5.x [INCORPORADO] |
| torchcodec | OK | Incluido con `coqui-tts[codec]` [INCORPORADO] |
