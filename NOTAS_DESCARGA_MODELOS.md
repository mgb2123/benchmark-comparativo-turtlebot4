# Notas sobre la descarga de modelos — desviaciones respecto a DESCARGAR_MODELOS_RPI4.md

Registro de lo que se hizo diferente al ejecutar la descarga en esta RPi4,
y los problemas encontrados.

---

## Paso 1 — LLMs: OMITIDOS

Los tres modelos `.gguf` (Qwen2.5-3B, Llama-3.2-3B, Phi-3.5-mini) no se
descargaron intencionalmente. En la RPi4 no se ejecutarán los benchmarks LLM
(`bench_llm_escenarios.py`), por lo que no son necesarios.

---

## Paso 2 — Vosk: SIN CAMBIOS

Descarga e instalación idéntica a lo indicado en el documento:

```bash
wget -c -P models/ https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip models/vosk-model-small-es-0.42.zip -d models/ && rm models/vosk-model-small-es-0.42.zip

wget -c -P models/ https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip
unzip models/vosk-model-es-0.42.zip -d models/ && rm models/vosk-model-es-0.42.zip
```

No hubo problemas. Velocidad media ~8 MB/s.

---

## Paso 3 — Whisper: PRIMER INTENTO INCORRECTO → CORRECCIÓN

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

### Solución aplicada (alternativa del documento, Paso 3 — sección "Alternativa con huggingface_hub"):

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Systran/faster-whisper-tiny', local_dir='models/whisper/tiny')
snapshot_download(repo_id='Systran/faster-whisper-base', local_dir='models/whisper/base')
```

`snapshot_download` con `local_dir` coloca los ficheros planos directamente
en la carpeta destino, incluido `model.bin`. Esto es lo que espera el benchmark.

**Conclusión:** En DESCARGAR_MODELOS_RPI4.md el método primario del Paso 3
(`WhisperModel(..., download_root=...)`) NO es compatible con la estructura
que espera `bench_stt_exhaustivo.py`. Usar siempre la alternativa con
`snapshot_download(..., local_dir=...)`.

---

## Paso 4 — Piper: SIN CAMBIOS

`wget` directo a HuggingFace funcionó sin problemas (no fue necesario el
fallback con `hf_hub_download`). Las 3 voces descargadas correctamente:

- `es_ES-davefx-medium` (~63 MB)
- `es_ES-mls_10246-low` (~28 MB)  
- `es_ES-sharvard-medium` (~77 MB)

---

## Paso 5 — Coqui TTS: PROBLEMA DE ESPACIO EN DISCO

### Primer intento:

```bash
pip install TTS
```

**Fallo:** `ERROR: [Errno 28] No space left on device`

Causa: el caché de pip (`~/.cache/pip`) ocupaba **1.35 GB** y las dependencias
de TTS (llvmlite ~55 MB, torch, spacy, etc.) no cabían.

### Solución:

```bash
pip cache purge          # liberó 1.35 GB
pip install TTS --no-cache-dir
```

Instalación completada correctamente.

**Recomendación para el documento:** añadir `pip cache purge` o usar
`--no-cache-dir` en entornos con disco ajustado (SD card de 32 GB con
SO + dependencias ya instaladas).

---

## Paso 6 — KittenTTS: NO VERIFICADO

No se instaló ni comprobó. El benchmark lo omite automáticamente con `[SKIP]`
si no está disponible, según el propio documento.

---

## Resumen de desviaciones

| Paso | Estado | Desviación |
|------|--------|-----------|
| 1 — LLMs | OMITIDO | Intencional (no se ejecutan en RPi4) |
| 2 — Vosk | OK | Ninguna |
| 3 — Whisper | OK (2.º intento) | `download_root` no funciona; usar `snapshot_download(..., local_dir=...)` |
| 4 — Piper | OK | Ninguna |
| 5 — Coqui TTS | OK (2.º intento) | Requirió `pip cache purge` + `--no-cache-dir` por falta de espacio |
| 6 — KittenTTS | NO VERIFICADO | Sin modelos locales, el bench lo skipea |
