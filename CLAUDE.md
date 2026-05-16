# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Benchmark suite comparing STT (Speech-to-Text) and TTS (Text-to-Speech) engines for the TurtleBot4 robot assistant on a **Raspberry Pi 4** (4 GB RAM, ARM Cortex-A72, no GPU). Evaluates faster-whisper vs Vosk (STT) and Piper vs Coqui TTS vs KittenTTS (TTS) on Spanish language tasks.

## Setup

```bash
# Activate venv (already created at .venv/)
source .venv/bin/activate
pip install -r requirements.txt
```

Download models (one-time):
```bash
# Vosk Spanish (~40 MB)
cd models/
wget https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip vosk-model-small-es-0.42.zip && rm vosk-model-small-es-0.42.zip

# Piper Spanish voice
mkdir -p models/piper && cd models/piper
wget -O es_ES-davefx-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
wget -O es_ES-davefx-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
```

## Running Benchmarks

### STT benchmark (`bench_stt_exhaustivo.py`)

```bash
# Standard run (9 faster-whisper configs + Vosk)
python3 bench_stt_exhaustivo.py

# Quick mode (2 clips per config)
python3 bench_stt_exhaustivo.py --quick

# Use a custom audio folder (default: audios/, fallback: audio_tests/)
python3 bench_stt_exhaustivo.py --audios-dir mis_audios/

# Parametric sweep (beam_size, VAD, Vosk chunk, temperature)
python3 bench_stt_exhaustivo.py --sweep

# Regenerate graphs from existing JSON without re-running
python3 bench_stt_exhaustivo.py --plot-only

# Generate synthetic audio with Piper (if no real recordings available)
python3 bench_stt_exhaustivo.py --generar-audio --piper-fallback
```

### TTS benchmarks

```bash
python3 bench_tts_exhaustivo.py                    # Piper vs Coqui TTS
python3 bench_tts_edgetts_v2.py                    # Edge TTS exhaustive (Piper/Coqui/KittenTTS)
python3 bench_tts_edgetts_v2.py --quick --no-quality
```

### Regenerate graphs

```bash
python3 generar_informe.py   # reads resultados/*.json, writes resultados/*.png
```

## Audio Test Data for STT

The STT benchmark reads WAV files from `audios/` (preferred) or `audio_tests/` (fallback).

**Using the `audios/` folder (recommended):**
- Place any number of WAV files inside `audios/`.
- Format: WAV, mono, 16 kHz, PCM 16-bit.
- To provide ground-truth transcriptions, create `audios/frases.txt` with one transcription per line, matching the alphabetical order of the WAV files. If absent, files matching the `fraseNN` naming convention are matched to the built-in `FRASES` list by index.

```bash
# Record a phrase
arecord -f S16_LE -r 16000 -c 1 audios/frase_00.wav
```

## Models Directory

```
models/
├── vosk-model-small-es-0.42/   # Vosk Spanish model (~40 MB)
├── piper/                       # Piper voice ONNX files
└── whisper/tiny/, base/         # Whisper model config/vocab (auto-downloaded)
```

The `models/` directory is gitignored. Pass a custom path with `--models-dir`.

## Output

All benchmark output goes to `resultados/`:
- `bench_stt_exhaustivo.json` / `bench_tts_exhaustivo.json` — raw results
- `graficas_stt/`, `graficas_tts_v2/` — PNG charts
- `informe_stt_parametros.md`, `informe_tts_edgetts_v2.md` — ranked Markdown reports
- Generated WAV files per config (TTS only, gitignored)

## Architecture

```
bench_stt_exhaustivo.py   ──► resultados/bench_stt_exhaustivo.json
bench_tts_exhaustivo.py   ──► resultados/bench_tts_exhaustivo.json
bench_tts_edgetts_v2.py   ──► resultados/bench_tts_edgetts_v2.json
                                              │
                          generar_informe.py ◄┘  (reads JSONs, writes PNGs)
```

**STT script structure:** `main()` resolves audio files and ground truths, then calls `benchmark_whisper_config()` and `benchmark_vosk()` for each engine. All benchmark functions accept `(archivos_wav, ground_truths, ...)`. A `MonitorRAM` thread samples RSS every 0.1 s for peak detection. Composite ranking: `α*(1-WER) + β*(1-RTF) + γ*(1-RAM)` (defaults: 0.4/0.4/0.2).

**Key methodological choices:**
- First repetition per clip is warmup and discarded; metrics from reps 2–N.
- WER uses diacritic-stripped comparison (`normalizar()`) and Wilson 95% CIs.
- RTF = inference_time / audio_duration (weighted sum across all clips, not mean of per-clip RTFs).
- RPi4 thermal throttling detected via `/sys/devices/platform/soc/soc:firmware/get_throttled`.
