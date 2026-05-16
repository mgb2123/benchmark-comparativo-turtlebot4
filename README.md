# Benchmarks STT / TTS — TurtleBot4 (RPi4)

Comparativas de rendimiento para los motores de voz usados en el asistente TurtleBot4.
Hardware objetivo: Raspberry Pi 4, 4 GB RAM, ARM Cortex-A72, sin GPU.

## Scripts

| Script | Qué compara | Salida |
|---|---|---|
| `bench_stt_exhaustivo.py` | faster-whisper (tiny/base, int8/fp32, beam 1-5, VAD) vs Vosk | `resultados/bench_stt_exhaustivo.json` |
| `bench_tts_exhaustivo.py` | Piper (5 configs length/noise) vs Coqui TTS (2 modelos) | `resultados/bench_tts_exhaustivo.json` |
| `generar_informe.py` | Genera gráficas PNG a partir de los JSON | `resultados/` |

## Instalación

```bash
pip install -r requirements.txt
```

Dependencias opcionales:
```bash
# matplotlib (para gráficas en generar_informe)
pip install matplotlib

# Coqui TTS (~800 MB, solo si hay RAM suficiente)
# pip install TTS
```

## Descarga de modelos

### Vosk (español)
```bash
cd ../models/
wget https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip vosk-model-small-es-0.42.zip && rm vosk-model-small-es-0.42.zip
```

### Piper (voz española)
```bash
mkdir -p ../models/piper && cd ../models/piper
wget -O es_ES-davefx-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
wget -O es_ES-davefx-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
```

## Uso

### Grabación de audio real para STT (recomendado)

Para obtener resultados representativos, graba las frases de test con voz humana real en condiciones
similares a las del robot (micrófono del TurtleBot, ruido ambiente típico).

Las 10 frases a grabar están definidas en `bench_stt_exhaustivo.py` (variable `FRASES`).
Guarda cada frase como `audio_tests/frase_00.wav` ... `audio_tests/frase_09.wav`:
- Formato: WAV, mono, 16 kHz, PCM 16-bit
- Herramienta sugerida: `arecord -f S16_LE -r 16000 -c 1 audio_tests/frase_00.wav`

Si no es posible grabar audio real, usa el fallback sintético con Piper (documenta la limitación):
```bash
python3 bench_stt_exhaustivo.py --generar-audio --piper-fallback
```

### STT exhaustivo
```bash
# Generar audio de prueba (una sola vez)
python3 bench_stt_exhaustivo.py --generar-audio   # ver instrucciones de grabación real abajo
python3 bench_stt_exhaustivo.py --generar-audio --piper-fallback  # fallback sintético con Piper

# Ejecutar benchmark (9 configs faster-whisper + Vosk)
python3 bench_stt_exhaustivo.py
python3 bench_stt_exhaustivo.py --quick           # 2 clips por config
```

### TTS exhaustivo
```bash
# 5 configs Piper + 2 configs Coqui TTS
python3 bench_tts_exhaustivo.py
```

### Generar gráficas
```bash
# Requiere que existan los JSON en resultados/
python3 generar_informe.py
# Las PNGs se guardan en resultados/
```

## Métricas

| Métrica | Descripción |
|---|---|
| Tiempo de carga | Segundos para inicializar el motor/modelo |
| RAM pico (MB) | Memoria RSS máxima durante la operación |
| WER | (STT) Word Error Rate vs texto de referencia |
| RTF | (TTS) Real-Time Factor: síntesis / duración audio — <1.0 = más rápido que tiempo real |
| CPU% | Uso de CPU durante la operación |

## Metodología

- **Repeticiones**: STT y TTS ejecutan cada configuración 3 veces; la primera se descarta como warmup. Se reporta media ± desviación estándar.
- **WER normalización**: se eliminan diacríticos antes de comparar (ej. "que" = "qué"). Documentado para transparencia.
- **Audio STT**: idealmente voz humana real (ver sección anterior). Si se usa audio sintético (Piper), los WER absolutos pueden no ser representativos del uso real.

## Notas RPi4

- **Piper**: Diseñado para embebido, usa ONNX Runtime. RTF ~0.3 en RPi4.
- **Vosk**: Modelo small-es ~40 MB. RTF ~1.2 en RPi4.
- Los WAVs generados en `resultados/` permiten comparar calidad auditiva entre motores.
