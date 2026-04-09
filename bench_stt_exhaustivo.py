#!/usr/bin/env python3
"""Benchmark STT exhaustivo en RPi4: faster-whisper vs Vosk.

Varía parámetros clave de faster-whisper (modelo, beam_size, compute_type)
y compara contra Vosk. Mide WER, latencia, RTF, RAM y CPU.

Uso:
    python3 bench_stt_exhaustivo.py
"""

import datetime
import gc
import json
import os
import platform
import sys
import threading
import time
import unicodedata
import wave

try:
    import psutil
except ImportError:
    print("[ERROR] psutil no instalado")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy no instalado")
    sys.exit(1)

# --- Ground truths ---
FRASES = [
    "ana avanza hacia adelante",
    "gira a la izquierda",
    "qué ves delante de ti",
    "llévame a la cocina",
    "para el robot",
    "ana cómo te llamas",
    "desacoplar del muelle",
    "buscar un objeto",
    "tomar una foto",
    "volver a la base",
]

SAMPLE_RATE = 16000

# --- Configuraciones a probar ---
WHISPER_CONFIGS = [
    {"modelo": "tiny",  "compute_type": "int8",    "beam_size": 1, "best_of": 1},
    {"modelo": "tiny",  "compute_type": "int8",    "beam_size": 3, "best_of": 1},
    {"modelo": "tiny",  "compute_type": "int8",    "beam_size": 5, "best_of": 3},
    {"modelo": "tiny",  "compute_type": "float32", "beam_size": 1, "best_of": 1},
    {"modelo": "tiny",  "compute_type": "float32", "beam_size": 3, "best_of": 1},
    {"modelo": "base",  "compute_type": "int8",    "beam_size": 1, "best_of": 1},
    {"modelo": "base",  "compute_type": "int8",    "beam_size": 3, "best_of": 1},
    {"modelo": "base",  "compute_type": "int8",    "beam_size": 5, "best_of": 3},
    {"modelo": "base",  "compute_type": "float32", "beam_size": 1, "best_of": 1},
    # {"modelo": "small", "compute_type": "int8",    "beam_size": 1, "best_of": 1},  # ~460 MB, descarga lenta
]


class MonitorRAM:
    def __init__(self):
        self.pico_rss = 0
        self._corriendo = False
        self._hilo = None

    def iniciar(self):
        self.pico_rss = psutil.Process().memory_info().rss
        self._corriendo = True
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()

    def _muestrear(self):
        proc = psutil.Process()
        while self._corriendo:
            self.pico_rss = max(self.pico_rss, proc.memory_info().rss)
            time.sleep(0.1)

    def detener(self):
        self._corriendo = False
        if self._hilo:
            self._hilo.join(timeout=1)
        return self.pico_rss


def fmt_mb(b):
    return round(b / (1024 * 1024), 1)


def normalizar(texto):
    texto = texto.lower().strip()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    texto = ''.join(c for c in texto if c.isalnum() or c.isspace())
    return ' '.join(texto.split())


def calcular_wer(ref, hyp):
    r = normalizar(ref).split()
    h = normalizar(hyp).split()
    if not r:
        return 1.0 if h else 0.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return d[len(r)][len(h)] / len(r)


def cargar_wav(ruta):
    with wave.open(ruta, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def duracion_wav(ruta):
    with wave.open(ruta, 'rb') as wf:
        return wf.getnframes() / wf.getframerate()


def benchmark_whisper_config(config, archivos_wav, ground_truths, models_dir=None):
    """Benchmark de una configuración específica de faster-whisper."""
    from faster_whisper import WhisperModel

    nombre = f"whisper-{config['modelo']}_{config['compute_type']}_beam{config['beam_size']}_best{config['best_of']}"
    print(f"\n  --- {nombre} ---")

    resultado = {
        "motor": "faster-whisper",
        "nombre": nombre,
        "config": config.copy(),
    }

    # Verificar RAM
    ram_libre = psutil.virtual_memory().available / (1024 * 1024)
    if config["modelo"] == "small" and ram_libre < 1500:
        print(f"  [SKIP] RAM insuficiente para small: {ram_libre:.0f} MB libre")
        resultado["skip"] = "RAM insuficiente"
        return resultado

    # Cargar modelo
    monitor = MonitorRAM()
    monitor.iniciar()
    proc = psutil.Process()
    proc.cpu_percent()

    # Usar modelo local si existe en models/whisper/<nombre>, sino descargar
    whisper_dir = os.path.join(models_dir, "whisper", config["modelo"]) if models_dir else None
    model_path = whisper_dir if (whisper_dir and os.path.isdir(whisper_dir)) else config["modelo"]

    t0 = time.perf_counter()
    try:
        modelo = WhisperModel(
            model_path,
            device="cpu",
            compute_type=config["compute_type"],
            local_files_only=os.path.isdir(str(model_path)),
        )
    except Exception as e:
        monitor.detener()
        print(f"  [ERROR] {e}")
        resultado["error"] = str(e)
        return resultado

    t_carga = time.perf_counter() - t0
    ram_carga = monitor.detener()

    resultado["tiempo_carga_s"] = round(t_carga, 2)
    resultado["ram_carga_mb"] = fmt_mb(ram_carga)
    print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

    # Transcribir
    resultado["clips"] = []

    for wav_path, truth in zip(archivos_wav, ground_truths):
        nombre_wav = os.path.basename(wav_path)
        audio = cargar_wav(wav_path)
        dur = duracion_wav(wav_path)

        monitor = MonitorRAM()
        monitor.iniciar()
        proc.cpu_percent()

        t0 = time.perf_counter()
        segmentos, info = modelo.transcribe(
            audio,
            language="es",
            beam_size=config["beam_size"],
            best_of=config["best_of"],
            vad_filter=False,
        )
        texto = " ".join(s.text.strip() for s in segmentos)
        t_trans = time.perf_counter() - t0

        cpu_pct = proc.cpu_percent()
        ram_pico = monitor.detener()
        wer_val = calcular_wer(truth, texto)

        resultado["clips"].append({
            "archivo": nombre_wav,
            "duracion_audio_s": round(dur, 2),
            "ground_truth": truth,
            "transcripcion": texto,
            "wer": round(wer_val, 3),
            "tiempo_s": round(t_trans, 3),
            "rtf": round(t_trans / dur, 3) if dur > 0 else 0,
            "ram_pico_mb": fmt_mb(ram_pico),
            "cpu_pct": round(cpu_pct, 1),
        })
        print(f"    {nombre_wav}: \"{texto[:45]}\" | WER={wer_val:.2f} | {t_trans:.2f}s | RTF={t_trans/dur:.2f}")

    # Promedios
    clips = resultado["clips"]
    if clips:
        resultado["promedios"] = {
            "wer": round(sum(c["wer"] for c in clips) / len(clips), 3),
            "tiempo_s": round(sum(c["tiempo_s"] for c in clips) / len(clips), 3),
            "rtf": round(sum(c["rtf"] for c in clips) / len(clips), 3),
            "ram_pico_mb": max(c["ram_pico_mb"] for c in clips),
            "cpu_pct": round(sum(c["cpu_pct"] for c in clips) / len(clips), 1),
        }

    # Ahora probar con VAD activado (solo beam_size=1 para no duplicar demasiado)
    if config["beam_size"] == 1:
        print(f"  [+VAD] Repitiendo con vad_filter=True...")
        resultado["clips_vad"] = []
        for wav_path, truth in zip(archivos_wav, ground_truths):
            audio = cargar_wav(wav_path)
            dur = duracion_wav(wav_path)

            t0 = time.perf_counter()
            segmentos, _ = modelo.transcribe(
                audio,
                language="es",
                beam_size=config["beam_size"],
                best_of=config["best_of"],
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            texto = " ".join(s.text.strip() for s in segmentos)
            t_trans = time.perf_counter() - t0
            wer_val = calcular_wer(truth, texto)

            resultado["clips_vad"].append({
                "archivo": os.path.basename(wav_path),
                "transcripcion": texto,
                "wer": round(wer_val, 3),
                "tiempo_s": round(t_trans, 3),
                "rtf": round(t_trans / dur, 3) if dur > 0 else 0,
            })

        clips_vad = resultado["clips_vad"]
        if clips_vad:
            resultado["promedios_vad"] = {
                "wer": round(sum(c["wer"] for c in clips_vad) / len(clips_vad), 3),
                "tiempo_s": round(sum(c["tiempo_s"] for c in clips_vad) / len(clips_vad), 3),
                "rtf": round(sum(c["rtf"] for c in clips_vad) / len(clips_vad), 3),
            }
            print(f"  [+VAD] WER={resultado['promedios_vad']['wer']:.3f} | RTF={resultado['promedios_vad']['rtf']:.3f}")

    del modelo
    gc.collect()
    time.sleep(1)
    return resultado


def benchmark_vosk(archivos_wav, ground_truths, models_dir):
    """Benchmark de Vosk."""
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError:
        print("  [SKIP] vosk no instalado")
        return None

    vosk_dir = None
    for nombre in os.listdir(models_dir):
        if nombre.startswith("vosk-model") and "es" in nombre:
            vosk_dir = os.path.join(models_dir, nombre)
            break

    if not vosk_dir or not os.path.isdir(vosk_dir):
        print(f"  [SKIP] Modelo Vosk no encontrado en {models_dir}")
        return None

    print(f"\n  --- vosk ({os.path.basename(vosk_dir)}) ---")
    resultado = {
        "motor": "vosk",
        "nombre": f"vosk-{os.path.basename(vosk_dir)}",
        "config": {"modelo": os.path.basename(vosk_dir)},
    }

    monitor = MonitorRAM()
    monitor.iniciar()
    proc = psutil.Process()
    proc.cpu_percent()

    t0 = time.perf_counter()
    modelo = Model(vosk_dir)
    t_carga = time.perf_counter() - t0
    ram_carga = monitor.detener()

    resultado["tiempo_carga_s"] = round(t_carga, 2)
    resultado["ram_carga_mb"] = fmt_mb(ram_carga)
    print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

    resultado["clips"] = []
    for wav_path, truth in zip(archivos_wav, ground_truths):
        nombre_wav = os.path.basename(wav_path)
        dur = duracion_wav(wav_path)

        monitor = MonitorRAM()
        monitor.iniciar()
        proc.cpu_percent()

        t0 = time.perf_counter()
        rec = KaldiRecognizer(modelo, SAMPLE_RATE)
        with wave.open(wav_path, 'rb') as wf:
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                rec.AcceptWaveform(data)
        res = json.loads(rec.FinalResult())
        texto = res.get("text", "")
        t_trans = time.perf_counter() - t0

        cpu_pct = proc.cpu_percent()
        ram_pico = monitor.detener()
        wer_val = calcular_wer(truth, texto)

        resultado["clips"].append({
            "archivo": nombre_wav,
            "duracion_audio_s": round(dur, 2),
            "ground_truth": truth,
            "transcripcion": texto,
            "wer": round(wer_val, 3),
            "tiempo_s": round(t_trans, 3),
            "rtf": round(t_trans / dur, 3) if dur > 0 else 0,
            "ram_pico_mb": fmt_mb(ram_pico),
            "cpu_pct": round(cpu_pct, 1),
        })
        print(f"    {nombre_wav}: \"{texto[:45]}\" | WER={wer_val:.2f} | {t_trans:.2f}s | RTF={t_trans/dur:.2f}")

    clips = resultado["clips"]
    if clips:
        resultado["promedios"] = {
            "wer": round(sum(c["wer"] for c in clips) / len(clips), 3),
            "tiempo_s": round(sum(c["tiempo_s"] for c in clips) / len(clips), 3),
            "rtf": round(sum(c["rtf"] for c in clips) / len(clips), 3),
            "ram_pico_mb": max(c["ram_pico_mb"] for c in clips),
            "cpu_pct": round(sum(c["cpu_pct"] for c in clips) / len(clips), 1),
        }

    # Probar con diferentes tamaños de chunk (frames por lectura)
    print(f"  [Variando chunk_size]...")
    resultado["chunks"] = {}
    for chunk_frames in [1000, 2000, 4000, 8000]:
        t0 = time.perf_counter()
        rec = KaldiRecognizer(modelo, SAMPLE_RATE)
        with wave.open(archivos_wav[0], 'rb') as wf:
            while True:
                data = wf.readframes(chunk_frames)
                if not data:
                    break
                rec.AcceptWaveform(data)
        res = json.loads(rec.FinalResult())
        t_trans = time.perf_counter() - t0
        texto = res.get("text", "")
        wer_val = calcular_wer(ground_truths[0], texto)
        resultado["chunks"][str(chunk_frames)] = {
            "tiempo_s": round(t_trans, 3),
            "wer": round(wer_val, 3),
        }
        print(f"    chunk={chunk_frames}: {t_trans:.3f}s | WER={wer_val:.2f}")

    del modelo
    gc.collect()
    return resultado


def imprimir_tabla(resultados):
    print("\n" + "=" * 100)
    print("RESUMEN COMPARATIVO STT — ANÁLISIS EXHAUSTIVO")
    print("=" * 100)

    encabezados = ["Config", "Carga(s)", "RAM(MB)", "WER", "Tiempo(s)", "RTF", "CPU%"]
    filas = []

    for r in resultados:
        if r is None or "skip" in r or "error" in r:
            continue
        p = r.get("promedios", {})
        filas.append([
            r["nombre"][:40],
            r.get("tiempo_carga_s", "N/A"),
            p.get("ram_pico_mb", r.get("ram_carga_mb", "N/A")),
            p.get("wer", "N/A"),
            p.get("tiempo_s", "N/A"),
            p.get("rtf", "N/A"),
            p.get("cpu_pct", "N/A"),
        ])

    if not filas:
        print("  Sin resultados.")
        return

    anchos = [max(len(str(f[i])) for f in [encabezados] + filas) for i in range(len(encabezados))]
    fmt = " | ".join(f"{{:<{a}}}" for a in anchos)

    print(fmt.format(*encabezados))
    print("-+-".join("-" * a for a in anchos))
    for fila in filas:
        print(fmt.format(*[str(c) for c in fila]))

    # VAD comparison
    vad_results = [r for r in resultados if r and "promedios_vad" in r]
    if vad_results:
        print("\n--- Efecto del VAD filter (beam_size=1) ---")
        print(f"{'Config':<40} | {'WER sin VAD':>11} | {'WER con VAD':>11} | {'RTF sin VAD':>11} | {'RTF con VAD':>11}")
        print("-" * 100)
        for r in vad_results:
            p = r["promedios"]
            pv = r["promedios_vad"]
            print(f"{r['nombre']:<40} | {p['wer']:>11.3f} | {pv['wer']:>11.3f} | {p['rtf']:>11.3f} | {pv['rtf']:>11.3f}")

    print()

    # Transcripciones comparativas primera frase
    print("TRANSCRIPCIONES PRIMERA FRASE:")
    print("-" * 80)
    for r in resultados:
        if r and "clips" in r and r["clips"]:
            c = r["clips"][0]
            print(f"  {r['nombre'][:35]:35s}: \"{c['transcripcion'][:50]}\" (WER={c['wer']:.2f})")
    print(f"  {'REFERENCIA':35s}: \"{FRASES[0]}\"")
    print()


def main():
    models_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models")
    )
    audio_dir = os.path.join(os.path.dirname(__file__), "audio_tests")
    output = os.path.join(os.path.dirname(__file__), "resultados", "bench_stt_exhaustivo.json")

    archivos_wav = sorted([
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir) if f.endswith(".wav")
    ]) if os.path.isdir(audio_dir) else []

    if not archivos_wav:
        print("[ERROR] No hay WAVs. Genera con: python3 bench_stt.py --generar-audio")
        sys.exit(1)

    ground_truths = FRASES[:len(archivos_wav)]

    print("=" * 70)
    print("BENCHMARK STT EXHAUSTIVO — RPi4")
    print("=" * 70)
    mem = psutil.virtual_memory()
    print(f"Hardware: {platform.machine()} | {psutil.cpu_count()} cores | "
          f"RAM: {fmt_mb(mem.total)} MB total, {fmt_mb(mem.available)} MB libre")
    print(f"Clips: {len(archivos_wav)} | Modelos dir: {models_dir}")
    print(f"Configs faster-whisper: {len(WHISPER_CONFIGS)}")
    print()

    resultados = []

    # faster-whisper con todas las configuraciones
    print("=" * 50)
    print("FASTER-WHISPER — Variación de parámetros")
    print("=" * 50)

    for config in WHISPER_CONFIGS:
        try:
            r = benchmark_whisper_config(config, archivos_wav, ground_truths, models_dir)
            resultados.append(r)
        except Exception as e:
            print(f"  [ERROR] {config}: {e}")
            resultados.append({"nombre": str(config), "error": str(e)})

    # Vosk
    print("\n" + "=" * 50)
    print("VOSK")
    print("=" * 50)
    r = benchmark_vosk(archivos_wav, ground_truths, models_dir)
    if r:
        resultados.append(r)

    imprimir_tabla(resultados)

    # Guardar JSON
    salida = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": {
            "plataforma": platform.machine(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_mb": fmt_mb(mem.total),
            "ram_disponible_mb": fmt_mb(mem.available),
        },
        "num_clips": len(archivos_wav),
        "resultados": [r for r in resultados if r is not None],
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en: {output}")


if __name__ == "__main__":
    main()
