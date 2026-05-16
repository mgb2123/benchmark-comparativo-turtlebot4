#!/usr/bin/env python3
"""bench_mini.py — Benchmark representativo para RPi4: 1 config por motor.

Reduce la carga de los benchmarks exhaustivos de horas a <15 min:
  STT: 5 configs × 9 frases × 3 reps = 135 transcripciones
  TTS: 4 configs (1/motor) × 9 frases × 3 reps = 108 síntesis

Motores STT: faster-whisper (tiny/base, int8) + Vosk
Motores TTS: Piper · Coqui VITS ES · Coqui XTTS-v2 (skip si RAM<2500 MB) · KittenTTS

Uso:
    python3 bench_mini.py
    python3 bench_mini.py --no-quality
    python3 bench_mini.py --skip-stt
    python3 bench_mini.py --skip-tts
    python3 bench_mini.py --output-dir /ruta/resultados
"""

import argparse
import csv
import datetime
import gc
import json
import math
import os
import platform
import re
import sys
import threading
import time
import unicodedata
import wave

try:
    import psutil
except ImportError:
    print("[ERROR] psutil no instalado. pip install psutil")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy no instalado. pip install numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CORPUS MINI — 9 frases (3 por grupo de longitud)
# ---------------------------------------------------------------------------
# Las frases STT coinciden con FRASES del bench_stt_exhaustivo para reutilizar
# los audios grabados en audios/.
CORPUS_MINI = {
    "corta": [
        "para el robot",
        "Sistema listo.",
        "Giro completado.",
    ],
    "media": [
        "navega hasta la cocina y espera mis instrucciones allí",
        "No entiendo, por favor repite el comando.",
        "Iniciando secuencia de navegación autónoma.",
    ],
    "larga": [
        "avanza hasta el salón, gira noventa grados a la derecha y para cuando llegues a la pared del fondo",
        "He detectado un obstáculo inesperado en mi ruta y estoy buscando un camino alternativo.",
        "El sistema de visión ha identificado con un noventa y cinco por ciento de confianza que el objeto es una persona.",
    ],
}

# Frases STT: solo las que tienen audio grabado (coinciden con bench_stt_exhaustivo)
FRASES_STT = [
    "para el robot",
    "gira a la derecha",
    "vuelve a la base",
    "navega hasta la cocina y espera mis instrucciones allí",
    "busca el objeto rojo que está encima de la mesa",
    "toma una foto del pasillo y guárdala en memoria",
    "avanza hasta el salón, gira noventa grados a la derecha y para cuando llegues a la pared del fondo",
    "localiza a la persona que está en la habitación y cuando la encuentres emite una señal sonora para avisarme",
    "detecta si hay obstáculos en el pasillo y si los hay traza una ruta alternativa hacia el destino principal",
]

# Frases TTS (9 frases del corpus mini, orden: corta→media→larga)
FRASES_TTS = [f for g in ("corta", "media", "larga") for f in CORPUS_MINI[g]]

GRUPO_MAP_TTS = {f: g for g, lista in CORPUS_MINI.items() for f in lista}
GRUPO_MAP_STT = {
    "para el robot": "corta",
    "gira a la derecha": "corta",
    "vuelve a la base": "corta",
    "navega hasta la cocina y espera mis instrucciones allí": "media",
    "busca el objeto rojo que está encima de la mesa": "media",
    "toma una foto del pasillo y guárdala en memoria": "media",
    "avanza hasta el salón, gira noventa grados a la derecha y para cuando llegues a la pared del fondo": "larga",
    "localiza a la persona que está en la habitación y cuando la encuentres emite una señal sonora para avisarme": "larga",
    "detecta si hay obstáculos en el pasillo y si los hay traza una ruta alternativa hacia el destino principal": "larga",
}

N_REPS = 3  # rep 0 = warmup descartado; reps 1-2 = medidas válidas

# ---------------------------------------------------------------------------
# CONFIGURACIONES REPRESENTATIVAS
# ---------------------------------------------------------------------------

STT_CONFIGS = [
    {"motor": "whisper", "modelo": "tiny",  "compute_type": "int8", "beam_size": 1, "best_of": 1},
    {"motor": "whisper", "modelo": "tiny",  "compute_type": "int8", "beam_size": 5, "best_of": 3},
    {"motor": "whisper", "modelo": "base",  "compute_type": "int8", "beam_size": 1, "best_of": 1},
    {"motor": "whisper", "modelo": "base",  "compute_type": "int8", "beam_size": 3, "best_of": 1},
    {"motor": "vosk",    "modelo": "vosk-model-small-es-0.42", "chunk_frames": 4000},
]

TTS_CONFIGS = [
    {
        "motor": "piper",
        "nombre": "piper_default",
        "params": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8},
    },
    {
        "motor": "coqui_vits",
        "nombre": "coqui_vits_speed10",
        "model": "tts_models/es/css10/vits",
        "length_scale": 1.0,
    },
    {
        "motor": "coqui_vits",
        "nombre": "coqui_vits_speed08",
        "model": "tts_models/es/css10/vits",
        "length_scale": 1.25,  # speed=0.8: más lento, más expresivo
    },
    {
        "motor": "kitten",
        "nombre": "kitten_es_female_s10_t08",
        "voice": "es_female",
        "speed": 1.0,
        "temperature": 0.8,
    },
]

# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------

class MonitorRAM:
    def __init__(self):
        self.pico_rss = 0
        self._corriendo = False
        self._hilo = None

    def iniciar(self):
        self.pico_rss = psutil.Process().memory_info().rss
        self._muestras = []
        self._corriendo = True
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()

    def _muestrear(self):
        proc = psutil.Process()
        while self._corriendo:
            try:
                rss = proc.memory_info().rss
                self.pico_rss = max(self.pico_rss, rss)
                self._muestras.append(rss)
            except Exception:
                pass
            time.sleep(0.05)

    def detener(self):
        """Devuelve (pico_rss, media_rss) en bytes."""
        self._corriendo = False
        if self._hilo:
            self._hilo.join(timeout=1)
        media = int(sum(self._muestras) / len(self._muestras)) if self._muestras else self.pico_rss
        return self.pico_rss, media


class MonitorCPU:
    def __init__(self):
        self.pico_pct = 0.0
        self._muestras = []
        self._corriendo = False
        self._hilo = None

    def iniciar(self):
        self.pico_pct = 0.0
        self._muestras = []
        self._corriendo = True
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()

    def _muestrear(self):
        n_cpus = psutil.cpu_count(logical=True) or 1
        proc = psutil.Process()
        proc.cpu_percent()
        while self._corriendo:
            try:
                pct = min(proc.cpu_percent() / n_cpus, 100.0)
                if pct > 0:
                    self._muestras.append(pct)
                    if pct > self.pico_pct:
                        self.pico_pct = pct
            except Exception:
                pass
            time.sleep(0.05)

    def detener(self):
        self._corriendo = False
        if self._hilo:
            self._hilo.join(timeout=1)
        media = round(sum(self._muestras) / len(self._muestras), 1) if self._muestras else 0.0
        return self.pico_pct, media


def leer_temperatura_cpu():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        return None


def leer_throttling_rpi():
    try:
        with open("/sys/devices/platform/soc/soc:firmware/get_throttled") as f:
            return int(f.read().strip(), 16)
    except Exception:
        return None


def esperar_enfriamiento(temp_objetivo=55.0, timeout_s=60):
    temp = leer_temperatura_cpu()
    if temp is None or temp <= temp_objetivo:
        return
    print(f"  [COOL] {temp}°C > {temp_objetivo}°C — esperando...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(5)
        temp = leer_temperatura_cpu()
        if temp is None or temp <= temp_objetivo:
            print(f"  [COOL] OK ({temp}°C) tras {time.time()-t0:.0f}s")
            return
    print(f"  [WARN] Timeout enfriamiento — temp: {leer_temperatura_cpu()}°C")


def fmt_mb(b):
    return round(b / (1024 * 1024), 1)


def duracion_wav(ruta):
    try:
        with wave.open(ruta, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def cargar_wav(ruta):
    with wave.open(ruta, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _std(valores):
    n = len(valores)
    if n < 2:
        return 0.0
    m = sum(valores) / n
    return round(math.sqrt(sum((v - m) ** 2 for v in valores) / (n - 1)), 4)


def normalizar(texto):
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = "".join(c for c in texto if c.isalnum() or c.isspace())
    return " ".join(texto.split())


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
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return round(d[len(r)][len(h)] / len(r), 4)


def _normalizar_tts(texto):
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _wer_tts(ref, hyp):
    ref_words = _normalizar_tts(ref).split()
    hyp_words = _normalizar_tts(hyp).split()
    if not ref_words:
        return 0.0
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return round(dp[m] / n, 4)


# ---------------------------------------------------------------------------
# STT BENCHMARKS
# ---------------------------------------------------------------------------

def _nombre_whisper(cfg):
    return f"whisper-{cfg['modelo']}_{cfg['compute_type']}_beam{cfg['beam_size']}"


def benchmark_stt_whisper(cfg, archivos_wav, ground_truths, models_dir):
    from faster_whisper import WhisperModel

    nombre = _nombre_whisper(cfg)
    print(f"\n  --- {nombre} ---")

    ram_libre = psutil.virtual_memory().available / (1024 * 1024)
    if cfg["modelo"] == "base" and ram_libre < 600:
        print(f"  [SKIP] RAM insuficiente: {ram_libre:.0f} MB")
        return {"motor": "whisper", "nombre": nombre, "skip": "RAM insuficiente"}

    whisper_dir = os.path.join(models_dir, "whisper", cfg["modelo"])
    model_path = whisper_dir if os.path.isdir(whisper_dir) else cfg["modelo"]

    mon = MonitorRAM()
    mon.iniciar()
    t0 = time.perf_counter()
    try:
        modelo = WhisperModel(
            model_path,
            device="cpu",
            compute_type=cfg["compute_type"],
            local_files_only=os.path.isdir(str(model_path)),
        )
    except Exception as e:
        mon.detener()
        print(f"  [ERROR] {e}")
        return {"motor": "whisper", "nombre": nombre, "error": str(e)}

    t_carga = time.perf_counter() - t0
    ram_carga, _ = mon.detener()
    print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

    resultado = {
        "motor": "faster-whisper",
        "nombre": nombre,
        "config": cfg.copy(),
        "tiempo_carga_s": round(t_carga, 2),
        "ram_carga_mb": fmt_mb(ram_carga),
        "temp_inicio_c": leer_temperatura_cpu(),
        "throttling_inicio": leer_throttling_rpi(),
        "clips": [],
    }

    temp_pico = resultado["temp_inicio_c"] or 0.0

    for wav_path, truth in zip(archivos_wav, ground_truths):
        audio = cargar_wav(wav_path)
        dur = duracion_wav(wav_path)
        tiempos = []
        ram_picos = []
        ram_medios = []
        cpu_picos = []
        warmup_t = 0.0
        texto = ""

        for rep in range(N_REPS):
            mon_r = MonitorRAM()
            mon_c = MonitorCPU()
            mon_r.iniciar()
            mon_c.iniciar()
            t0 = time.perf_counter()
            try:
                segs, _ = modelo.transcribe(
                    audio,
                    language="es",
                    beam_size=cfg["beam_size"],
                    best_of=cfg["best_of"],
                    vad_filter=False,
                )
                texto = " ".join(s.text.strip() for s in segs)
            except Exception as e:
                mon_r.detener()  # descartado en error
                mon_c.detener()
                print(f"    [ERROR rep {rep}]: {e}")
                break
            t_inf = time.perf_counter() - t0
            ram_p, ram_m = mon_r.detener()
            cpu_p, _ = mon_c.detener()
            t_cpu = leer_temperatura_cpu()
            if t_cpu and t_cpu > temp_pico:
                temp_pico = t_cpu
            if rep == 0:
                warmup_t = t_inf
            else:
                tiempos.append(t_inf)
                ram_picos.append(ram_p)
                ram_medios.append(ram_m)
                cpu_picos.append(cpu_p)

        if not tiempos:
            continue

        wer_val = calcular_wer(truth, texto)
        rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
        media_t = sum(tiempos) / len(tiempos)
        media_rtf = sum(rtfs) / len(rtfs)

        resultado["clips"].append({
            "archivo": os.path.basename(wav_path),
            "grupo": GRUPO_MAP_STT.get(truth, "media"),
            "ground_truth": truth,
            "transcripcion": texto,
            "duracion_audio_s": round(dur, 2),
            "tiempo_s": round(media_t, 3),
            "std_tiempo_s": _std(tiempos),
            "warmup_s": round(warmup_t, 3),
            "rtf": round(media_rtf, 3),
            "std_rtf": _std(rtfs),
            "wer": round(wer_val, 3),
            "ram_pico_mb": fmt_mb(max(ram_picos)),
            "ram_media_mb": fmt_mb(sum(ram_medios) / len(ram_medios)) if ram_medios else fmt_mb(max(ram_picos)),
            "cpu_pct_pico": round(max(cpu_picos), 1) if cpu_picos else 0.0,
        })
        print(f"    {os.path.basename(wav_path)}: \"{texto[:40]}\" | "
              f"WER={wer_val:.2f} | RTF={media_rtf:.2f} | {media_t:.2f}±{_std(tiempos):.3f}s")

    clips = resultado["clips"]
    if clips:
        resultado["promedios"] = {
            "wer": round(sum(c["wer"] for c in clips) / len(clips), 3),
            "rtf": round(sum(c["rtf"] for c in clips) / len(clips), 3),
            "std_rtf": round(sum(c["std_rtf"] for c in clips) / len(clips), 3),
            "tiempo_s": round(sum(c["tiempo_s"] for c in clips) / len(clips), 3),
            "std_tiempo_s": round(sum(c["std_tiempo_s"] for c in clips) / len(clips), 3),
            "ram_pico_mb": max(c["ram_pico_mb"] for c in clips),
            "ram_media_mb": round(sum(c["ram_media_mb"] for c in clips) / len(clips), 1),
            "cpu_pct_pico": max(c.get("cpu_pct_pico", 0) for c in clips),
        }

    resultado["temp_pico_c"] = temp_pico if temp_pico else None
    resultado["temp_fin_c"] = leer_temperatura_cpu()
    resultado["throttling_fin"] = leer_throttling_rpi()

    del modelo
    gc.collect()
    return resultado


def benchmark_stt_vosk(cfg, archivos_wav, ground_truths, models_dir):
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError:
        print("  [SKIP] vosk no instalado. pip install vosk")
        return {"motor": "vosk", "nombre": "vosk", "skip": "no instalado"}

    nombre = "vosk"
    print(f"\n  --- {nombre} ---")

    model_dir = os.path.join(models_dir, cfg["modelo"])
    if not os.path.isdir(model_dir):
        print(f"  [SKIP] Modelo Vosk no encontrado en {model_dir}")
        return {"motor": "vosk", "nombre": nombre, "skip": f"modelo no encontrado: {model_dir}"}

    mon = MonitorRAM()
    mon.iniciar()
    t0 = time.perf_counter()
    try:
        vosk_model = Model(model_dir)
    except Exception as e:
        mon.detener()
        print(f"  [ERROR] {e}")
        return {"motor": "vosk", "nombre": nombre, "error": str(e)}

    t_carga = time.perf_counter() - t0
    ram_carga, _ = mon.detener()
    print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

    chunk_frames = cfg.get("chunk_frames", 4000)
    resultado = {
        "motor": "vosk",
        "nombre": nombre,
        "config": cfg.copy(),
        "tiempo_carga_s": round(t_carga, 2),
        "ram_carga_mb": fmt_mb(ram_carga),
        "temp_inicio_c": leer_temperatura_cpu(),
        "throttling_inicio": leer_throttling_rpi(),
        "clips": [],
    }
    temp_pico = resultado["temp_inicio_c"] or 0.0

    for wav_path, truth in zip(archivos_wav, ground_truths):
        dur = duracion_wav(wav_path)
        tiempos = []
        ram_picos = []
        ram_medios = []
        cpu_picos = []
        warmup_t = 0.0
        texto = ""

        for rep in range(N_REPS):
            mon_r = MonitorRAM()
            mon_c = MonitorCPU()
            mon_r.iniciar()
            mon_c.iniciar()
            t0 = time.perf_counter()
            try:
                rec = KaldiRecognizer(vosk_model, 16000)
                rec.SetMaxAlternatives(0)
                with wave.open(wav_path, "rb") as wf:
                    while True:
                        data = wf.readframes(chunk_frames)
                        if not data:
                            break
                        rec.AcceptWaveform(data)
                res = json.loads(rec.FinalResult())
                texto = res.get("text", "")
            except Exception as e:
                mon_r.detener()  # descartado en error
                mon_c.detener()
                print(f"    [ERROR rep {rep}]: {e}")
                break
            t_inf = time.perf_counter() - t0
            ram_p, ram_m = mon_r.detener()
            cpu_p, _ = mon_c.detener()
            t_cpu = leer_temperatura_cpu()
            if t_cpu and t_cpu > temp_pico:
                temp_pico = t_cpu
            if rep == 0:
                warmup_t = t_inf
            else:
                tiempos.append(t_inf)
                ram_picos.append(ram_p)
                ram_medios.append(ram_m)
                cpu_picos.append(cpu_p)

        if not tiempos:
            continue

        wer_val = calcular_wer(truth, texto)
        rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
        media_t = sum(tiempos) / len(tiempos)
        media_rtf = sum(rtfs) / len(rtfs)

        resultado["clips"].append({
            "archivo": os.path.basename(wav_path),
            "grupo": GRUPO_MAP_STT.get(truth, "media"),
            "ground_truth": truth,
            "transcripcion": texto,
            "duracion_audio_s": round(dur, 2),
            "tiempo_s": round(media_t, 3),
            "std_tiempo_s": _std(tiempos),
            "warmup_s": round(warmup_t, 3),
            "rtf": round(media_rtf, 3),
            "std_rtf": _std(rtfs),
            "wer": round(wer_val, 3),
            "ram_pico_mb": fmt_mb(max(ram_picos)) if ram_picos else 0.0,
            "ram_media_mb": fmt_mb(sum(ram_medios) / len(ram_medios)) if ram_medios else 0.0,
            "cpu_pct_pico": round(max(cpu_picos), 1) if cpu_picos else 0.0,
        })
        print(f"    {os.path.basename(wav_path)}: \"{texto[:40]}\" | "
              f"WER={wer_val:.2f} | RTF={media_rtf:.2f} | {media_t:.2f}±{_std(tiempos):.3f}s")

    clips = resultado["clips"]
    if clips:
        resultado["promedios"] = {
            "wer": round(sum(c["wer"] for c in clips) / len(clips), 3),
            "rtf": round(sum(c["rtf"] for c in clips) / len(clips), 3),
            "std_rtf": round(sum(c["std_rtf"] for c in clips) / len(clips), 3),
            "tiempo_s": round(sum(c["tiempo_s"] for c in clips) / len(clips), 3),
            "std_tiempo_s": round(sum(c["std_tiempo_s"] for c in clips) / len(clips), 3),
            "ram_pico_mb": max(c["ram_pico_mb"] for c in clips) if clips else 0.0,
            "ram_media_mb": round(sum(c["ram_media_mb"] for c in clips) / len(clips), 1),
            "cpu_pct_pico": max(c.get("cpu_pct_pico", 0) for c in clips),
        }

    resultado["temp_pico_c"] = temp_pico if temp_pico else None
    resultado["temp_fin_c"] = leer_temperatura_cpu()
    resultado["throttling_fin"] = leer_throttling_rpi()

    del vosk_model
    gc.collect()
    return resultado


def benchmark_stt_mini(archivos_wav, ground_truths, models_dir):
    resultados = []
    for cfg in STT_CONFIGS:
        esperar_enfriamiento()
        if cfg["motor"] == "whisper":
            r = benchmark_stt_whisper(cfg, archivos_wav, ground_truths, models_dir)
        else:
            r = benchmark_stt_vosk(cfg, archivos_wav, ground_truths, models_dir)
        if r:
            resultados.append(r)
    return resultados


# ---------------------------------------------------------------------------
# TTS BENCHMARKS
# ---------------------------------------------------------------------------

def _registrar_resultado_tts(motor, nombre, config_dict):
    return {
        "motor": motor,
        "nombre": nombre,
        "config": config_dict,
        "n_reps": N_REPS,
        "temp_inicio_c": leer_temperatura_cpu(),
        "throttling_inicio": leer_throttling_rpi(),
        "frases": [],
    }


def _agregar_promedios_tts(resultado, todos_tiempos, todos_duraciones, streaming=False):
    datos = resultado["frases"]
    if not datos:
        return
    n = len(datos)
    total_dur = sum(todos_duraciones)
    rtf_global = round(sum(todos_tiempos) / total_dur, 3) if total_dur > 0 else 0.0
    ttfbs = [d["ttfb_s"] for d in datos if d.get("ttfb_s") is not None]
    resultado["promedios"] = {
        "tiempo_sintesis_s": round(sum(d["tiempo_sintesis_s"] for d in datos) / n, 3),
        "std_tiempo_s": _std(todos_tiempos),
        "ttfb_s": round(sum(ttfbs) / len(ttfbs), 4) if ttfbs else None,
        "es_streaming": streaming,
        "rtf": rtf_global,
        "std_rtf": _std([d["rtf"] for d in datos]),
        "ram_pico_mb": max(d["ram_pico_mb"] for d in datos),
        "ram_media_mb": round(sum(d["ram_media_mb"] for d in datos) / n, 1),
        "cpu_pct_pico": max(d.get("cpu_pct_pico", 0) for d in datos),
        "cpu_pct_medio": round(sum(d.get("cpu_pct_medio", 0) for d in datos) / n, 1),
        "wer_medio": None,
    }


def benchmark_tts_piper(cfg, frases, output_dir, models_dir):
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig
    except ImportError:
        print("  [SKIP] piper-tts no instalado. pip install piper-tts")
        return None

    piper_dir = os.path.join(models_dir, "piper")
    modelos_onnx = []
    if os.path.isdir(piper_dir):
        modelos_onnx = [
            os.path.join(piper_dir, f)
            for f in sorted(os.listdir(piper_dir))
            if f.endswith(".onnx") and not f.endswith(".onnx.json")
        ]
    if not modelos_onnx:
        print(f"  [SKIP] No se encontraron modelos .onnx en {piper_dir}")
        return None

    modelo_onnx = modelos_onnx[0]
    modelo_nombre = os.path.splitext(os.path.basename(modelo_onnx))[0]
    nombre = f"piper_{modelo_nombre}_default"

    mon = MonitorRAM()
    mon.iniciar()
    t0 = time.perf_counter()
    try:
        voice = PiperVoice.load(modelo_onnx)
    except Exception as e:
        mon.detener()
        print(f"  [ERROR] {e}")
        return None
    t_carga = time.perf_counter() - t0
    ram_carga, _ = mon.detener()
    print(f"  Piper ({modelo_nombre}) cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

    params = cfg["params"]
    syn_config = SynthesisConfig(
        length_scale=params["length_scale"],
        noise_scale=params["noise_scale"],
        noise_w_scale=params["noise_w"],
    )

    resultado = _registrar_resultado_tts("piper", nombre, params)
    resultado["modelo"] = modelo_nombre
    resultado["tiempo_carga_s"] = round(t_carga, 2)
    resultado["ram_carga_mb"] = fmt_mb(ram_carga)

    subdir = os.path.join(output_dir, nombre)
    os.makedirs(subdir, exist_ok=True)

    todos_tiempos = []
    todos_duraciones = []
    temp_pico = resultado["temp_inicio_c"] or 0.0

    for i, frase in enumerate(frases):
        wav_path = os.path.join(subdir, f"frase_{i:02d}.wav")
        tiempos, ttfbs, ram_picos, ram_medios, cpu_picos, cpu_medios = [], [], [], [], [], []
        warmup_t = 0.0

        for rep in range(N_REPS):
            mon_r = MonitorRAM()
            mon_c = MonitorCPU()
            mon_r.iniciar()
            mon_c.iniciar()
            t0 = time.perf_counter()
            t_first = None
            try:
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(voice.config.sample_rate)
                    for chunk in voice.synthesize(frase, syn_config=syn_config):
                        if t_first is None:
                            t_first = time.perf_counter()
                        wf.writeframes(chunk.audio_int16_bytes)
            except Exception as e:
                mon_r.detener()  # descartado en error
                mon_c.detener()
                print(f"    [ERROR rep {rep}]: {e}")
                break
            t_sint = time.perf_counter() - t0
            ttfb = (t_first - t0) if t_first else t_sint
            ram_p, ram_m = mon_r.detener()
            cpu_p, cpu_m = mon_c.detener()
            t_cpu = leer_temperatura_cpu()
            if t_cpu and t_cpu > temp_pico:
                temp_pico = t_cpu
            if rep == 0:
                warmup_t = t_sint
            else:
                tiempos.append(t_sint)
                ttfbs.append(ttfb)
                ram_picos.append(ram_p)
                ram_medios.append(ram_m)
                cpu_picos.append(cpu_p)
                cpu_medios.append(cpu_m)

        if not tiempos:
            continue

        dur = duracion_wav(wav_path)
        rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
        media_t = sum(tiempos) / len(tiempos)
        todos_tiempos.extend(tiempos)
        todos_duraciones.extend([dur] * len(tiempos))

        resultado["frases"].append({
            "frase": frase,
            "grupo": GRUPO_MAP_TTS.get(frase, "media"),
            "wav": wav_path,
            "tiempo_sintesis_s": round(media_t, 3),
            "std_tiempo_s": _std(tiempos),
            "warmup_s": round(warmup_t, 3),
            "ttfb_s": round(sum(ttfbs) / len(ttfbs), 4),
            "duracion_audio_s": round(dur, 2),
            "rtf": round(sum(rtfs) / len(rtfs), 3),
            "std_rtf": _std(rtfs),
            "ram_pico_mb": fmt_mb(max(ram_picos)),
            "ram_media_mb": fmt_mb(sum(ram_medios) / len(ram_medios)) if ram_medios else fmt_mb(max(ram_picos)),
            "cpu_pct_pico": round(max(cpu_picos), 1) if cpu_picos else 0.0,
            "cpu_pct_medio": round(sum(cpu_medios) / len(cpu_medios), 1) if cpu_medios else 0.0,
            "wer": None,
        })
        print(f"    [{i+1}/{len(frases)}] \"{frase[:35]}\" | "
              f"{media_t:.2f}±{_std(tiempos):.3f}s | TTFB={ttfbs[-1]:.3f}s | RTF={rtfs[-1]:.2f}")

    _agregar_promedios_tts(resultado, todos_tiempos, todos_duraciones, streaming=True)
    resultado["temp_pico_c"] = temp_pico if temp_pico else None
    resultado["temp_fin_c"] = leer_temperatura_cpu()
    resultado["throttling_fin"] = leer_throttling_rpi()

    del voice
    gc.collect()
    return resultado


def benchmark_tts_coqui(cfg, frases, output_dir):
    try:
        from TTS.api import TTS
    except ImportError:
        print(f"  [SKIP] Coqui TTS no instalado. pip install TTS")
        return None

    motor = cfg["motor"]
    nombre = cfg["nombre"]
    model = cfg["model"]
    ram_minima = cfg.get("ram_minima_mb", 0)

    ram_libre = psutil.virtual_memory().available / (1024 * 1024)
    if ram_minima and ram_libre < ram_minima:
        print(f"  [SKIP] {nombre}: RAM libre {ram_libre:.0f} MB < {ram_minima} MB requeridos")
        return {"motor": motor, "nombre": nombre, "skip": f"RAM insuficiente ({ram_libre:.0f} MB libre)"}

    print(f"\n  --- {nombre} ({model}) ---")
    es_multilingual = "multilingual" in model or "xtts" in model.lower()

    mon = MonitorRAM()
    mon.iniciar()
    t0 = time.perf_counter()
    try:
        tts = TTS(model_name=model, progress_bar=False, gpu=False)
    except Exception as e:
        mon.detener()
        print(f"  [ERROR] {e}")
        return {"motor": motor, "nombre": nombre, "error": str(e)}
    t_carga = time.perf_counter() - t0
    ram_carga, _ = mon.detener()
    print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

    resultado = _registrar_resultado_tts(motor, nombre, {"model": model, "length_scale": cfg.get("length_scale", 1.0)})
    resultado["tiempo_carga_s"] = round(t_carga, 2)
    resultado["ram_carga_mb"] = fmt_mb(ram_carga)

    subdir = os.path.join(output_dir, nombre)
    os.makedirs(subdir, exist_ok=True)

    # XTTS-v2 necesita WAV de referencia de hablante
    speaker_wav_ref = None
    if es_multilingual:
        import glob as _glob
        wavs_ref = sorted(_glob.glob(os.path.join(output_dir, "**", "*.wav"), recursive=True))
        if wavs_ref:
            speaker_wav_ref = wavs_ref[0]
        else:
            print("  [WARN] XTTS-v2: no se encontró WAV de referencia en output_dir")

    _DIGITOS_ES = {"0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
                   "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve"}

    todos_tiempos = []
    todos_duraciones = []
    temp_pico = resultado["temp_inicio_c"] or 0.0
    length_scale = cfg.get("length_scale", 1.0)

    for i, frase in enumerate(frases):
        wav_path = os.path.join(subdir, f"frase_{i:02d}.wav")
        tiempos, ram_picos, ram_medios, cpu_picos, cpu_medios = [], [], [], [], []
        warmup_t = 0.0

        texto_tts = frase
        if not es_multilingual:
            texto_tts = re.sub(r"[¿¡]", "", texto_tts)
            texto_tts = re.sub(r"\d", lambda m: _DIGITOS_ES.get(m.group(), ""), texto_tts)

        for rep in range(N_REPS):
            mon_r = MonitorRAM()
            mon_c = MonitorCPU()
            mon_r.iniciar()
            mon_c.iniciar()
            t0 = time.perf_counter()
            try:
                kwargs = {}
                if length_scale != 1.0:
                    kwargs["length_scale"] = length_scale
                if es_multilingual:
                    if speaker_wav_ref:
                        kwargs["speaker_wav"] = speaker_wav_ref
                    tts.tts_to_file(text=texto_tts, file_path=wav_path, language="es", **kwargs)
                else:
                    tts.tts_to_file(text=texto_tts, file_path=wav_path, **kwargs)
            except Exception as e:
                mon_r.detener()  # descartado en error
                mon_c.detener()
                print(f"    [ERROR rep {rep}]: {e}")
                break
            t_sint = time.perf_counter() - t0
            ram_p, ram_m = mon_r.detener()
            cpu_p, cpu_m = mon_c.detener()
            t_cpu = leer_temperatura_cpu()
            if t_cpu and t_cpu > temp_pico:
                temp_pico = t_cpu
            if rep == 0:
                warmup_t = t_sint
            else:
                tiempos.append(t_sint)
                ram_picos.append(ram_p)
                ram_medios.append(ram_m)
                cpu_picos.append(cpu_p)
                cpu_medios.append(cpu_m)

        if not tiempos:
            continue

        dur = duracion_wav(wav_path)
        rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
        media_t = sum(tiempos) / len(tiempos)
        todos_tiempos.extend(tiempos)
        todos_duraciones.extend([dur] * len(tiempos))

        resultado["frases"].append({
            "frase": frase,
            "grupo": GRUPO_MAP_TTS.get(frase, "media"),
            "wav": wav_path,
            "tiempo_sintesis_s": round(media_t, 3),
            "std_tiempo_s": _std(tiempos),
            "warmup_s": round(warmup_t, 3),
            "ttfb_s": None,
            "duracion_audio_s": round(dur, 2),
            "rtf": round(sum(rtfs) / len(rtfs), 3),
            "std_rtf": _std(rtfs),
            "ram_pico_mb": fmt_mb(max(ram_picos)),
            "ram_media_mb": fmt_mb(sum(ram_medios) / len(ram_medios)) if ram_medios else fmt_mb(max(ram_picos)),
            "cpu_pct_pico": round(max(cpu_picos), 1) if cpu_picos else 0.0,
            "cpu_pct_medio": round(sum(cpu_medios) / len(cpu_medios), 1) if cpu_medios else 0.0,
            "wer": None,
        })
        print(f"    [{i+1}/{len(frases)}] \"{frase[:35]}\" | "
              f"{media_t:.2f}±{_std(tiempos):.3f}s | RTF={rtfs[-1]:.2f}")

    _agregar_promedios_tts(resultado, todos_tiempos, todos_duraciones, streaming=False)
    resultado["temp_pico_c"] = temp_pico if temp_pico else None
    resultado["temp_fin_c"] = leer_temperatura_cpu()
    resultado["throttling_fin"] = leer_throttling_rpi()

    del tts
    gc.collect()
    return resultado


def benchmark_tts_kitten(cfg, frases, output_dir):
    try:
        from kittentts import KittenTTS
    except ImportError:
        print("  [SKIP] KittenTTS no instalado. pip install kittentts")
        return None

    nombre = cfg["nombre"]
    print(f"\n  --- {nombre} (voice={cfg['voice']}, speed={cfg['speed']}, temp={cfg['temperature']}) ---")

    mon = MonitorRAM()
    mon.iniciar()
    t0 = time.perf_counter()
    try:
        model = KittenTTS(voice=cfg["voice"])
    except TypeError:
        try:
            model = KittenTTS()
        except Exception as e:
            mon.detener()
            print(f"  [ERROR] {e}")
            return {"motor": "kitten", "nombre": nombre, "error": str(e)}
    except Exception as e:
        mon.detener()
        print(f"  [ERROR] {e}")
        return {"motor": "kitten", "nombre": nombre, "error": str(e)}

    t_carga = time.perf_counter() - t0
    ram_carga, _ = mon.detener()
    print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

    resultado = _registrar_resultado_tts("kitten", nombre, cfg.copy())
    resultado["tiempo_carga_s"] = round(t_carga, 2)
    resultado["ram_carga_mb"] = fmt_mb(ram_carga)

    subdir = os.path.join(output_dir, nombre)
    os.makedirs(subdir, exist_ok=True)

    todos_tiempos = []
    todos_duraciones = []
    temp_pico = resultado["temp_inicio_c"] or 0.0

    for i, frase in enumerate(frases):
        wav_path = os.path.join(subdir, f"frase_{i:02d}.wav")
        tiempos, ram_picos, ram_medios, cpu_picos, cpu_medios = [], [], [], [], []
        warmup_t = 0.0

        for rep in range(N_REPS):
            mon_r = MonitorRAM()
            mon_c = MonitorCPU()
            mon_r.iniciar()
            mon_c.iniciar()
            t0 = time.perf_counter()
            generated = False
            for fn in [
                lambda: model.generate(frase, voice=cfg["voice"],
                                       speed=cfg["speed"], temperature=cfg["temperature"],
                                       output=wav_path),
                lambda: model.synthesize(frase, speed=cfg["speed"],
                                         temperature=cfg["temperature"],
                                         output_path=wav_path),
                lambda: model.tts(frase, wav_path),
            ]:
                try:
                    fn()
                    generated = True
                    break
                except (TypeError, AttributeError):
                    continue
                except Exception as e:
                    print(f"    [ERROR rep {rep}]: {e}")
                    break

            t_sint = time.perf_counter() - t0
            ram_p, ram_m = mon_r.detener()
            cpu_p, cpu_m = mon_c.detener()
            t_cpu = leer_temperatura_cpu()
            if t_cpu and t_cpu > temp_pico:
                temp_pico = t_cpu
            if not generated:
                break
            if rep == 0:
                warmup_t = t_sint
            else:
                tiempos.append(t_sint)
                ram_picos.append(ram_p)
                ram_medios.append(ram_m)
                cpu_picos.append(cpu_p)
                cpu_medios.append(cpu_m)

        if not tiempos:
            continue

        dur = duracion_wav(wav_path)
        rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
        media_t = sum(tiempos) / len(tiempos)
        todos_tiempos.extend(tiempos)
        todos_duraciones.extend([dur] * len(tiempos))

        resultado["frases"].append({
            "frase": frase,
            "grupo": GRUPO_MAP_TTS.get(frase, "media"),
            "wav": wav_path,
            "tiempo_sintesis_s": round(media_t, 3),
            "std_tiempo_s": _std(tiempos),
            "warmup_s": round(warmup_t, 3),
            "ttfb_s": None,
            "duracion_audio_s": round(dur, 2),
            "rtf": round(sum(rtfs) / len(rtfs), 3),
            "std_rtf": _std(rtfs),
            "ram_pico_mb": fmt_mb(max(ram_picos)),
            "ram_media_mb": fmt_mb(sum(ram_medios) / len(ram_medios)) if ram_medios else fmt_mb(max(ram_picos)),
            "cpu_pct_pico": round(max(cpu_picos), 1) if cpu_picos else 0.0,
            "cpu_pct_medio": round(sum(cpu_medios) / len(cpu_medios), 1) if cpu_medios else 0.0,
            "wer": None,
        })
        print(f"    [{i+1}/{len(frases)}] \"{frase[:35]}\" | "
              f"{media_t:.2f}±{_std(tiempos):.3f}s | RTF={rtfs[-1]:.2f}")

    _agregar_promedios_tts(resultado, todos_tiempos, todos_duraciones, streaming=False)
    resultado["temp_pico_c"] = temp_pico if temp_pico else None
    resultado["temp_fin_c"] = leer_temperatura_cpu()
    resultado["throttling_fin"] = leer_throttling_rpi()

    del model
    gc.collect()
    return resultado


def benchmark_tts_mini(frases, output_dir, models_dir):
    resultados = []
    for cfg in TTS_CONFIGS:
        print(f"\n  Config: {cfg['nombre']}")
        esperar_enfriamiento()
        motor = cfg["motor"]
        try:
            if motor == "piper":
                r = benchmark_tts_piper(cfg, frases, output_dir, models_dir)
            elif motor in ("coqui_vits", "coqui_xtts"):
                r = benchmark_tts_coqui(cfg, frases, output_dir)
            elif motor == "kitten":
                r = benchmark_tts_kitten(cfg, frases, output_dir)
            else:
                r = None
        except Exception as e:
            print(f"  [ERROR] {cfg['nombre']}: {e}")
            r = {"motor": motor, "nombre": cfg["nombre"], "error": str(e)}
        if r is not None:
            resultados.append(r)
    return resultados


# ---------------------------------------------------------------------------
# EVALUACIÓN WER TTS
# ---------------------------------------------------------------------------

def evaluar_calidad_tts(resultados_tts, no_quality=False):
    if no_quality:
        print("\n  [INFO] WER TTS omitido (--no-quality).")
        return

    try:
        import whisper
    except ImportError:
        print("\n  [SKIP] openai-whisper no instalado. pip install openai-whisper")
        return

    ram_libre = psutil.virtual_memory().available / (1024 * 1024)
    if ram_libre < 500:
        print(f"\n  [SKIP] WER TTS: RAM libre insuficiente ({ram_libre:.0f} MB)")
        return

    print("\n" + "=" * 50)
    print("EVALUACIÓN WER TTS — Whisper tiny")
    print("=" * 50)
    print("  Cargando Whisper tiny...")

    try:
        modelo_whisper = whisper.load_model("tiny")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    try:
        import soundfile as _sf
        import numpy as _np
    except ImportError:
        print("  [SKIP] soundfile no instalado. pip install soundfile")
        del modelo_whisper
        gc.collect()
        return

    for resultado in resultados_tts:
        if resultado is None or "frases" not in resultado:
            continue
        nombre = resultado.get("nombre", "?")
        wers = []
        for fd in resultado["frases"]:
            wav_path = fd.get("wav", "")
            ref = fd.get("frase", "")
            if not wav_path or not os.path.exists(wav_path):
                continue
            try:
                audio_np, sr = _sf.read(wav_path, dtype="float32")
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if sr != 16000:
                    n_out = int(len(audio_np) * 16000 / sr)
                    audio_np = _np.interp(
                        _np.linspace(0, len(audio_np) - 1, n_out),
                        _np.arange(len(audio_np)), audio_np
                    ).astype(_np.float32)
                out = modelo_whisper.transcribe(audio_np, language="es", fp16=False)
                hyp = out.get("text", "")
                wer = _wer_tts(ref, hyp)
                fd["wer"] = wer
                wers.append(wer)
            except Exception as e:
                print(f"    [WARN] WER frase: {e}")

        if wers and "promedios" in resultado:
            media_wer = round(sum(wers) / len(wers), 4)
            resultado["promedios"]["wer_medio"] = media_wer
            print(f"  {nombre[:40]:<40} WER={media_wer:.4f}")

    del modelo_whisper
    gc.collect()


# ---------------------------------------------------------------------------
# GRÁFICAS
# ---------------------------------------------------------------------------

def generar_graficas_mini(resultados_stt, resultados_tts, graficas_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("  [SKIP] matplotlib no instalado. pip install matplotlib")
        return []

    os.makedirs(graficas_dir, exist_ok=True)
    archivos = []

    COLORES_STT = {"faster-whisper": "#2196F3", "vosk": "#03A9F4"}
    COLORES_TTS = {"piper": "#4CAF50", "coqui_vits": "#FF9800", "coqui_xtts": "#E91E63", "kitten": "#9C27B0"}
    COLORES_GRUPO = {"corta": "#66BB6A", "media": "#FFA726", "larga": "#EF5350"}

    stt_validos = [r for r in resultados_stt if r and "promedios" in r]
    tts_validos = [r for r in resultados_tts if r and "promedios" in r]

    if not stt_validos and not tts_validos:
        return archivos

    # -----------------------------------------------------------------------
    # 1. RTF comparativo STT + TTS (barras, 2 subplots)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("bench_mini — RTF por configuración (RPi4)", fontsize=13)

    for ax, datos, colores, titulo in [
        (axes[0], stt_validos, COLORES_STT, "STT — Real-Time Factor"),
        (axes[1], tts_validos, COLORES_TTS, "TTS — Real-Time Factor"),
    ]:
        if not datos:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(titulo)
            continue
        nombres = [r["nombre"][:22] for r in datos]
        rtfs = [r["promedios"].get("rtf") or 0 for r in datos]
        cols = [colores.get(r.get("motor", "?"), "#9E9E9E") for r in datos]
        x = range(len(nombres))
        bars = ax.bar(x, rtfs, color=cols, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, rtfs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="RTF=1 (tiempo real)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(nombres, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("RTF (menor = mejor)")
        ax.set_title(titulo)
        ax.legend(fontsize=8)
        motores_presentes = list(dict.fromkeys(r.get("motor", "?") for r in datos))
        leyenda = [Patch(color=colores.get(m, "#9E9E9E"), label=m) for m in motores_presentes]
        ax.legend(handles=leyenda, fontsize=7, loc="upper right")

    fig.tight_layout()
    ruta = os.path.join(graficas_dir, "01_rtf_comparativa.png")
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    archivos.append(ruta)
    print(f"  Gráfica: {ruta}")

    # -----------------------------------------------------------------------
    # 2. WER vs RTF scatter (STT) — burbuja = RAM pico
    # -----------------------------------------------------------------------
    if stt_validos:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        rams = [r["promedios"].get("ram_pico_mb", 100) for r in stt_validos]
        ram_max = max(rams) or 1
        for r, ram in zip(stt_validos, rams):
            p = r["promedios"]
            rtf = p.get("rtf", 0)
            wer = p.get("wer", 0)
            color = COLORES_STT.get(r.get("motor", "?"), "#9E9E9E")
            size = 60 + 600 * (ram / ram_max)
            ax2.scatter(rtf, wer, s=size, color=color, alpha=0.8, edgecolors="white", linewidth=0.8, zorder=3)
            ax2.annotate(r["nombre"][:20], (rtf, wer), textcoords="offset points",
                         xytext=(6, 4), fontsize=7.5)
        ax2.set_xlabel("RTF (↓ mejor)")
        ax2.set_ylabel("WER (↓ mejor)")
        ax2.set_title("STT — Velocidad vs Precisión\n(tamaño burbuja ∝ RAM pico)")
        ax2.grid(True, alpha=0.25)
        leyenda2 = [Patch(color=c, label=m) for m, c in COLORES_STT.items()
                    if any(r.get("motor") == m for r in stt_validos)]
        ax2.legend(handles=leyenda2, fontsize=8)
        fig2.tight_layout()
        ruta2 = os.path.join(graficas_dir, "02_stt_wer_vs_rtf.png")
        fig2.savefig(ruta2, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        archivos.append(ruta2)
        print(f"  Gráfica: {ruta2}")

    # -----------------------------------------------------------------------
    # 3. RAM pico + media — todos los modelos STT + TTS (barras agrupadas horizontales)
    #    Pico = worst-case; Media = consumo sostenido durante inferencia.
    #    Líneas de referencia para RPi4 con ROS2 activo.
    # -----------------------------------------------------------------------
    todos_r = stt_validos + tts_validos
    if todos_r:
        nombres3 = [r["nombre"][:28] for r in todos_r]
        rams_pico = [r["promedios"].get("ram_pico_mb", 0) for r in todos_r]
        rams_media = [r["promedios"].get("ram_media_mb", 0) for r in todos_r]
        motores3 = [r.get("motor", "?") for r in todos_r]
        cols3 = [COLORES_STT.get(m, COLORES_TTS.get(m, "#9E9E9E")) for m in motores3]

        n3 = len(nombres3)
        alto_barra = 0.35
        fig3, ax3 = plt.subplots(figsize=(10, max(4, n3 * 0.75)))
        y3 = range(n3)

        # Pico: barra entera, color del motor, bordes
        bars_pico = ax3.barh(
            [yi + alto_barra / 2 for yi in y3], rams_pico,
            alto_barra, color=cols3, alpha=0.9, edgecolor="white", label="RAM pico"
        )
        # Media: barra más pequeña, mismo color pero más claro (hatch)
        bars_media = ax3.barh(
            [yi - alto_barra / 2 for yi in y3], rams_media,
            alto_barra, color=cols3, alpha=0.45, edgecolor="white",
            hatch="///", label="RAM media (inferencia)"
        )

        for bar, val in zip(bars_pico, rams_pico):
            ax3.text(val + 8, bar.get_y() + bar.get_height() / 2,
                     f"{val:.0f}", va="center", fontsize=7, fontweight="bold")
        for bar, val in zip(bars_media, rams_media):
            ax3.text(val + 8, bar.get_y() + bar.get_height() / 2,
                     f"{val:.0f}", va="center", fontsize=7, color="#555")

        ax3.axvline(1800, color="red", linestyle="--", linewidth=1.2,
                    label="~1800 MB disponibles (RPi4 con ROS2)")
        ax3.axvline(2500, color="orange", linestyle=":", linewidth=1.2,
                    label="~2500 MB límite XTTS-v2")
        ax3.set_yticks(list(y3))
        ax3.set_yticklabels(nombres3, fontsize=8)
        ax3.set_xlabel("RAM (MB) — menor = mejor")
        ax3.set_title("RAM pico vs media por modelo\n(RPi4 4 GB: ~1800 MB disponibles con ROS2 activo)")
        ax3.legend(fontsize=8, loc="lower right")
        ax3.invert_yaxis()
        fig3.tight_layout()
        ruta3 = os.path.join(graficas_dir, "03_ram_pico_media.png")
        fig3.savefig(ruta3, dpi=120, bbox_inches="tight")
        plt.close(fig3)
        archivos.append(ruta3)
        print(f"  Gráfica: {ruta3}")

    # -----------------------------------------------------------------------
    # 4. RTF por grupo de longitud — TTS (barras agrupadas: motor × grupo)
    #    Revela si un motor escala mal con frases largas o tiene overhead fijo
    # -----------------------------------------------------------------------
    if tts_validos:
        grupos = ["corta", "media", "larga"]
        # Calcular RTF por grupo para cada motor (promedio de rtf de frases de ese grupo)
        rtf_por_grupo = {}
        for r in tts_validos:
            nombre_r = r["nombre"][:18]
            rtf_por_grupo[nombre_r] = {}
            for g in grupos:
                frases_g = [fd for fd in r.get("frases", []) if fd.get("grupo") == g]
                if frases_g:
                    durs = [fd["duracion_audio_s"] for fd in frases_g]
                    tiempos = [fd["tiempo_sintesis_s"] for fd in frases_g]
                    total_dur = sum(durs)
                    rtf_g = sum(tiempos) / total_dur if total_dur > 0 else 0.0
                    rtf_por_grupo[nombre_r][g] = round(rtf_g, 3)

        nombres4 = list(rtf_por_grupo.keys())
        if nombres4 and any(rtf_por_grupo[n] for n in nombres4):
            n_motores = len(nombres4)
            n_grupos = len(grupos)
            ancho = 0.25
            x4 = range(n_grupos)

            fig4, ax4 = plt.subplots(figsize=(9, 5))
            for i, nombre_r in enumerate(nombres4):
                motor_r = next((r.get("motor", "?") for r in tts_validos
                                if r["nombre"][:18] == nombre_r), "?")
                color4 = COLORES_TTS.get(motor_r, "#9E9E9E")
                offset = (i - n_motores / 2 + 0.5) * ancho
                vals = [rtf_por_grupo[nombre_r].get(g, 0) for g in grupos]
                bars4 = ax4.bar([xi + offset for xi in x4], vals,
                                ancho * 0.9, label=nombre_r, color=color4,
                                alpha=0.85, edgecolor="white")
                for bar, val in zip(bars4, vals):
                    if val > 0:
                        ax4.text(bar.get_x() + bar.get_width() / 2,
                                 bar.get_height() + 0.003,
                                 f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

            ax4.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="RTF=1")
            ax4.set_xticks(list(x4))
            ax4.set_xticklabels(["Frases cortas\n(≤5 pal.)", "Frases medias\n(6-12 pal.)",
                                  "Frases largas\n(>12 pal.)"], fontsize=9)
            ax4.set_ylabel("RTF ponderado (menor = mejor)")
            ax4.set_title("TTS — RTF por grupo de longitud\n(detecta overhead fijo vs escalado lineal)")
            ax4.legend(fontsize=7, loc="upper left")
            fig4.tight_layout()
            ruta4 = os.path.join(graficas_dir, "04_tts_rtf_por_longitud.png")
            fig4.savefig(ruta4, dpi=120, bbox_inches="tight")
            plt.close(fig4)
            archivos.append(ruta4)
            print(f"  Gráfica: {ruta4}")

    # -----------------------------------------------------------------------
    # 5. WER STT por grupo de longitud (barras agrupadas: config × grupo)
    #    Muestra en qué tipo de frase falla más cada modelo
    # -----------------------------------------------------------------------
    if stt_validos:
        grupos = ["corta", "media", "larga"]
        wer_por_grupo = {}
        for r in stt_validos:
            nombre_r = r["nombre"][:18]
            wer_por_grupo[nombre_r] = {}
            for g in grupos:
                clips_g = [c for c in r.get("clips", []) if c.get("grupo") == g]
                if clips_g:
                    wer_g = sum(c["wer"] for c in clips_g) / len(clips_g)
                    wer_por_grupo[nombre_r][g] = round(wer_g, 3)

        nombres5 = list(wer_por_grupo.keys())
        if nombres5 and any(wer_por_grupo[n] for n in nombres5):
            n_configs = len(nombres5)
            ancho5 = 0.18
            x5 = range(len(grupos))

            fig5, ax5 = plt.subplots(figsize=(9, 5))
            for i, nombre_r in enumerate(nombres5):
                motor_r = next((r.get("motor", "?") for r in stt_validos
                                if r["nombre"][:18] == nombre_r), "?")
                color5 = COLORES_STT.get(motor_r, "#9E9E9E")
                offset5 = (i - n_configs / 2 + 0.5) * ancho5
                vals5 = [wer_por_grupo[nombre_r].get(g, 0) for g in grupos]
                bars5 = ax5.bar([xi + offset5 for xi in x5], vals5,
                                ancho5 * 0.9, label=nombre_r, color=color5,
                                alpha=0.85, edgecolor="white")
                for bar, val in zip(bars5, vals5):
                    if val > 0:
                        ax5.text(bar.get_x() + bar.get_width() / 2,
                                 bar.get_height() + 0.005,
                                 f"{val:.2f}", ha="center", va="bottom", fontsize=6)

            ax5.set_xticks(list(x5))
            ax5.set_xticklabels(["Frases cortas\n(≤5 pal.)", "Frases medias\n(6-12 pal.)",
                                  "Frases largas\n(>12 pal.)"], fontsize=9)
            ax5.set_ylabel("WER medio (menor = mejor)")
            ax5.set_title("STT — WER por grupo de longitud\n(muestra dónde falla cada modelo)")
            max_wer = max((v for n in nombres5 for v in wer_por_grupo[n].values()), default=0.1)
            ax5.set_ylim(0, min(1.05, max_wer * 1.3 + 0.02))
            ax5.legend(fontsize=7, loc="upper left")
            fig5.tight_layout()
            ruta5 = os.path.join(graficas_dir, "05_stt_wer_por_longitud.png")
            fig5.savefig(ruta5, dpi=120, bbox_inches="tight")
            plt.close(fig5)
            archivos.append(ruta5)
            print(f"  Gráfica: {ruta5}")

    return archivos


# ---------------------------------------------------------------------------
# INFORME MARKDOWN
# ---------------------------------------------------------------------------

def generar_informe_mini(resultados_stt, resultados_tts, archivos_graficas, out_path):
    mem = psutil.virtual_memory()
    lineas = [
        "# bench_mini — Informe Benchmark Representativo RPi4",
        "",
        f"**Fecha:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Plataforma:** {platform.machine()} — {platform.system()} {platform.release()}  ",
        f"**CPU cores:** {psutil.cpu_count(logical=True)}  ",
        f"**RAM total:** {fmt_mb(mem.total)} MB | RAM libre inicio: {fmt_mb(mem.available)} MB  ",
        "",
        "## Metodología",
        "",
        f"- N_REPS={N_REPS} (rep 0 = warmup descartado; {N_REPS-1} medida(s) válida(s))",
        "- Corpus STT: 9 frases (3 cortas / 3 medias / 3 largas) — mismas del bench exhaustivo.",
        "- Corpus TTS: 9 frases del CORPUS_MINI.",
        "- RTF ponderado: Σtiempos / Σduraciones.",
        "- WER STT: diacríticos eliminados antes de comparar (normalizar()).",
        "- WER TTS: Whisper tiny ES sobre los WAVs generados (si --no-quality, se omite).",
        "- Cooldown térmico entre configs (esperar <55°C).",
        "",
        "## STT — Resultados",
        "",
        "| Config | WER | RTF±std | Carga(s) | RAM(MB) | CPU%pico | Throttle |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for r in resultados_stt:
        if not r or "promedios" not in r:
            estado = r.get("skip") or r.get("error") or "sin datos"
            lineas.append(f"| {r.get('nombre','?')} | — | — | — | — | — | — | _{estado}_ |")
            continue
        p = r["promedios"]
        tf = r.get("throttling_fin")
        tf_str = f"0x{tf:x}" if tf is not None else "N/A"
        lineas.append(
            f"| {r['nombre'][:30]} | {p.get('wer','N/A')} | "
            f"{p.get('rtf','N/A')}±{p.get('std_rtf','N/A')} | "
            f"{r.get('tiempo_carga_s','N/A')} | {p.get('ram_pico_mb','N/A')} | "
            f"{p.get('cpu_pct_pico','N/A')} | {tf_str} |"
        )

    lineas += [
        "",
        "## TTS — Resultados",
        "",
        "| Config | Motor | RTF±std | Carga(s) | RAM(MB) | TTFB(s)* | WER | CPU%pico | Throttle |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for r in resultados_tts:
        if not r or "promedios" not in r:
            estado = r.get("skip") or r.get("error") or "sin datos"
            lineas.append(f"| {r.get('nombre','?')} | — | — | — | — | — | — | — | — | _{estado}_ |")
            continue
        p = r["promedios"]
        tf = r.get("throttling_fin")
        tf_str = f"0x{tf:x}" if tf is not None else "N/A"
        ttfb_str = str(p.get("ttfb_s")) if p.get("es_streaming") else "—"
        lineas.append(
            f"| {r['nombre'][:25]} | {r.get('motor','?')} | "
            f"{p.get('rtf','N/A')}±{p.get('std_rtf','N/A')} | "
            f"{r.get('tiempo_carga_s','N/A')} | {p.get('ram_pico_mb','N/A')} | "
            f"{ttfb_str} | {p.get('wer_medio','N/A')} | "
            f"{p.get('cpu_pct_pico','N/A')} | {tf_str} |"
        )

    lineas += ["", "> \\* TTFB real solo Piper (streaming). Coqui/Kitten: latencia total.", ""]

    if archivos_graficas:
        lineas.append("## Gráficas\n")
        for ruta in archivos_graficas:
            nombre_img = os.path.basename(ruta)
            ruta_rel = os.path.relpath(ruta, os.path.dirname(out_path))
            lineas.append(f"![{nombre_img}]({ruta_rel})\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print(f"  Informe: {out_path}")


# ---------------------------------------------------------------------------
# EXPORTAR CSV
# ---------------------------------------------------------------------------

def exportar_csv_mini(resultados_stt, resultados_tts, out_path):
    campos = ["tipo", "motor", "nombre", "frase_o_clip", "grupo", "tiempo_s",
              "std_tiempo_s", "rtf", "wer", "ram_pico_mb", "cpu_pct_pico"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        for r in resultados_stt:
            if not r or "clips" not in r:
                continue
            for c in r["clips"]:
                writer.writerow({
                    "tipo": "stt", "motor": r.get("motor"), "nombre": r.get("nombre"),
                    "frase_o_clip": c.get("archivo"), "grupo": c.get("grupo"),
                    "tiempo_s": c.get("tiempo_s"), "std_tiempo_s": c.get("std_tiempo_s"),
                    "rtf": c.get("rtf"), "wer": c.get("wer"),
                    "ram_pico_mb": c.get("ram_pico_mb"), "cpu_pct_pico": c.get("cpu_pct_pico"),
                })
        for r in resultados_tts:
            if not r or "frases" not in r:
                continue
            for fd in r["frases"]:
                writer.writerow({
                    "tipo": "tts", "motor": r.get("motor"), "nombre": r.get("nombre"),
                    "frase_o_clip": fd.get("frase", "")[:40], "grupo": fd.get("grupo"),
                    "tiempo_s": fd.get("tiempo_sintesis_s"), "std_tiempo_s": fd.get("std_tiempo_s"),
                    "rtf": fd.get("rtf"), "wer": fd.get("wer"),
                    "ram_pico_mb": fd.get("ram_pico_mb"), "cpu_pct_pico": fd.get("cpu_pct_pico"),
                })
    print(f"  CSV: {out_path}")


# ---------------------------------------------------------------------------
# TABLA CONSOLA
# ---------------------------------------------------------------------------

def imprimir_tabla_consola_mini(resultados_stt, resultados_tts):
    print("\n" + "=" * 90)
    print("RESUMEN BENCH MINI")
    print("=" * 90)

    def _fila(r, tipo):
        if not r or "promedios" not in r:
            return [tipo, r.get("nombre","?") if r else "?", r.get("skip") or r.get("error","?"), "—", "—", "—", "—"]
        p = r["promedios"]
        wer = p.get("wer") or p.get("wer_medio") or "N/A"
        return [tipo, r["nombre"][:30], r.get("motor","?"),
                f"{p.get('rtf','N/A')}±{p.get('std_rtf','N/A')}",
                str(p.get("ram_pico_mb","N/A")), str(wer),
                f"{r.get('tiempo_carga_s','N/A')}s carga"]

    enc = ["Tipo", "Config", "Motor", "RTF±std", "RAM(MB)", "WER", "Extra"]
    filas = ([_fila(r, "STT") for r in resultados_stt] +
             [_fila(r, "TTS") for r in resultados_tts])

    anchos = [max(len(str(f[i])) for f in [enc] + filas) for i in range(len(enc))]
    fmt = " | ".join(f"{{:<{a}}}" for a in anchos)
    print(fmt.format(*enc))
    print("-+-".join("-" * a for a in anchos))
    for fila in filas:
        print(fmt.format(*[str(c) for c in fila]))
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def resolver_archivos_stt(audios_dir, base_dir):
    candidatos = [audios_dir, os.path.join(base_dir, "audios"), os.path.join(base_dir, "audio_tests")]
    for d in candidatos:
        if not os.path.isdir(d):
            continue
        # Buscar WAVs directamente en d
        wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        if wavs:
            archivos = [os.path.join(d, w) for w in wavs]
            n = min(len(archivos), len(FRASES_STT))
            return archivos[:n], FRASES_STT[:n]
        # Si no hay WAVs directos, buscar en subdirectorios (e.g. audios/abuela/)
        subdirs = sorted(
            sd for sd in os.listdir(d)
            if os.path.isdir(os.path.join(d, sd))
        )
        for sd in subdirs:
            ruta_sd = os.path.join(d, sd)
            wavs_sd = sorted(f for f in os.listdir(ruta_sd) if f.endswith(".wav"))
            if wavs_sd:
                archivos = [os.path.join(ruta_sd, w) for w in wavs_sd]
                n = min(len(archivos), len(FRASES_STT))
                print(f"  [INFO] Usando subdirectorio: {ruta_sd} ({len(archivos)} WAVs)")
                return archivos[:n], FRASES_STT[:n]
    return [], []


def main():
    parser = argparse.ArgumentParser(
        description="bench_mini — Benchmark representativo RPi4 (1 config/motor)"
    )
    parser.add_argument("--no-quality", action="store_true",
                        help="Omitir WER TTS con Whisper (ahorra RAM y tiempo)")
    parser.add_argument("--skip-stt", action="store_true",
                        help="Saltar benchmarks STT")
    parser.add_argument("--skip-tts", action="store_true",
                        help="Saltar benchmarks TTS")
    parser.add_argument("--output-dir", default=None,
                        help="Directorio de salida (default: resultados/)")
    parser.add_argument("--audios-dir", default=None,
                        help="Directorio con WAVs para STT (default: audios/)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(base_dir, "resultados")
    graficas_dir = os.path.join(output_dir, "graficas_mini")
    json_path = os.path.join(output_dir, "bench_mini.json")
    csv_path = os.path.join(output_dir, "bench_mini.csv")
    md_path = os.path.join(output_dir, "bench_mini_informe.md")

    os.makedirs(output_dir, exist_ok=True)

    models_dir_local = os.path.join(base_dir, "models")
    models_dir_global = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "..", "models"))
    models_dir = models_dir_local if os.path.isdir(models_dir_local) else models_dir_global

    mem = psutil.virtual_memory()
    print("=" * 70)
    print("BENCH MINI — Benchmark representativo RPi4")
    print("=" * 70)
    print(f"Hardware: {platform.machine()} | {psutil.cpu_count()} cores | "
          f"RAM: {fmt_mb(mem.total)} MB total, {fmt_mb(mem.available)} MB libre")
    print(f"N_REPS={N_REPS} (rep 0=warmup) | STT configs: {len(STT_CONFIGS)} | TTS configs: {len(TTS_CONFIGS)}")
    temp_sys = leer_temperatura_cpu()
    if temp_sys:
        print(f"Temperatura CPU: {temp_sys}°C")
    throttle = leer_throttling_rpi()
    if throttle is not None:
        estado = "OK" if throttle == 0 else f"ALERTA 0x{throttle:x}"
        print(f"Throttling RPi4: {estado}")
    print()

    resultados_stt = []
    resultados_tts = []

    # --- STT ---
    if not args.skip_stt:
        print("=" * 50)
        print("STT BENCHMARKS")
        print("=" * 50)
        archivos_wav, ground_truths = resolver_archivos_stt(args.audios_dir or "", base_dir)
        if not archivos_wav:
            print("  [WARN] No se encontraron WAVs para STT. "
                  "Usa --audios-dir o coloca audios en audios/")
            print("  Para generar audio sintético: python3 bench_stt_exhaustivo.py "
                  "--generar-audio --piper-fallback")
        else:
            print(f"  Audio dir: {os.path.dirname(archivos_wav[0])} ({len(archivos_wav)} WAVs)")
            resultados_stt = benchmark_stt_mini(archivos_wav, ground_truths, models_dir)

    # --- TTS ---
    if not args.skip_tts:
        print("\n" + "=" * 50)
        print("TTS BENCHMARKS")
        print("=" * 50)
        resultados_tts = benchmark_tts_mini(FRASES_TTS, output_dir, models_dir)

        # WER TTS
        evaluar_calidad_tts(resultados_tts, no_quality=args.no_quality)

    # --- Tabla consola ---
    imprimir_tabla_consola_mini(resultados_stt, resultados_tts)

    # --- Gráficas ---
    print("=" * 50)
    print("GENERANDO GRÁFICAS")
    print("=" * 50)
    try:
        archivos_graficas = generar_graficas_mini(resultados_stt, resultados_tts, graficas_dir)
    except Exception as e:
        print(f"  [WARN] Gráficas: {e}")
        archivos_graficas = []

    # --- Informe MD ---
    try:
        generar_informe_mini(resultados_stt, resultados_tts, archivos_graficas, md_path)
    except Exception as e:
        print(f"  [WARN] Informe: {e}")

    # --- JSON ---
    salida = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": {
            "plataforma": platform.machine(),
            "sistema": platform.system(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_mb": fmt_mb(mem.total),
            "ram_libre_mb": fmt_mb(mem.available),
            "throttling_inicial": leer_throttling_rpi(),
            "temp_inicial_c": leer_temperatura_cpu(),
        },
        "parametros": {
            "n_reps": N_REPS,
            "skip_stt": args.skip_stt,
            "skip_tts": args.skip_tts,
            "no_quality": args.no_quality,
        },
        "stt": [r for r in resultados_stt if r is not None],
        "tts": [r for r in resultados_tts if r is not None],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    # --- CSV ---
    try:
        exportar_csv_mini(resultados_stt, resultados_tts, csv_path)
    except Exception as e:
        print(f"  [WARN] CSV: {e}")

    print("\nBench mini completado.")


if __name__ == "__main__":
    main()
