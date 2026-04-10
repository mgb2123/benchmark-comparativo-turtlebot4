#!/usr/bin/env python3
"""Benchmark TTS exhaustivo en RPi4: Piper vs Coqui TTS.

Varía parámetros clave: length_scale y noise_scale (Piper), modelos (Coqui).
Mide tiempo de síntesis, RTF, RAM, CPU y genera WAVs para comparar calidad.

Metodología:
- N_REPS=3 repeticiones por frase; la primera se descarta como warmup.
- Se reporta media ± desviación estándar por métrica.

Uso:
    python3 bench_tts_exhaustivo.py
"""

import datetime
import gc
import json
import math
import os
import platform
import sys
import threading
import time
import wave

try:
    import psutil
except ImportError:
    print("[ERROR] psutil no instalado")
    sys.exit(1)

FRASES = [
    "Entendido, avanzando.",
    "Veo una persona y una silla.",
    "No entiendo, repite por favor.",
    "Navegando hacia la cocina.",
    "Estoy acoplada en la base.",
    "Hola, soy Ana, tu robot asistente.",
    "He detectado tres objetos.",
    "Giro completado.",
    "No puedo hacer eso ahora.",
    "Tiempo agotado, intenta de nuevo.",
]

N_REPS = 3  # repeticiones por frase (la 1ª se descarta como warmup)

# --- Configuraciones Piper ---
PIPER_CONFIGS = [
    {"length_scale": 0.7, "noise_scale": 0.667, "noise_w": 0.8, "nombre": "piper_fast"},
    {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8, "nombre": "piper_default"},
    {"length_scale": 1.3, "noise_scale": 0.667, "noise_w": 0.8, "nombre": "piper_slow"},
    {"length_scale": 1.0, "noise_scale": 0.3,   "noise_w": 0.3, "nombre": "piper_lowvar"},
    {"length_scale": 1.0, "noise_scale": 1.0,   "noise_w": 1.0, "nombre": "piper_highvar"},
]

# --- Configuraciones Coqui TTS ---
COQUI_CONFIGS = [
    {"model": "tts_models/es/css10/vits", "nombre": "coqui_css10_vits",
     "descripcion": "VITS español (CSS10)"},
    {"model": "tts_models/multilingual/multi-dataset/xtts_v2", "nombre": "coqui_xtts_v2",
     "descripcion": "XTTS v2 multilingüe (~1.8 GB)"},
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


def duracion_wav(ruta):
    try:
        with wave.open(ruta, 'rb') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _std(valores):
    """Desviación estándar muestral (n-1). Retorna 0.0 si hay menos de 2 valores."""
    n = len(valores)
    if n < 2:
        return 0.0
    m = sum(valores) / n
    return round(math.sqrt(sum((v - m) ** 2 for v in valores) / (n - 1)), 3)


def benchmark_piper_config(config, frases, output_dir, models_dir):
    """Benchmark de una configuración Piper con N_REPS repeticiones (1ª=warmup)."""
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig
    except ImportError:
        print("  [SKIP] piper-tts no instalado")
        return None

    piper_dir = os.path.join(models_dir, "piper")
    modelo_onnx = None

    if os.path.isdir(piper_dir):
        for f in os.listdir(piper_dir):
            if f.endswith(".onnx") and not f.endswith(".onnx.json"):
                modelo_onnx = os.path.join(piper_dir, f)
                break

    if not modelo_onnx:
        print(f"  [SKIP] Modelo Piper no encontrado en {piper_dir}")
        return None

    nombre = config["nombre"]
    print(f"\n  --- {nombre} (len={config['length_scale']}, noise={config['noise_scale']}, w={config['noise_w']}) ---")

    resultado = {
        "motor": "piper",
        "nombre": nombre,
        "config": config.copy(),
        "modelo": os.path.basename(modelo_onnx),
        "n_reps": N_REPS,
        "nota_warmup": "La primera repetición se descarta como warmup.",
    }

    monitor = MonitorRAM()
    monitor.iniciar()
    proc = psutil.Process()
    proc.cpu_percent()

    t0 = time.perf_counter()
    voice = PiperVoice.load(modelo_onnx)
    t_carga = time.perf_counter() - t0
    ram_carga = monitor.detener()

    resultado["tiempo_carga_s"] = round(t_carga, 2)
    resultado["ram_carga_mb"] = fmt_mb(ram_carga)
    print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

    resultado["frases"] = []
    subdir = os.path.join(output_dir, nombre)
    os.makedirs(subdir, exist_ok=True)

    syn_config = SynthesisConfig(
        length_scale=config["length_scale"],
        noise_scale=config["noise_scale"],
        noise_w_scale=config["noise_w"],
    )

    for i, frase in enumerate(frases):
        wav_path = os.path.join(subdir, f"frase_{i:02d}.wav")
        tiempos = []

        for rep in range(N_REPS):
            monitor = MonitorRAM()
            monitor.iniciar()
            proc.cpu_percent()

            t0 = time.perf_counter()
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(voice.config.sample_rate)
                for chunk in voice.synthesize(frase, syn_config=syn_config):
                    wf.writeframes(chunk.audio_int16_bytes)
            t_sint = time.perf_counter() - t0

            cpu_pct = proc.cpu_percent()
            ram_pico = monitor.detener()

            if rep == 0:
                warmup_t = t_sint  # warmup excluido del promedio
            else:
                tiempos.append(t_sint)

        dur = duracion_wav(wav_path)
        rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
        media_t = sum(tiempos) / len(tiempos) if tiempos else 0

        resultado["frases"].append({
            "frase": frase,
            "tiempo_sintesis_s": round(media_t, 3),
            "std_tiempo_s": _std(tiempos),
            "warmup_s": round(warmup_t, 3),
            "duracion_audio_s": round(dur, 2),
            "rtf": round(sum(rtfs) / len(rtfs), 3) if rtfs else 0,
            "std_rtf": _std(rtfs),
            "ram_pico_mb": fmt_mb(ram_pico),
            "cpu_pct": round(cpu_pct, 1),
        })
        print(f"    [{i+1}/{len(frases)}] \"{frase[:35]}\" | "
              f"{media_t:.2f}±{_std(tiempos):.3f}s | dur={dur:.1f}s | "
              f"RTF={sum(rtfs)/len(rtfs):.2f}")

    datos = resultado["frases"]
    if datos:
        resultado["promedios"] = {
            "tiempo_sintesis_s": round(sum(d["tiempo_sintesis_s"] for d in datos) / len(datos), 3),
            "std_tiempo_s": round(sum(d["std_tiempo_s"] for d in datos) / len(datos), 3),
            "duracion_audio_s": round(sum(d["duracion_audio_s"] for d in datos) / len(datos), 2),
            "rtf": round(sum(d["rtf"] for d in datos) / len(datos), 3),
            "std_rtf": round(sum(d["std_rtf"] for d in datos) / len(datos), 3),
            "ram_pico_mb": max(d["ram_pico_mb"] for d in datos),
            "cpu_pct": round(sum(d["cpu_pct"] for d in datos) / len(datos), 1),
        }

    del voice
    gc.collect()
    time.sleep(1)
    return resultado


def benchmark_coqui_config(config, frases, output_dir):
    """Benchmark de una configuración Coqui TTS con N_REPS repeticiones (1ª=warmup)."""
    try:
        from TTS.api import TTS
    except ImportError:
        print("  [SKIP] Coqui TTS no instalado. Instala con: pip install TTS")
        return None

    nombre = config["nombre"]
    print(f"\n  --- {nombre} ({config.get('descripcion', config['model'])}) ---")

    # Verificar RAM antes de cargar XTTS v2 (~1.8 GB)
    ram_libre = psutil.virtual_memory().available / (1024 * 1024)
    if "xtts" in config["model"].lower() and ram_libre < 2500:
        print(f"  [SKIP] RAM insuficiente para XTTS v2: {ram_libre:.0f} MB libre (necesita ~2500 MB)")
        return None

    resultado = {
        "motor": "coqui",
        "nombre": nombre,
        "config": config.copy(),
        "n_reps": N_REPS,
        "nota_warmup": "La primera repetición se descarta como warmup.",
    }

    monitor = MonitorRAM()
    monitor.iniciar()
    proc = psutil.Process()
    proc.cpu_percent()

    t0 = time.perf_counter()
    try:
        tts = TTS(model_name=config["model"], progress_bar=False, gpu=False)
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

    resultado["frases"] = []
    subdir = os.path.join(output_dir, nombre)
    os.makedirs(subdir, exist_ok=True)

    for i, frase in enumerate(frases):
        wav_path = os.path.join(subdir, f"frase_{i:02d}.wav")
        tiempos = []

        for rep in range(N_REPS):
            monitor = MonitorRAM()
            monitor.iniciar()
            proc.cpu_percent()

            t0 = time.perf_counter()
            try:
                if "multilingual" in config["model"] or "xtts" in config["model"].lower():
                    tts.tts_to_file(text=frase, file_path=wav_path, language="es")
                else:
                    tts.tts_to_file(text=frase, file_path=wav_path)
            except Exception as e:
                monitor.detener()
                print(f"    [ERROR rep {rep}]: {e}")
                break
            t_sint = time.perf_counter() - t0

            cpu_pct = proc.cpu_percent()
            ram_pico = monitor.detener()

            if rep == 0:
                warmup_t = t_sint
            else:
                tiempos.append(t_sint)

        if not tiempos:
            continue

        dur = duracion_wav(wav_path)
        rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
        media_t = sum(tiempos) / len(tiempos)

        resultado["frases"].append({
            "frase": frase,
            "tiempo_sintesis_s": round(media_t, 3),
            "std_tiempo_s": _std(tiempos),
            "warmup_s": round(warmup_t, 3),
            "duracion_audio_s": round(dur, 2),
            "rtf": round(sum(rtfs) / len(rtfs), 3) if rtfs else 0,
            "std_rtf": _std(rtfs),
            "ram_pico_mb": fmt_mb(ram_pico),
            "cpu_pct": round(cpu_pct, 1),
        })
        print(f"    [{i+1}/{len(frases)}] \"{frase[:35]}\" | "
              f"{media_t:.2f}±{_std(tiempos):.3f}s | dur={dur:.1f}s | "
              f"RTF={sum(rtfs)/len(rtfs):.2f}")

    datos = resultado["frases"]
    if datos:
        resultado["promedios"] = {
            "tiempo_sintesis_s": round(sum(d["tiempo_sintesis_s"] for d in datos) / len(datos), 3),
            "std_tiempo_s": round(sum(d["std_tiempo_s"] for d in datos) / len(datos), 3),
            "duracion_audio_s": round(sum(d["duracion_audio_s"] for d in datos) / len(datos), 2),
            "rtf": round(sum(d["rtf"] for d in datos) / len(datos), 3),
            "std_rtf": round(sum(d["std_rtf"] for d in datos) / len(datos), 3),
            "ram_pico_mb": max(d["ram_pico_mb"] for d in datos),
            "cpu_pct": round(sum(d["cpu_pct"] for d in datos) / len(datos), 1),
        }

    del tts
    gc.collect()
    time.sleep(1)
    return resultado


def imprimir_tabla(resultados):
    print("\n" + "=" * 115)
    print("RESUMEN COMPARATIVO TTS — ANÁLISIS EXHAUSTIVO")
    print("=" * 115)

    encabezados = ["Config", "Motor", "Carga(s)", "RAM(MB)", "Síntesis(s)", "±std", "Dur.Audio(s)", "RTF", "CPU%"]
    filas = []

    for r in resultados:
        if r is None:
            continue
        p = r.get("promedios", {})
        filas.append([
            r["nombre"][:30],
            r.get("motor", "?"),
            r.get("tiempo_carga_s", "N/A"),
            p.get("ram_pico_mb", r.get("ram_carga_mb", "N/A")),
            p.get("tiempo_sintesis_s", "N/A"),
            p.get("std_tiempo_s", "N/A"),
            p.get("duracion_audio_s", "N/A"),
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

    # Separar por motor para análisis de parámetros
    piper_results = [r for r in resultados if r and r.get("motor") == "piper"]
    coqui_results = [r for r in resultados if r and r.get("motor") == "coqui"]

    if piper_results:
        print("\n--- Impacto de parámetros en Piper ---")
        print(f"{'Config':<20} | {'len_scale':>9} | {'noise':>6} | {'noise_w':>7} | "
              f"{'Síntesis(s)':>11} | {'±std':>7} | {'Dur(s)':>6} | {'RTF':>6}")
        print("-" * 100)
        for r in piper_results:
            c = r["config"]
            p = r.get("promedios", {})
            print(f"{r['nombre']:<20} | {c['length_scale']:>9} | {c['noise_scale']:>6} | "
                  f"{c['noise_w']:>7} | {p.get('tiempo_sintesis_s','?'):>11} | "
                  f"{p.get('std_tiempo_s','?'):>7} | {p.get('duracion_audio_s','?'):>6} | "
                  f"{p.get('rtf','?'):>6}")

    if coqui_results:
        print("\n--- Coqui TTS ---")
        print(f"{'Config':<25} | {'Modelo':>30} | {'Síntesis(s)':>11} | {'±std':>7} | {'RTF':>6}")
        print("-" * 90)
        for r in coqui_results:
            c = r["config"]
            p = r.get("promedios", {})
            print(f"{r['nombre']:<25} | {c['model'][-30:]:>30} | "
                  f"{p.get('tiempo_sintesis_s','?'):>11} | {p.get('std_tiempo_s','?'):>7} | "
                  f"{p.get('rtf','?'):>6}")

    print()
    print(f"N_REPS={N_REPS} | RTF = Real-Time Factor (< 1.0 = más rápido que tiempo real)")
    print("WAVs en resultados/<config>/frase_*.wav para comparar calidad auditiva.")
    print()


def main():
    models_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models")
    )
    output = os.path.join(os.path.dirname(__file__), "resultados", "bench_tts_exhaustivo.json")
    output_dir = os.path.join(os.path.dirname(__file__), "resultados")

    print("=" * 70)
    print("BENCHMARK TTS EXHAUSTIVO — RPi4")
    print("=" * 70)
    mem = psutil.virtual_memory()
    print(f"Hardware: {platform.machine()} | {psutil.cpu_count()} cores | "
          f"RAM: {fmt_mb(mem.total)} MB total, {fmt_mb(mem.available)} MB libre")
    print(f"Frases: {len(FRASES)} | N_REPS: {N_REPS} (1ª=warmup) | "
          f"Configs Piper: {len(PIPER_CONFIGS)} | Configs Coqui: {len(COQUI_CONFIGS)}")
    print()

    resultados = []

    # Piper con todas las configuraciones
    print("=" * 50)
    print("PIPER — Variación de length_scale/noise")
    print("=" * 50)

    for config in PIPER_CONFIGS:
        try:
            r = benchmark_piper_config(config, FRASES, output_dir, models_dir)
            resultados.append(r)
        except Exception as e:
            print(f"  [ERROR] {config['nombre']}: {e}")

    # Coqui TTS con todas las configuraciones
    print("\n" + "=" * 50)
    print("COQUI TTS — Comparativa de modelos")
    print("=" * 50)

    for config in COQUI_CONFIGS:
        try:
            r = benchmark_coqui_config(config, FRASES, output_dir)
            resultados.append(r)
        except Exception as e:
            print(f"  [ERROR] {config['nombre']}: {e}")

    imprimir_tabla(resultados)

    # Guardar JSON
    salida = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": {
            "plataforma": platform.machine(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_mb": fmt_mb(mem.total),
        },
        "n_reps": N_REPS,
        "num_frases": len(FRASES),
        "resultados": [r for r in resultados if r is not None],
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en: {output}")


if __name__ == "__main__":
    main()
