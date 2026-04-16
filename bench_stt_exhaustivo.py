#!/usr/bin/env python3
"""Benchmark STT exhaustivo en RPi4: faster-whisper vs Vosk.

Varía parámetros clave de faster-whisper (modelo, beam_size, compute_type)
y compara contra Vosk. Mide WER, latencia, RTF, RAM y CPU.

Metodología:
- N_REPS=3 repeticiones por clip; la primera se descarta como warmup.
- Se reporta media ± desviación estándar por métrica.
- WER: se eliminan diacríticos antes de comparar (ej. "que"="qué"). Documentado
  para transparencia, puede inflar/deflactar WER según el motor.
- Audio: idealmente voz humana real (ver README). Usar --piper-fallback solo
  como alternativa cuando no sea posible grabar audio real.

Uso:
    python3 bench_stt_exhaustivo.py
    python3 bench_stt_exhaustivo.py --generar-audio          # instrucciones grabación real
    python3 bench_stt_exhaustivo.py --generar-audio --piper-fallback  # genera con Piper
"""

import argparse
import datetime
import gc
import json
import math
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

try:
    import matplotlib
    matplotlib.use("Agg")  # backend headless-safe (RPi4 sin pantalla)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# --- Ground truths ---
# 9 frases organizadas en 3 niveles de longitud para el benchmark STT.
# Cortas (frase00-02): 3-5 palabras — latencia mínima, vocabulario básico.
# Medianas (frase03-05): 8-12 palabras — frase completa con contexto semántico.
# Largas (frase06-08): 20-30 palabras — par de frases encadenadas, prueba de degradación.
FRASES = [
    # --- Cortas ---
    "para el robot",
    "gira a la derecha",
    "vuelve a la base",
    # --- Medianas ---
    "navega hasta la cocina y espera mis instrucciones allí",
    "busca el objeto rojo que está encima de la mesa",
    "toma una foto del pasillo y guárdala en memoria",
    # --- Largas ---
    "avanza hasta el salón, gira noventa grados a la derecha y para cuando llegues a la pared del fondo",
    "localiza a la persona que está en la habitación y cuando la encuentres emite una señal sonora para avisarme",
    "detecta si hay obstáculos en el pasillo y si los hay traza una ruta alternativa hacia el destino principal",
]

SAMPLE_RATE = 16000
N_REPS = 3  # repeticiones por clip (la 1ª se descarta como warmup)

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

# --- Espacios de parámetros para barrido paramétrico ---
SWEEP_BEAM_SIZES   = [1, 2, 3, 4, 5]
SWEEP_VAD_MS       = [200, 350, 500, 750, 1000]
SWEEP_CHUNK_FRAMES = [500, 1000, 2000, 4000, 8000]
SWEEP_TEMPERATURES = [0.0, 0.2]

# Pesos por defecto para score compuesto (α=precisión, β=velocidad, γ=RAM)
# Ajustados para asistente de voz robótico: velocidad y precisión igual de importantes
DEFAULT_WEIGHTS = (0.4, 0.4, 0.2)


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
    # NOTA: se eliminan diacríticos (NFD + filtro Mn), por lo que "que"="qué".
    # Esto es una decisión explícita para hacer la comparación más robusta,
    # pero puede inflar o deflactar el WER según cómo transcriba cada motor.
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


def _std(valores):
    """Desviación estándar muestral (n-1). Retorna 0 si hay menos de 2 valores."""
    n = len(valores)
    if n < 2:
        return 0.0
    m = sum(valores) / n
    return round(math.sqrt(sum((v - m) ** 2 for v in valores) / (n - 1)), 3)


def wilson_interval(k, n, z=1.96):
    """Intervalo de confianza de Wilson al 95% para una proporción.

    k = número de 'éxitos' (errores de palabras en WER)
    n = total de palabras de referencia
    Devuelve (low, high) como proporciones en [0, 1].
    Recomendado para n pequeño (supera al intervalo normal de Wald).
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    center = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def calcular_ranking(resultados_validos, pesos=DEFAULT_WEIGHTS, frases=None):
    """Calcula un ranking objetivo de configuraciones STT.

    Devuelve un dict con:
      - ranking: lista ordenada por score compuesto
      - veredicto: string legible con el mejor modelo según distintos criterios
      - matriz_wer_por_frase: {frase_idx: {nombre: wer}} para heatmap
      - pesos_usados: dict con alpha, beta, gamma
    """
    if not resultados_validos:
        return {"ranking": [], "veredicto": "Sin datos", "matriz_wer_por_frase": {}, "pesos_usados": {}}

    alpha, beta, gamma = pesos

    nombres = [r["nombre"] for r in resultados_validos]
    wers    = [r["promedios"]["wer"] for r in resultados_validos]
    rtfs    = [r["promedios"]["rtf"] for r in resultados_validos]
    rams    = [r["promedios"]["ram_pico_mb"] for r in resultados_validos]

    rtf_min, rtf_max = min(rtfs), max(rtfs)
    ram_min, ram_max = min(rams), max(rams)
    rtf_norm = [(v - rtf_min) / (rtf_max - rtf_min + 1e-9) for v in rtfs]
    ram_norm = [(v - ram_min) / (ram_max - ram_min + 1e-9) for v in rams]

    scores = [
        alpha * max(0.0, 1.0 - wer) + beta * (1.0 - rn) + gamma * (1.0 - mn)
        for wer, rn, mn in zip(wers, rtf_norm, ram_norm)
    ]

    # Intervalo de Wilson por config
    wer_ci = []
    for r in resultados_validos:
        clips = r.get("clips", [])
        total_ref = sum(len(normalizar(c["ground_truth"]).split()) for c in clips)
        total_err = sum(
            round(c["wer"] * len(normalizar(c["ground_truth"]).split()))
            for c in clips
        )
        wer_ci.append(wilson_interval(total_err, total_ref))

    # Matriz WER por frase (para heatmap)
    matriz = {}
    if frases:
        for fi in range(len(frases)):
            matriz[fi] = {}
    for r in resultados_validos:
        for fi, clip in enumerate(r.get("clips", [])):
            if fi not in matriz:
                matriz[fi] = {}
            matriz[fi][r["nombre"]] = clip["wer"]

    # Ranking ordenado por score descendente
    orden = sorted(range(len(nombres)), key=lambda i: scores[i], reverse=True)
    ranking = []
    for pos, i in enumerate(orden):
        ranking.append({
            "posicion": pos + 1,
            "nombre": nombres[i],
            "score": round(scores[i], 4),
            "wer": wers[i],
            "wer_ci_95": list(wer_ci[i]),
            "rtf": rtfs[i],
            "ram_mb": rams[i],
        })

    mejor_global    = nombres[orden[0]]
    mejor_velocidad = nombres[rtfs.index(min(rtfs))]
    mejor_precision = nombres[wers.index(min(wers))]

    veredicto = (
        f"Mejor global (score={scores[orden[0]]:.3f}): {mejor_global}\n"
        f"Mejor velocidad (RTF={min(rtfs):.3f}):       {mejor_velocidad}\n"
        f"Mejor precision (WER={min(wers):.3f}):       {mejor_precision}\n"
        f"Pesos usados: alpha={alpha} (precision), beta={beta} (velocidad), gamma={gamma} (RAM)"
    )

    return {
        "ranking": ranking,
        "veredicto": veredicto,
        "mejor_global": mejor_global,
        "mejor_velocidad": mejor_velocidad,
        "mejor_precision": mejor_precision,
        "matriz_wer_por_frase": {str(k): v for k, v in matriz.items()},
        "pesos_usados": {"alpha": alpha, "beta": beta, "gamma": gamma},
    }


def imprimir_ranking(ranking_data):
    """Imprime el ranking y el veredicto en stdout."""
    print("\n" + "=" * 100)
    print("RANKING POR EFICIENCIA COMPUESTA")
    print("=" * 100)
    encabezados = ["Pos", "Configuración", "Score", "WER", "CI 95% WER", "RTF", "RAM(MB)"]
    filas = []
    for r in ranking_data.get("ranking", []):
        ci = r["wer_ci_95"]
        filas.append([
            r["posicion"],
            r["nombre"][:42],
            f"{r['score']:.4f}",
            f"{r['wer']:.3f}",
            f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            f"{r['rtf']:.3f}",
            r["ram_mb"],
        ])
    if filas:
        anchos = [max(len(str(f[i])) for f in [encabezados] + filas) for i in range(len(encabezados))]
        fmt = " | ".join(f"{{:<{a}}}" for a in anchos)
        print(fmt.format(*encabezados))
        print("-+-".join("-" * a for a in anchos))
        for fila in filas:
            print(fmt.format(*[str(c) for c in fila]))
    veredicto = ranking_data.get("veredicto", "")
    if veredicto:
        print()
        for linea in veredicto.splitlines():
            print(f"  {linea}")
    print()


def generar_audio_piper(frases, output_dir, models_dir):
    """Genera WAVs de test con Piper TTS como fallback sintético.
    Para mejores resultados, usar voz humana real (ver README).
    """
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig
    except ImportError:
        print("[ERROR] piper-tts no instalado. Instala con: pip install piper-tts")
        return False

    piper_dir = os.path.join(models_dir, "piper")
    modelo_onnx = None
    if os.path.isdir(piper_dir):
        for f in os.listdir(piper_dir):
            if f.endswith(".onnx") and not f.endswith(".onnx.json"):
                modelo_onnx = os.path.join(piper_dir, f)
                break

    if not modelo_onnx:
        print(f"[ERROR] Modelo Piper no encontrado en {piper_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    voice = PiperVoice.load(modelo_onnx)
    syn_config = SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8)

    print(f"Generando {len(frases)} WAVs con Piper (fallback sintético)...")
    for i, frase in enumerate(frases):
        wav_path = os.path.join(output_dir, f"frase_{i:02d}.wav")
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(voice.config.sample_rate)
            for chunk in voice.synthesize(frase, syn_config=syn_config):
                wf.writeframes(chunk.audio_int16_bytes)
        print(f"  [{i+1}/{len(frases)}] {wav_path}")

    del voice
    gc.collect()
    print("[AVISO] Audio generado con Piper (sintético). Los WER pueden no reflejar uso real.")
    return True


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

    # Transcribir — N_REPS repeticiones; la primera (warmup) se excluye del promedio
    resultado["clips"] = []

    for wav_path, truth in zip(archivos_wav, ground_truths):
        nombre_wav = os.path.basename(wav_path)
        audio = cargar_wav(wav_path)
        dur = duracion_wav(wav_path)

        tiempos = []
        for rep in range(N_REPS):
            monitor = MonitorRAM()
            monitor.iniciar()
            proc.cpu_percent()

            t0 = time.perf_counter()
            segmentos, _ = modelo.transcribe(
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

            if rep == 0:
                warmup_t = t_trans  # warmup excluido de promedios
            else:
                tiempos.append(t_trans)

        wer_val = calcular_wer(truth, texto)
        rtfs = [t / dur for t in tiempos] if dur > 0 else [0] * len(tiempos)
        media_t = sum(tiempos) / len(tiempos) if tiempos else 0

        resultado["clips"].append({
            "archivo": nombre_wav,
            "duracion_audio_s": round(dur, 2),
            "ground_truth": truth,
            "transcripcion": texto,
            "wer": round(wer_val, 3),
            "tiempo_s": round(media_t, 3),
            "std_tiempo_s": _std(tiempos),
            "rtf": round(sum(rtfs) / len(rtfs), 3) if rtfs else 0,
            "std_rtf": _std(rtfs),
            "warmup_s": round(warmup_t, 3),
            "ram_pico_mb": fmt_mb(ram_pico),
            "cpu_pct": round(cpu_pct, 1),
        })
        print(f"    {nombre_wav}: \"{texto[:45]}\" | WER={wer_val:.2f} | "
              f"{media_t:.2f}±{_std(tiempos):.3f}s | RTF={sum(rtfs)/len(rtfs):.2f}")

    # Promedios con std
    clips = resultado["clips"]
    if clips:
        resultado["promedios"] = {
            "wer": round(sum(c["wer"] for c in clips) / len(clips), 3),
            "tiempo_s": round(sum(c["tiempo_s"] for c in clips) / len(clips), 3),
            "std_tiempo_s": round(sum(c["std_tiempo_s"] for c in clips) / len(clips), 3),
            "rtf": round(sum(c["rtf"] for c in clips) / len(clips), 3),
            "std_rtf": round(sum(c["std_rtf"] for c in clips) / len(clips), 3),
            "ram_pico_mb": max(c["ram_pico_mb"] for c in clips),
            "cpu_pct": round(sum(c["cpu_pct"] for c in clips) / len(clips), 1),
        }

    # Ahora probar con VAD activado (solo beam_size=1 para no duplicar demasiado)
    if config["beam_size"] == 1:
        print(f"  [+VAD] Repitiendo con vad_filter=True ({N_REPS} reps, 1ª=warmup)...")
        resultado["clips_vad"] = []
        for wav_path, truth in zip(archivos_wav, ground_truths):
            audio = cargar_wav(wav_path)
            dur = duracion_wav(wav_path)
            tiempos_vad = []

            for rep in range(N_REPS):
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
                if rep > 0:
                    tiempos_vad.append(t_trans)

            wer_val = calcular_wer(truth, texto)
            media_t = sum(tiempos_vad) / len(tiempos_vad) if tiempos_vad else 0
            resultado["clips_vad"].append({
                "archivo": os.path.basename(wav_path),
                "transcripcion": texto,
                "wer": round(wer_val, 3),
                "tiempo_s": round(media_t, 3),
                "std_tiempo_s": _std(tiempos_vad),
                "rtf": round(media_t / dur, 3) if dur > 0 else 0,
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


def _resolver_vosk_dir(vosk_dir_candidato):
    """Devuelve una ruta ASCII-safe al modelo Vosk.

    Vosk usa una librería C que puede fallar con rutas no-ASCII en Windows
    (ej. el carácter ñ en 'Coruña'). Si la ruta no es ASCII comprobamos si
    existe ya una copia en una ruta simple; en caso contrario la copiamos.
    """
    if vosk_dir_candidato.isascii():
        return vosk_dir_candidato

    nombre_modelo = os.path.basename(vosk_dir_candidato)
    # Candidatos de ruta simple (sin caracteres especiales)
    candidatos = [
        os.path.join(r"C:\vosk-models", nombre_modelo),
        os.path.join(os.environ.get("TEMP", r"C:\Temp"), nombre_modelo),
        os.path.join(r"C:\\", nombre_modelo),
    ]
    for ruta in candidatos:
        if os.path.isdir(ruta):
            print(f"  [INFO] Ruta con caracteres especiales, usando copia en: {ruta}")
            return ruta

    # Copiar al primer destino disponible
    import shutil
    for ruta in candidatos:
        try:
            destino_padre = os.path.dirname(ruta)
            os.makedirs(destino_padre, exist_ok=True)
            print(f"  [INFO] Copiando modelo a ruta ASCII: {ruta} (puede tardar...)")
            shutil.copytree(vosk_dir_candidato, ruta)
            return ruta
        except Exception as e:
            print(f"  [WARN] No se pudo copiar a {ruta}: {e}")

    # Último recurso: usar la ruta original y esperar que funcione
    print(f"  [WARN] Usando ruta no-ASCII, puede fallar: {vosk_dir_candidato}")
    return vosk_dir_candidato


def _vosk_tag(nombre_modelo):
    """Devuelve 'small' o 'large' a partir del nombre de carpeta del modelo."""
    n = nombre_modelo.lower()
    if "small" in n:
        return "small"
    return "large"


def benchmark_vosk(archivos_wav, ground_truths, models_dir):
    """Benchmark de Vosk con múltiples modelos y múltiples chunk_sizes.

    Itera sobre TODOS los modelos Vosk españoles encontrados en models_dir
    (small, large…). Por cada modelo: carga UNA vez y prueba los 4 chunk_sizes
    sobre todos los clips. Devuelve lista con el mismo schema que faster-whisper.
    """
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError:
        print("  [SKIP] vosk no instalado")
        return []

    # Descubrir todos los modelos Vosk españoles disponibles
    vosk_modelos = []
    if os.path.isdir(models_dir):
        for nombre in sorted(os.listdir(models_dir)):
            if nombre.startswith("vosk-model") and "es" in nombre:
                ruta = os.path.join(models_dir, nombre)
                if os.path.isdir(ruta):
                    vosk_modelos.append((nombre, ruta))

    if not vosk_modelos:
        print(f"  [SKIP] Ningún modelo Vosk encontrado en {models_dir}")
        return []

    # chunk_sizes a comparar: 250 ms / 125 ms / 62 ms / 31 ms a 16 kHz
    VOSK_CHUNK_SIZES = [1000, 2000, 4000, 8000]

    resultados_vosk = []

    for modelo_nombre, vosk_dir_raw in vosk_modelos:
        tag = _vosk_tag(modelo_nombre)
        vosk_dir = _resolver_vosk_dir(vosk_dir_raw)

        print(f"\n  === Vosk {tag} ({modelo_nombre}) — {len(VOSK_CHUNK_SIZES)} chunk_sizes ===")

        monitor = MonitorRAM()
        monitor.iniciar()
        proc = psutil.Process()
        proc.cpu_percent()

        try:
            t0 = time.perf_counter()
            modelo = Model(vosk_dir)
            t_carga = time.perf_counter() - t0
        except Exception as e:
            print(f"  [ERROR] No se pudo cargar {modelo_nombre}: {e}")
            continue

        ram_carga = monitor.detener()
        print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

        for chunk_frames in VOSK_CHUNK_SIZES:
            nombre_cfg = f"vosk-{tag}-es_c{chunk_frames}"
            print(f"\n  --- {nombre_cfg} ---")

            resultado = {
                "motor": "vosk",
                "nombre": nombre_cfg,
                "config": {"modelo": modelo_nombre, "chunk_frames": chunk_frames, "tag": tag},
                "tiempo_carga_s": round(t_carga, 2),
                "ram_carga_mb": fmt_mb(ram_carga),
            }

            clips = []
            for wav_path, truth in zip(archivos_wav, ground_truths):
                nombre_wav = os.path.basename(wav_path)
                dur = duracion_wav(wav_path)
                tiempos = []

                for rep in range(N_REPS):
                    mon = MonitorRAM()
                    mon.iniciar()
                    proc.cpu_percent()

                    t0 = time.perf_counter()
                    rec = KaldiRecognizer(modelo, SAMPLE_RATE)
                    with wave.open(wav_path, 'rb') as wf:
                        while True:
                            data = wf.readframes(chunk_frames)
                            if not data:
                                break
                            rec.AcceptWaveform(data)
                    res = json.loads(rec.FinalResult())
                    texto = res.get("text", "")
                    t_trans = time.perf_counter() - t0

                    cpu_pct = proc.cpu_percent()
                    ram_pico = mon.detener()

                    if rep == 0:
                        warmup_t = t_trans
                    else:
                        tiempos.append(t_trans)

                wer_val = calcular_wer(truth, texto)
                rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
                media_t = sum(tiempos) / len(tiempos) if tiempos else 0.0

                clips.append({
                    "archivo": nombre_wav,
                    "duracion_audio_s": round(dur, 2),
                    "ground_truth": truth,
                    "transcripcion": texto,
                    "wer": round(wer_val, 3),
                    "tiempo_s": round(media_t, 3),
                    "std_tiempo_s": _std(tiempos),
                    "rtf": round(sum(rtfs) / len(rtfs), 3) if rtfs else 0,
                    "std_rtf": _std(rtfs),
                    "warmup_s": round(warmup_t, 3),
                    "ram_pico_mb": fmt_mb(ram_pico),
                    "cpu_pct": round(cpu_pct, 1),
                })
                print(f"    {nombre_wav}: \"{texto[:40]}\" | WER={wer_val:.2f} | "
                      f"{media_t:.2f}±{_std(tiempos):.3f}s | RTF={sum(rtfs)/len(rtfs):.2f}")

            resultado["clips"] = clips
            if clips:
                resultado["promedios"] = {
                    "wer": round(sum(c["wer"] for c in clips) / len(clips), 3),
                    "tiempo_s": round(sum(c["tiempo_s"] for c in clips) / len(clips), 3),
                    "std_tiempo_s": round(sum(c["std_tiempo_s"] for c in clips) / len(clips), 3),
                    "rtf": round(sum(c["rtf"] for c in clips) / len(clips), 3),
                    "std_rtf": round(sum(c["std_rtf"] for c in clips) / len(clips), 3),
                    "ram_pico_mb": max(c["ram_pico_mb"] for c in clips),
                    "cpu_pct": round(sum(c["cpu_pct"] for c in clips) / len(clips), 1),
                }

            resultados_vosk.append(resultado)

        del modelo
        gc.collect()

    return resultados_vosk


def benchmark_whisper_sweep_beams(archivos_wav, ground_truths, models_dir,
                                  beam_sizes=None, compute_types=None, temperatures=None):
    """Barrido de beam_size × temperatura. Carga el modelo UNA vez por (modelo, compute_type)."""
    from faster_whisper import WhisperModel

    beam_sizes    = beam_sizes    or SWEEP_BEAM_SIZES
    compute_types = compute_types or ["int8"]
    temperatures  = temperatures  or [None]
    resultados = []

    for modelo_name in ["tiny", "base"]:
        for compute_type in compute_types:
            whisper_dir = os.path.join(models_dir, "whisper", modelo_name) if models_dir else None
            model_path = whisper_dir if (whisper_dir and os.path.isdir(whisper_dir)) else modelo_name

            print(f"\n  [SWEEP] Cargando whisper-{modelo_name} {compute_type}...")
            try:
                modelo = WhisperModel(
                    model_path, device="cpu", compute_type=compute_type,
                    local_files_only=os.path.isdir(str(model_path)),
                )
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue

            for beam_size in beam_sizes:
                for temperature in temperatures:
                    temp_str = f"_temp{temperature}" if temperature is not None else ""
                    nombre = f"sweep-whisper-{modelo_name}_{compute_type}_beam{beam_size}{temp_str}"
                    print(f"    {nombre} ...", end="", flush=True)

                    clips = []
                    for wav_path, truth in zip(archivos_wav, ground_truths):
                        audio = cargar_wav(wav_path)
                        dur = duracion_wav(wav_path)
                        tiempos = []
                        texto = ""
                        for rep in range(N_REPS):
                            kwargs = dict(language="es", beam_size=beam_size, vad_filter=False)
                            if temperature is not None:
                                kwargs["temperature"] = temperature
                            t0 = time.perf_counter()
                            segs, _ = modelo.transcribe(audio, **kwargs)
                            texto = " ".join(s.text.strip() for s in segs)
                            t_trans = time.perf_counter() - t0
                            if rep > 0:
                                tiempos.append(t_trans)
                        wer_val = calcular_wer(truth, texto)
                        rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0]
                        media_t = sum(tiempos) / len(tiempos) if tiempos else 0.0
                        clips.append({
                            "archivo": os.path.basename(wav_path),
                            "duracion_audio_s": round(dur, 2),
                            "ground_truth": truth,
                            "transcripcion": texto,
                            "wer": round(wer_val, 3),
                            "tiempo_s": round(media_t, 3),
                            "std_tiempo_s": _std(tiempos),
                            "rtf": round(sum(rtfs) / len(rtfs), 3),
                            "std_rtf": _std(rtfs),
                        })

                    if clips:
                        promedios = {
                            "wer": round(sum(c["wer"] for c in clips) / len(clips), 3),
                            "tiempo_s": round(sum(c["tiempo_s"] for c in clips) / len(clips), 3),
                            "rtf": round(sum(c["rtf"] for c in clips) / len(clips), 3),
                            "ram_pico_mb": 0,  # no monitoreado en sweep para mayor velocidad
                            "cpu_pct": 0,
                        }
                        print(f" WER={promedios['wer']:.3f} RTF={promedios['rtf']:.3f}")
                    else:
                        promedios = {}
                        print(" sin clips")

                    resultados.append({
                        "motor": "faster-whisper",
                        "nombre": nombre,
                        "config": {
                            "modelo": modelo_name,
                            "compute_type": compute_type,
                            "beam_size": beam_size,
                            "temperature": temperature,
                        },
                        "clips": clips,
                        "promedios": promedios,
                    })

            del modelo
            gc.collect()

    return resultados


def benchmark_whisper_vad_sweep(archivos_wav, ground_truths, models_dir,
                                best_config, vad_ms_values=None):
    """Barrido de min_silence_duration_ms sobre el mejor config de whisper.
    Carga el modelo UNA sola vez.
    """
    from faster_whisper import WhisperModel

    vad_ms_values = vad_ms_values or SWEEP_VAD_MS
    modelo_name   = best_config.get("modelo", "tiny")
    compute_type  = best_config.get("compute_type", "int8")
    beam_size     = best_config.get("beam_size", 1)

    whisper_dir = os.path.join(models_dir, "whisper", modelo_name) if models_dir else None
    model_path  = whisper_dir if (whisper_dir and os.path.isdir(whisper_dir)) else modelo_name

    print(f"\n  [VAD SWEEP] Cargando whisper-{modelo_name} {compute_type} beam{beam_size}...")
    try:
        modelo = WhisperModel(
            model_path, device="cpu", compute_type=compute_type,
            local_files_only=os.path.isdir(str(model_path)),
        )
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []

    resultados = []
    for ms in vad_ms_values:
        nombre = f"vad_sweep_{modelo_name}_{compute_type}_beam{beam_size}_ms{ms}"
        print(f"    {nombre} ...", end="", flush=True)
        clips = []
        for wav_path, truth in zip(archivos_wav, ground_truths):
            audio = cargar_wav(wav_path)
            dur = duracion_wav(wav_path)
            tiempos = []
            texto = ""
            for rep in range(N_REPS):
                t0 = time.perf_counter()
                segs, _ = modelo.transcribe(
                    audio, language="es", beam_size=beam_size,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": ms},
                )
                texto = " ".join(s.text.strip() for s in segs)
                t_trans = time.perf_counter() - t0
                if rep > 0:
                    tiempos.append(t_trans)
            wer_val = calcular_wer(truth, texto)
            rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0]
            media_t = sum(tiempos) / len(tiempos) if tiempos else 0.0
            clips.append({
                "archivo": os.path.basename(wav_path),
                "ground_truth": truth,
                "transcripcion": texto,
                "wer": round(wer_val, 3),
                "tiempo_s": round(media_t, 3),
                "rtf": round(sum(rtfs) / len(rtfs), 3),
            })
        promedios = {
            "wer": round(sum(c["wer"] for c in clips) / len(clips), 3),
            "tiempo_s": round(sum(c["tiempo_s"] for c in clips) / len(clips), 3),
            "rtf": round(sum(c["rtf"] for c in clips) / len(clips), 3),
        }
        print(f" WER={promedios['wer']:.3f} RTF={promedios['rtf']:.3f}")
        resultados.append({
            "nombre": nombre,
            "vad_ms": ms,
            "clips": clips,
            "promedios": promedios,
        })

    del modelo
    gc.collect()
    return resultados


def benchmark_vosk_chunk_sweep(archivos_wav, ground_truths, models_dir, chunk_sizes=None):
    """Barrido de chunk_size sobre TODOS los clips. Carga Vosk UNA sola vez."""
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError:
        print("  [SKIP] vosk no instalado (chunk sweep)")
        return {}

    chunk_sizes = chunk_sizes or SWEEP_CHUNK_FRAMES

    VOSK_FALLBACK_DIRS = [
        os.path.join(os.environ.get("USERPROFILE", ""), "vosk-model-small-es-0.42"),
        r"C:\vosk-model-small-es-0.42",
    ]
    vosk_dir = None
    if os.path.isdir(models_dir):
        for nombre in os.listdir(models_dir):
            if nombre.startswith("vosk-model") and "es" in nombre:
                vosk_dir = os.path.join(models_dir, nombre)
                break
    if vosk_dir and not vosk_dir.isascii():
        for fb in VOSK_FALLBACK_DIRS:
            if os.path.isdir(fb):
                vosk_dir = fb
                break
    if not vosk_dir or not os.path.isdir(vosk_dir):
        print("  [SKIP] Modelo Vosk no encontrado (chunk sweep)")
        return {}

    print(f"\n  [CHUNK SWEEP] Cargando Vosk...")
    modelo = Model(vosk_dir)
    resultados = {}

    for chunk_frames in chunk_sizes:
        print(f"    chunk={chunk_frames} ...", end="", flush=True)
        clips = []
        for wav_path, truth in zip(archivos_wav, ground_truths):
            dur = duracion_wav(wav_path)
            t0 = time.perf_counter()
            rec = KaldiRecognizer(modelo, SAMPLE_RATE)
            with wave.open(wav_path, 'rb') as wf:
                while True:
                    data = wf.readframes(chunk_frames)
                    if not data:
                        break
                    rec.AcceptWaveform(data)
            res = json.loads(rec.FinalResult())
            texto = res.get("text", "")
            t_trans = time.perf_counter() - t0
            wer_val = calcular_wer(truth, texto)
            clips.append({
                "archivo": os.path.basename(wav_path),
                "wer": round(wer_val, 3),
                "tiempo_s": round(t_trans, 3),
                "rtf": round(t_trans / dur, 3) if dur > 0 else 0.0,
            })
        promedios = {
            "wer": round(sum(c["wer"] for c in clips) / len(clips), 3),
            "tiempo_s": round(sum(c["tiempo_s"] for c in clips) / len(clips), 3),
            "rtf": round(sum(c["rtf"] for c in clips) / len(clips), 3),
        }
        print(f" WER={promedios['wer']:.3f} RTF={promedios['rtf']:.3f}")
        resultados[str(chunk_frames)] = {"clips": clips, "promedios": promedios}

    del modelo
    gc.collect()
    return resultados


# ---------------------------------------------------------------------------
# VISUALIZACIONES
# ---------------------------------------------------------------------------

def _preparar_datos_graficas(datos, ranking_data=None):
    """Extrae estructura limpia de datos para las funciones de gráficas."""
    resultados = [r for r in datos.get("resultados", []) if r and "promedios" in r]
    ranking_list = (ranking_data or {}).get("ranking", [])
    ci_map = {r["nombre"]: r["wer_ci_95"] for r in ranking_list}

    configs = []
    for r in resultados:
        p = r["promedios"]
        ci = ci_map.get(r["nombre"], [p["wer"], p["wer"]])
        configs.append({
            "nombre": r["nombre"],
            "motor": r.get("motor", "desconocido"),
            "modelo": r.get("config", {}).get("modelo", ""),
            "compute_type": r.get("config", {}).get("compute_type", ""),
            "beam_size": r.get("config", {}).get("beam_size", 0),
            "wer": p["wer"],
            "wer_ci_low": ci[0],
            "wer_ci_high": ci[1],
            "rtf": p["rtf"],
            "std_rtf": p.get("std_rtf", 0),
            "ram_pico_mb": p.get("ram_pico_mb", 0),
            "ram_carga_mb": r.get("ram_carga_mb", 0),
            "cpu_pct": p.get("cpu_pct", 0),
        })

    clips_matrix = []
    for r in resultados:
        for fi, clip in enumerate(r.get("clips", [])):
            clips_matrix.append({
                "nombre": r["nombre"],
                "frase_idx": fi,
                "wer": clip["wer"],
                "tiempo_s": clip["tiempo_s"],
                "rtf": clip["rtf"],
            })

    vad_pairs = []
    for r in resultados:
        if "promedios_vad" in r:
            vad_pairs.append({
                "nombre": r["nombre"],
                "wer_novad": r["promedios"]["wer"],
                "wer_vad": r["promedios_vad"]["wer"],
                "rtf_novad": r["promedios"]["rtf"],
                "rtf_vad": r["promedios_vad"]["rtf"],
            })

    sweep_beams  = datos.get("sweep_beams", [])
    sweep_vad    = datos.get("sweep_vad", [])
    chunks_sweep = datos.get("chunks_sweep", None)

    return {
        "configs": configs,
        "clips_matrix": clips_matrix,
        "vad_pairs": vad_pairs,
        "sweep_beams": sweep_beams,
        "sweep_vad": sweep_vad,
        "chunks_sweep": chunks_sweep,
        "ranking": ranking_list,
        "pesos": (ranking_data or {}).get("pesos_usados", {}),
    }


def _etiqueta(nombre):
    """Etiqueta descriptiva y corta para ejes de gráficas."""
    n = nombre

    # Vosk: vosk-small-es_c4000  →  Vosk c=4000
    if n.startswith("vosk"):
        if "_c" in n:
            chunk = n.rsplit("_c", 1)[-1]
            return f"Vosk c={chunk}"
        return "Vosk small-es"

    # VAD sweep: vad_sweep_tiny_int8_beam1_ms500  →  VAD 500ms
    if n.startswith("vad_sweep"):
        ms = n.split("_ms")[-1] if "_ms" in n else "?"
        return f"VAD {ms}ms"

    # faster-whisper (benchmark principal y sweep):
    # whisper-tiny_int8_beam1_best1  →  Tiny int8 b=1
    # sweep-whisper-base_int8_beam3_temp0.2  →  Base int8 b=3 T=0.2
    n = n.replace("sweep-whisper-", "").replace("whisper-", "")
    partes = n.split("_")
    modelo   = partes[0].capitalize() if partes else "?"
    cuant    = "int8" if "int8" in n else ("f32" if "float32" in n else "")
    beam     = next((p.replace("beam", "b=") for p in partes if p.startswith("beam")), "")
    temp_str = next((f" T={p.replace('temp','')}" for p in partes if p.startswith("temp")), "")
    partes_label = [x for x in [modelo, cuant, beam] if x]
    return " ".join(partes_label) + temp_str


def _grafica_wer_barras(d, out):
    if not d["configs"]:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    configs_s = sorted(d["configs"], key=lambda x: x["wer"], reverse=True)
    etiquetas = [_etiqueta(c["nombre"]) for c in configs_s]
    wers = [c["wer"] for c in configs_s]
    yerr_low  = [max(0.0, c["wer"] - c["wer_ci_low"])  for c in configs_s]
    yerr_high = [max(0.0, c["wer_ci_high"] - c["wer"]) for c in configs_s]
    colores = ["darkorange" if c["motor"] == "vosk" else "steelblue" for c in configs_s]
    bars = ax.bar(range(len(etiquetas)), wers, color=colores, edgecolor="white")
    ax.errorbar(range(len(etiquetas)), wers,
                yerr=[yerr_low, yerr_high],
                fmt="none", color="black", capsize=4, linewidth=1.5, label="IC Wilson 95%")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("WER (Word Error Rate)")
    ax.set_title("Comparación WER — todos los modelos (peor → mejor)")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles_extra = [
        Patch(color="steelblue", label="faster-whisper"),
        Patch(color="darkorange", label="Vosk"),
        Line2D([0],[0], color="black", marker="|", markersize=8, label="IC Wilson 95%"),
    ]
    ax.legend(handles=handles_extra)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "01_wer_comparacion.png"), dpi=150)
    plt.close(fig)


def _grafica_rtf_barras(d, out):
    if not d["configs"]:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    configs_s = sorted(d["configs"], key=lambda x: x["rtf"], reverse=True)
    etiquetas = [_etiqueta(c["nombre"]) for c in configs_s]
    rtfs = [c["rtf"] for c in configs_s]
    colores = []
    for c in configs_s:
        if c["rtf"] < 0.2:
            colores.append("#2ca02c")   # verde
        elif c["rtf"] < 0.5:
            colores.append("#ff7f0e")   # naranja
        else:
            colores.append("#d62728")   # rojo
    ax.bar(range(len(etiquetas)), rtfs, color=colores, edgecolor="white")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="RTF = 1.0 (tiempo real)")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RTF (Real-Time Factor)")
    ax.set_title("RTF por config — por debajo de 1.0 = más rápido que tiempo real")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02_rtf_comparacion.png"), dpi=150)
    plt.close(fig)


def _grafica_wer_rtf_scatter(d, out):
    if not d["configs"]:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in d["configs"]:
        color = "darkorange" if c["motor"] == "vosk" else "steelblue"
        ax.scatter(c["rtf"], c["wer"], color=color, s=90, zorder=3)
        ax.annotate(_etiqueta(c["nombre"])[:22],
                    (c["rtf"], c["wer"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    # Frontera Pareto
    pareto = []
    for c in d["configs"]:
        dominado = any(
            (o["wer"] <= c["wer"] and o["rtf"] <= c["rtf"] and
             (o["wer"] < c["wer"] or o["rtf"] < c["rtf"]))
            for o in d["configs"] if o is not c
        )
        if not dominado:
            pareto.append(c)
    if len(pareto) > 1:
        pareto_s = sorted(pareto, key=lambda x: x["rtf"])
        ax.plot([p["rtf"] for p in pareto_s], [p["wer"] for p in pareto_s],
                "r--", linewidth=1.5, label="Frontera Pareto", zorder=2)
    ax.set_xlabel("RTF (menor = más rápido)")
    ax.set_ylabel("WER (menor = más preciso)")
    ax.set_title("Análisis Pareto: WER vs RTF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03_wer_rtf_scatter.png"), dpi=150)
    plt.close(fig)


def _grafica_latencia_boxplot(d, out):
    if not d["clips_matrix"]:
        return
    grupos = {}
    for clip in d["clips_matrix"]:
        grupos.setdefault(clip["nombre"], []).append(clip["tiempo_s"])
    nombres_s = sorted(grupos.keys(), key=lambda n: sum(grupos[n]) / len(grupos[n]))
    data = [grupos[n] for n in nombres_s]
    etiquetas = [_etiqueta(n) for n in nombres_s]
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data, tick_labels=etiquetas, patch_artist=True, vert=True,
                    medianprops=dict(color="red", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_ylabel("Latencia por clip (s)")
    ax.set_title("Distribución de latencia de inferencia por configuración (10 clips)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "04_latencia_boxplot.png"), dpi=150)
    plt.close(fig)


def _grafica_ram_barras(d, out):
    if not d["configs"]:
        return
    configs_s = sorted(d["configs"], key=lambda x: x["ram_pico_mb"])
    etiquetas = [_etiqueta(c["nombre"]) for c in configs_s]
    ram_pico  = [c["ram_pico_mb"] for c in configs_s]
    ram_carga = [c["ram_carga_mb"] for c in configs_s]
    x = range(len(etiquetas))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar([xi - w/2 for xi in x], ram_carga, w, label="RAM carga modelo", color="steelblue")
    ax.bar([xi + w/2 for xi in x], ram_pico,  w, label="RAM pico inferencia", color="coral")
    ax.axhline(4096, color="red", linestyle="--", linewidth=1.5, label="Límite RPi4 (4 GB)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RAM (MB)")
    ax.set_title("Uso de RAM por configuración")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "05_ram_uso.png"), dpi=150)
    plt.close(fig)


def _grafica_beam_size_linea(d, out):
    # Usar sweep si disponible, si no usar configs base
    fuente = d.get("sweep_beams") or []
    if not fuente:
        fuente = [c for c in d["configs"] if c["motor"] == "faster-whisper"]
    if not fuente:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, modelo_n in zip(axes, ["tiny", "base"]):
        for ct, color, ls in [("int8", "steelblue", "-o"), ("float32", "darkorange", "-s")]:
            subset = []
            for r in fuente:
                cfg = r.get("config", {}) if isinstance(r, dict) else {}
                bs = cfg.get("beam_size") or r.get("beam_size")
                mn = cfg.get("modelo") or r.get("modelo")
                ctype = cfg.get("compute_type") or r.get("compute_type")
                # fuente puede ser sweep_beams (tiene "promedios") o d["configs"] (tiene "wer" directo)
                wer_v = (r.get("promedios") or {}).get("wer") or r.get("wer")
                if mn == modelo_n and ctype == ct and bs is not None and wer_v is not None and wer_v > 0:
                    subset.append((bs, wer_v))
            if subset:
                subset.sort()
                ax.plot([s[0] for s in subset], [s[1] for s in subset],
                        ls, label=ct, color=color)
        ax.set_title(f"whisper-{modelo_n}")
        ax.set_xlabel("beam_size")
        ax.set_ylabel("WER")
        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    fig.suptitle("Efecto del beam_size sobre WER")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "06_beam_size_efecto.png"), dpi=150)
    plt.close(fig)


def _grafica_int8_vs_float32(d, out):
    whisper_cfgs = [c for c in d["configs"] if c["motor"] == "faster-whisper"]
    if not whisper_cfgs:
        return

    def mean_metric(modelo_n, ct, metrica):
        vals = [c[metrica] for c in whisper_cfgs
                if c["modelo"] == modelo_n and c["compute_type"] == ct]
        return round(sum(vals) / len(vals), 3) if vals else 0

    modelos = ["tiny", "base"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    x = range(len(modelos))
    w = 0.35
    for ax, metrica, titulo in [(ax1, "wer", "WER"), (ax2, "rtf", "RTF")]:
        int8_v = [mean_metric(m, "int8", metrica) for m in modelos]
        f32_v  = [mean_metric(m, "float32", metrica) for m in modelos]
        ax.bar([xi - w/2 for xi in x], int8_v,  w, label="int8",    color="steelblue")
        ax.bar([xi + w/2 for xi in x], f32_v,   w, label="float32", color="coral")
        ax.set_xticks(list(x))
        ax.set_xticklabels(modelos)
        ax.set_ylabel(titulo)
        ax.set_title(f"{titulo}: int8 vs float32 (media sobre beam_sizes)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "07_int8_vs_float32.png"), dpi=150)
    plt.close(fig)


def _grafica_vad_impacto(d, out):
    if not d["vad_pairs"]:
        return
    pares = d["vad_pairs"]
    x = range(len(pares))
    w = 0.35
    etiquetas = [_etiqueta(p["nombre"]) for p in pares]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, k_sin, k_con, titulo in [
        (ax1, "wer_novad", "wer_vad", "WER"),
        (ax2, "rtf_novad", "rtf_vad", "RTF"),
    ]:
        ax.bar([xi - w/2 for xi in x], [p[k_sin] for p in pares], w,
               label="Sin VAD", color="steelblue")
        ax.bar([xi + w/2 for xi in x], [p[k_con] for p in pares], w,
               label="Con VAD", color="darkorange")
        ax.set_xticks(list(x))
        ax.set_xticklabels(etiquetas, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(titulo)
        ax.legend()
    fig.suptitle("Impacto del filtro VAD (vad_filter=True, beam_size=1)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "08_vad_impacto.png"), dpi=150)
    plt.close(fig)


def _grafica_heatmap_frases(d, out):
    if not d["clips_matrix"]:
        return
    nombres_cfg = sorted(set(c["nombre"] for c in d["clips_matrix"]))
    n_frases = max(c["frase_idx"] for c in d["clips_matrix"]) + 1
    matrix = []
    for fi in range(n_frases):
        row = []
        for n in nombres_cfg:
            vals = [c["wer"] for c in d["clips_matrix"]
                    if c["nombre"] == n and c["frase_idx"] == fi]
            row.append(vals[0] if vals else float("nan"))
        matrix.append(row)

    etiq_x = [_etiqueta(n)[:18] for n in nombres_cfg]
    etiq_y = [f"{fi}: {FRASES[fi][:28]}" if fi < len(FRASES) else f"{fi}: (frase {fi})" for fi in range(n_frases)]
    fig_w = max(10, len(nombres_cfg) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    mat_np = np.array(matrix)

    if HAS_SEABORN:
        sns.heatmap(mat_np, annot=True, fmt=".2f", cmap="RdYlGn_r",
                    xticklabels=etiq_x, yticklabels=etiq_y,
                    ax=ax, vmin=0, vmax=1.5, linewidths=0.4)
    else:
        img = ax.imshow(mat_np, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1.5)
        fig.colorbar(img, ax=ax)
        ax.set_xticks(range(len(nombres_cfg)))
        ax.set_xticklabels(etiq_x, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_frases))
        ax.set_yticklabels(etiq_y, fontsize=8)

    ax.set_title("WER por frase y configuración (rojo = peor, verde = mejor)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "09_heatmap_frases.png"), dpi=150)
    plt.close(fig)


def _grafica_eficiencia_compuesta(d, out):
    ranking = d.get("ranking", [])
    if not ranking:
        return
    pesos = d.get("pesos", {})
    alpha = pesos.get("alpha", 0.4)
    beta  = pesos.get("beta", 0.4)
    gamma = pesos.get("gamma", 0.2)

    ranking_s = sorted(ranking, key=lambda r: r["score"], reverse=True)
    etiquetas = [_etiqueta(r["nombre"]) for r in ranking_s]
    scores    = [r["score"] for r in ranking_s]
    colores   = ["gold" if i == 0 else "steelblue" for i in range(len(etiquetas))]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(etiquetas)), scores, color=colores, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"Score compuesto (α={alpha}·acc + β={beta}·vel + γ={gamma}·RAM)")
    ax.set_title("Ranking por eficiencia compuesta (dorado = mejor)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "10_eficiencia_compuesta.png"), dpi=150)
    plt.close(fig)


def _grafica_vad_threshold_sweep(d, out):
    sweep = d.get("sweep_vad", [])
    if not sweep:
        return
    ms_vals = [s["vad_ms"] for s in sweep]
    wers    = [s["promedios"]["wer"] for s in sweep]
    rtfs    = [s["promedios"]["rtf"] for s in sweep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ms_vals, wers, "-o", color="steelblue")
    ax1.set_xlabel("min_silence_duration_ms")
    ax1.set_ylabel("WER")
    ax1.set_title("WER vs umbral de silencio VAD")
    ax2.plot(ms_vals, rtfs, "-o", color="darkorange")
    ax2.set_xlabel("min_silence_duration_ms")
    ax2.set_ylabel("RTF")
    ax2.set_title("RTF vs umbral de silencio VAD")
    fig.suptitle(f"Barrido VAD threshold — {sweep[0]['nombre'].rsplit('_ms', 1)[0]}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "11_vad_threshold_sweep.png"), dpi=150)
    plt.close(fig)


def _grafica_vosk_chunk_sweep(d, out):
    chunks = d.get("chunks_sweep")
    if not chunks:
        return
    chunk_sizes = sorted(int(k) for k in chunks.keys())
    wers = [chunks[str(k)]["promedios"]["wer"] for k in chunk_sizes]
    rtfs = [chunks[str(k)]["promedios"]["rtf"] for k in chunk_sizes]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(chunk_sizes, wers, "-o", color="steelblue")
    ax1.set_xlabel("chunk_frames")
    ax1.set_ylabel("WER (media 10 clips)")
    ax1.set_title("WER vs tamaño de chunk Vosk")
    ax2.plot(chunk_sizes, rtfs, "-o", color="darkorange")
    ax2.set_xlabel("chunk_frames")
    ax2.set_ylabel("RTF (media 10 clips)")
    ax2.set_title("RTF vs tamaño de chunk Vosk")
    fig.suptitle("Barrido chunk_size Vosk — todos los clips")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "12_vosk_chunk_sweep.png"), dpi=150)
    plt.close(fig)


def generar_graficas(datos, graficas_dir, ranking_data=None):
    """Punto de entrada principal para generar todas las gráficas STT."""
    if not HAS_MATPLOTLIB:
        print("[AVISO] matplotlib no disponible — gráficas omitidas.")
        return
    os.makedirs(graficas_dir, exist_ok=True)
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", palette="muted")
    else:
        plt.style.use("ggplot")

    d = _preparar_datos_graficas(datos, ranking_data)
    if not d["configs"]:
        print("[AVISO] Sin configuraciones válidas para graficar.")
        return

    _grafica_wer_barras(d, graficas_dir)
    _grafica_rtf_barras(d, graficas_dir)
    _grafica_wer_rtf_scatter(d, graficas_dir)
    _grafica_latencia_boxplot(d, graficas_dir)
    _grafica_ram_barras(d, graficas_dir)
    _grafica_beam_size_linea(d, graficas_dir)
    _grafica_int8_vs_float32(d, graficas_dir)
    _grafica_vad_impacto(d, graficas_dir)
    _grafica_heatmap_frases(d, graficas_dir)
    _grafica_eficiencia_compuesta(d, graficas_dir)
    _grafica_vad_threshold_sweep(d, graficas_dir)
    _grafica_vosk_chunk_sweep(d, graficas_dir)

    n_archivos = len([f for f in os.listdir(graficas_dir) if f.endswith(".png")])
    print(f"[OK] {n_archivos} gráficas guardadas en: {graficas_dir}")


# ---------------------------------------------------------------------------
# INFORME MARKDOWN DE PARÁMETROS
# ---------------------------------------------------------------------------

def generar_informe_parametros(datos, ranking_data, output_path):
    """Genera un documento Markdown explicativo con metodología, parámetros y resultados."""
    hw = datos.get("hardware", {})
    pesos = ranking_data.get("pesos_usados", {})
    alpha = pesos.get("alpha", 0.4)
    beta  = pesos.get("beta", 0.4)
    gamma = pesos.get("gamma", 0.2)
    veredicto = ranking_data.get("veredicto", "N/A")
    ranking = ranking_data.get("ranking", [])
    n_reps = datos.get("n_reps", N_REPS)
    timestamp = datos.get("timestamp", "desconocido")

    lineas = []
    lineas.append("# Benchmark STT — Documentación de Parámetros y Metodología\n")
    lineas.append(f"> Generado automáticamente el {timestamp}\n")
    lineas.append(f"> Hardware: {hw.get('plataforma','?')} | {hw.get('cpu_cores','?')} cores | "
                  f"RAM {hw.get('ram_total_mb','?')} MB\n")
    lineas.append("")

    lineas.append("## 1. Objetivo del Benchmark\n")
    lineas.append(
        "Comparar motores de reconocimiento de voz (STT) para su uso como interfaz de lenguaje natural "
        "del robot TurtleBot4 con Raspberry Pi 4. El objetivo es identificar el motor que mejor equilibre "
        "**precisión de transcripción**, **latencia en tiempo real** y **huella de memoria**, dado que "
        "el hardware tiene recursos limitados (4 cores ARM Cortex-A72, 4 GB RAM).\n"
    )

    lineas.append("## 2. Motores Evaluados\n")
    lineas.append("### faster-whisper\n")
    lineas.append(
        "Implementación optimizada de OpenAI Whisper basada en CTranslate2. Whisper es un modelo "
        "Transformer encoder-decoder entrenado con 680.000 horas de audio multilingüe supervisado. "
        "El encoder procesa espectrogramas log-mel; el decoder genera tokens de texto autoregressivamente. "
        "CTranslate2 permite cuantización en int8 reduciendo uso de memoria y latencia con pérdida mínima de WER.\n"
    )
    lineas.append("### Vosk (Kaldi)\n")
    lineas.append(
        "Motor basado en Kaldi, que usa modelos acústicos DNN-HMM y modelos de lenguaje de n-gramas. "
        "Diseñado para reconocimiento en streaming (chunk por chunk), con bajo uso de RAM y latencia "
        "predecible. El modelo `vosk-model-small-es-0.42` está optimizado para español en hardware embebido.\n"
    )

    lineas.append("## 3. Parámetros del Experimento\n")
    lineas.append("### 3.1 Parámetros de faster-whisper\n")
    lineas.append("| Parámetro | Valores testados | Significado | Efecto esperado |")
    lineas.append("|-----------|-----------------|-------------|-----------------|")
    lineas.append("| `modelo` | `tiny`, `base` | Tamaño de la arquitectura Transformer (tiny: 39M params, base: 74M params) | Más grande → mejor WER, más RAM, más latencia |")
    lineas.append("| `compute_type` | `int8`, `float32` | Precisión numérica de los pesos durante la inferencia | int8: ~2-4× más rápido, ~40% menos RAM; pérdida de WER mínima (<2 pp) |")
    lineas.append("| `beam_size` | 1, 2, 3, 4, 5 | Amplitud del haz de búsqueda del decodificador: nº de hipótesis mantenidas simultáneamente | Más alto → mejor WER (exploración más amplia), más latencia (O(beam×L) tokens) |")
    lineas.append("| `best_of` | 1, 3 | Nº de hipótesis independientes generadas (requiere temperatura > 0) | Solo efectivo con temperatura > 0; aumenta WER a costa de latencia |")
    lineas.append("| `temperature` | 0.0, 0.2 | Aleatoriedad del muestreo del decodificador (0 = greedy determinista) | 0.0 = reproducible y estable; > 0 = estocástico, puede mejorar WER con `best_of` > 1 |")
    lineas.append("| `vad_filter` | True, False | Filtro de actividad de voz (WebRTC VAD) aplicado antes de la transcripción | Elimina segmentos de silencio; puede reducir WER en frases cortas y latencia en silencios largos |")
    lineas.append("| `min_silence_duration_ms` | 200–1000 ms | Duración mínima de silencio para que el VAD corte un segmento | Más corto: más reactivo, riesgo de cortar palabras; más largo: más conservador |")
    lineas.append("")

    lineas.append("### 3.2 Parámetros de Vosk\n")
    lineas.append("| Parámetro | Valores testados | Significado | Efecto esperado |")
    lineas.append("|-----------|-----------------|-------------|-----------------|")
    lineas.append("| `chunk_frames` | 500, 1000, 2000, 4000, 8000 | Nº de muestras de audio enviadas al reconocedor por iteración (a 16 kHz: 31–500 ms de audio por chunk) | Chunks pequeños: mayor latencia por overhead de llamadas; chunks grandes: menor overhead pero mayor buffer |")
    lineas.append("")

    lineas.append("### 3.3 Parámetros del Experimento\n")
    lineas.append("| Parámetro | Default | Flag CLI | Significado |")
    lineas.append("|-----------|---------|----------|-------------|")
    lineas.append(f"| `N_REPS` | 3 | `--n-reps N` | Repeticiones por clip; la 1ª siempre se descarta como warmup para eliminar sesgos de inicialización JIT |")
    lineas.append(f"| `alpha` | 0.4 | `--weights α β γ` | Peso de la precisión `(1-WER)` en el score compuesto |")
    lineas.append(f"| `beta` | 0.4 | `--weights α β γ` | Peso de la velocidad `(1-RTF_norm)` en el score compuesto |")
    lineas.append(f"| `gamma` | 0.2 | `--weights α β γ` | Peso del consumo de RAM `(1-RAM_norm)` en el score compuesto |")
    lineas.append("")

    lineas.append("## 4. Métricas Recogidas\n")
    lineas.append("### Word Error Rate (WER)\n")
    lineas.append(
        "Mide la distancia de edición a nivel de palabras entre la hipótesis (transcripción STT) "
        "y la referencia (ground truth), normalizada por la longitud de la referencia:\n\n"
        "```\nWER = (S + D + I) / N\n```\n\n"
        "donde S = sustituciones, D = borrados, I = inserciones, N = palabras en la referencia. "
        "Un WER de 0.0 es perfecto; valores > 1.0 indican más inserciones que palabras de referencia. "
        "**Normalización aplicada**: minúsculas, eliminación de diacríticos (NFD), solo alfanuméricos. "
        "Esto hace que `\"qué\"` = `\"que\"` y `\"llévame\"` = `\"llevame\"`, haciéndola más robusta "
        "ante variaciones ortográficas entre motores.\n"
    )

    lineas.append("### Real-Time Factor (RTF)\n")
    lineas.append(
        "Ratio entre el tiempo de inferencia y la duración del audio:\n\n"
        "```\nRTF = t_inferencia / t_audio\n```\n\n"
        "- **RTF < 1.0**: el motor transcribe más rápido que el audio en tiempo real → viable para asistente de voz\n"
        "- **RTF > 1.0**: el motor no puede seguir el ritmo → inviable para uso en tiempo real\n"
        "- En RPi4, se busca RTF < 0.5 para dejar margen a otros procesos del sistema\n"
    )

    lineas.append("### RAM pico (`ram_pico_mb`)\n")
    lineas.append(
        "Diferencia de RSS (Resident Set Size) del proceso antes y durante la transcripción, "
        "medida cada 100 ms con `psutil`. No incluye memoria compartida de bibliotecas. "
        "Relevante porque el RPi4 tiene 4 GB compartidos con el SO y otros procesos del robot.\n"
    )

    lineas.append("### CPU (`cpu_pct`)\n")
    lineas.append(
        "Media de `psutil.cpu_percent()` durante la inferencia. faster-whisper usa todos los "
        "cores disponibles vía OpenMP; Vosk es principalmente single-core. En RPi4, "
        "un CPU% alto puede interferir con otros nodos ROS2.\n"
    )

    lineas.append("### Warmup (`warmup_s`)\n")
    lineas.append(
        "Tiempo de la primera transcripción, excluida de los promedios. La primera inferencia "
        "suele ser más lenta por inicialización de buffers internos, caché JIT (ONNX/PyTorch) "
        "y carga de pesos a la caché de CPU. Reportarlo permite identificar configuraciones "
        "con overhead de inicialización alto.\n"
    )

    lineas.append("## 5. Corpus de Prueba\n")
    n_clips = datos.get("num_clips", len(FRASES))
    lineas.append(f"- **{n_clips} frases** de comandos de voz en español para robótica")
    lineas.append("- Duración media: ~2.7 s por frase")
    lineas.append("- Formato: WAV, mono, 16 kHz, PCM 16-bit")
    lineas.append("- **Fuente**: audio sintético generado con Piper TTS (voz `es_ES-davefice-medium`). "
                  "Para resultados representativos del uso real se recomienda audio humano grabado "
                  "en condiciones similares al entorno de despliegue del robot.")
    lineas.append("")
    lineas.append("| # | Frase de referencia |")
    lineas.append("|---|---------------------|")
    for i, f in enumerate(FRASES[:n_clips]):
        lineas.append(f"| {i:02d} | `{f}` |")
    lineas.append("")

    lineas.append("## 6. Metodología de Scoring Compuesto\n")
    lineas.append(
        f"Para elegir objetivamente el mejor modelo se utiliza un **score compuesto** "
        f"que pondera las tres dimensiones clave:\n\n"
        f"```\nScore = α·max(0, 1-WER) + β·(1-RTF_norm) + γ·(1-RAM_norm)\n```\n\n"
        f"donde RTF_norm y RAM_norm están normalizados min-max sobre todos los modelos testados "
        f"(0 = peor, 1 = mejor en esa métrica). Los pesos usados en esta ejecución: "
        f"**α={alpha}** (precisión), **β={beta}** (velocidad), **γ={gamma}** (RAM).\n\n"
        "Los pesos por defecto (0.4, 0.4, 0.2) reflejan el criterio de que para un asistente "
        "de voz robótico la velocidad de respuesta y la precisión son igualmente críticas, "
        "mientras que el consumo de RAM es una restricción más blanda (siempre que quepa en 4 GB). "
        "Se pueden ajustar con `--weights α β γ`.\n"
    )

    lineas.append("## 7. Intervalo de Confianza Wilson al 95%\n")
    lineas.append(
        "El WER medio sobre 10 clips es una estimación puntual con incertidumbre. "
        "Se aplica el **intervalo de Wilson** sobre la proporción total de errores de palabras "
        "(total_errores / total_palabras_referencia). Este método es más robusto que el intervalo "
        "normal de Wald para n pequeño y proporciones cercanas a 0 o 1.\n\n"
        "Un intervalo estrecho indica resultados más estables y reproducibles. "
        "Configuraciones con CI amplio pueden variar significativamente entre ejecuciones.\n"
    )

    lineas.append("## 8. Flags CLI Disponibles\n")
    lineas.append("```bash")
    lineas.append("# Ejecución normal (benchmark + ranking + gráficas):")
    lineas.append("python bench_stt_exhaustivo.py")
    lineas.append("")
    lineas.append("# Solo regenerar gráficas e informe desde JSON existente (sin re-ejecutar):")
    lineas.append("python bench_stt_exhaustivo.py --plot-only")
    lineas.append("")
    lineas.append("# Más repeticiones para mayor rigor estadístico:")
    lineas.append("python bench_stt_exhaustivo.py --n-reps 5")
    lineas.append("")
    lineas.append("# Barrido paramétrico completo (beam_size, VAD threshold, chunk, temperatura):")
    lineas.append("python bench_stt_exhaustivo.py --sweep")
    lineas.append("")
    lineas.append("# Cambiar pesos del score (priorizar precisión sobre velocidad):")
    lineas.append("python bench_stt_exhaustivo.py --weights 0.6 0.2 0.2")
    lineas.append("```\n")

    lineas.append("## 9. Resultados — Veredicto\n")
    lineas.append("```")
    for linea in veredicto.splitlines():
        lineas.append(linea)
    lineas.append("```\n")

    if ranking:
        lineas.append("## 10. Ranking Completo\n")
        lineas.append("| Pos | Configuración | Score | WER | CI 95% | RTF | RAM (MB) |")
        lineas.append("|-----|--------------|-------|-----|--------|-----|----------|")
        for r in ranking:
            ci = r["wer_ci_95"]
            lineas.append(
                f"| {r['posicion']} | `{r['nombre']}` | {r['score']:.4f} | "
                f"{r['wer']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{r['rtf']:.3f} | {r['ram_mb']} |"
            )
        lineas.append("")

    contenido = "\n".join(lineas)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(contenido)
    print(f"[OK] Informe de parámetros guardado en: {output_path}")


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
    parser = argparse.ArgumentParser(description="Benchmark STT exhaustivo RPi4")
    parser.add_argument("--generar-audio", action="store_true",
                        help="Mostrar instrucciones para grabar audio real (o generar con Piper si --piper-fallback)")
    parser.add_argument("--piper-fallback", action="store_true",
                        help="Generar audio sintético con Piper en lugar de audio humano real")
    parser.add_argument("--quick", action="store_true",
                        help="Modo rápido: solo primeros 2 clips por configuración")
    parser.add_argument("--models-dir",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
                        help="Directorio de modelos (default: models/ junto al script)")
    parser.add_argument("--sweep", action="store_true",
                        help="Ejecutar barrido paramétrico completo (beam_size, VAD threshold, "
                             "Vosk chunk_size, temperatura)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerar gráficas e informe desde JSON existente sin ejecutar benchmarks")
    parser.add_argument("--n-reps", type=int, default=3, metavar="N",
                        help="Repeticiones por clip (la 1ª es warmup). Default: 3")
    parser.add_argument("--weights", type=float, nargs=3,
                        default=list(DEFAULT_WEIGHTS), metavar=("ALPHA", "BETA", "GAMMA"),
                        help="Pesos score compuesto: ALPHA*(1-WER) + BETA*(1-RTF) + GAMMA*(1-RAM). "
                             "Default: 0.4 0.4 0.2")
    args = parser.parse_args()

    global N_REPS
    N_REPS = args.n_reps
    pesos = tuple(args.weights)

    models_dir = os.path.abspath(args.models_dir)
    audio_dir = os.path.join(os.path.dirname(__file__), "audio_tests")
    output = os.path.join(os.path.dirname(__file__), "resultados", "bench_stt_exhaustivo.json")
    graficas_dir = os.path.join(os.path.dirname(__file__), "resultados", "graficas_stt")
    informe_path = os.path.join(os.path.dirname(__file__), "resultados", "informe_stt_parametros.md")

    # --- Modo solo-gráficas: cargar JSON y regenerar sin ejecutar benchmarks ---
    if args.plot_only:
        if not os.path.exists(output):
            print(f"[ERROR] No existe {output}. Ejecuta primero sin --plot-only.")
            sys.exit(1)
        with open(output, encoding="utf-8") as f:
            datos = json.load(f)
        resultados_validos = [r for r in datos.get("resultados", [])
                              if r and "promedios" in r]
        ranking_data = calcular_ranking(resultados_validos, pesos=pesos, frases=FRASES)
        datos["ranking"] = ranking_data
        imprimir_ranking(ranking_data)
        generar_graficas(datos, graficas_dir, ranking_data)
        generar_informe_parametros(datos, ranking_data, informe_path)
        sys.exit(0)

    if args.generar_audio:
        if args.piper_fallback:
            ok = generar_audio_piper(FRASES, audio_dir, models_dir)
            if not ok:
                sys.exit(1)
        else:
            print("=" * 60)
            print("INSTRUCCIONES: Grabación de audio real para STT")
            print("=" * 60)
            print(f"\nGraba las siguientes {len(FRASES)} frases con voz humana real")
            print(f"(condiciones del robot, micrófono real, ruido ambiente):\n")
            for i, frase in enumerate(FRASES):
                print(f"  frase_{i:02d}.wav : \"{frase}\"")
            print(f"\nFormato requerido: WAV, mono, 16 kHz, PCM 16-bit")
            print(f"Destino: {audio_dir}/frase_NN.wav")
            print(f"\nHerramienta sugerida (RPi4):")
            print(f"  arecord -f S16_LE -r 16000 -c 1 {audio_dir}/frase_00.wav")
            print(f"\nAlternativa sintética (sin representatividad real):")
            print(f"  python3 {os.path.basename(__file__)} --generar-audio --piper-fallback")
            sys.exit(0)

    # Recoge solo archivos fraseNN.wav (con o sin guión bajo), ordena por número extraído
    # para garantizar que frase00→FRASES[0], frase01→FRASES[1], etc.
    def _indice_frase(nombre):
        import re
        m = re.search(r'frase_?(\d+)', nombre)
        return int(m.group(1)) if m else 999

    if os.path.isdir(audio_dir):
        candidatos = [f for f in os.listdir(audio_dir)
                      if f.endswith(".wav") and _indice_frase(f) < 999]
        archivos_wav = [
            os.path.join(audio_dir, f)
            for f in sorted(candidatos, key=_indice_frase)
        ]
        if not archivos_wav:
            # fallback: cualquier .wav presente, orden alfabético
            archivos_wav = sorted([
                os.path.join(audio_dir, f)
                for f in os.listdir(audio_dir) if f.endswith(".wav")
            ])
    else:
        archivos_wav = []

    if args.quick:
        archivos_wav = archivos_wav[:2]

    if not archivos_wav:
        print("[ERROR] No hay WAVs en audio_tests/.")
        print("  Opción A: Grabar voz real → python3 bench_stt_exhaustivo.py --generar-audio")
        print("  Opción B: Fallback Piper  → python3 bench_stt_exhaustivo.py --generar-audio --piper-fallback")
        sys.exit(1)

    # Empareja cada archivo con su ground truth por índice numérico del nombre
    ground_truths = []
    for f in archivos_wav:
        idx = _indice_frase(os.path.basename(f))
        if idx < len(FRASES):
            ground_truths.append(FRASES[idx])
        else:
            ground_truths.append("")   # frase sin referencia conocida

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

    # Vosk — devuelve lista de resultados (un config por chunk_size)
    print("\n" + "=" * 50)
    print("VOSK — variantes por chunk_size")
    print("=" * 50)
    vosk_lista = benchmark_vosk(archivos_wav, ground_truths, models_dir)
    resultados.extend(vosk_lista)

    imprimir_tabla(resultados)

    # --- Ranking y scoring compuesto ---
    resultados_validos = [r for r in resultados if r and "promedios" in r]
    ranking_data = calcular_ranking(resultados_validos, pesos=pesos, frases=FRASES)
    imprimir_ranking(ranking_data)

    # --- Barrido paramétrico (opcional) ---
    sweep_beams = []
    sweep_vad   = []
    if args.sweep:
        print("\n" + "=" * 60)
        print("BARRIDO PARAMÉTRICO")
        print("=" * 60)

        sweep_beams = benchmark_whisper_sweep_beams(
            archivos_wav, ground_truths, models_dir,
            beam_sizes=SWEEP_BEAM_SIZES,
            compute_types=["int8"],
            temperatures=SWEEP_TEMPERATURES,
        )

        whisper_valid = [r for r in resultados_validos if r.get("motor") == "faster-whisper"]
        if whisper_valid:
            best_whisper = min(whisper_valid, key=lambda r: r["promedios"]["wer"])
            sweep_vad = benchmark_whisper_vad_sweep(
                archivos_wav, ground_truths, models_dir,
                best_config=best_whisper["config"],
            )

        vosk_chunks = benchmark_vosk_chunk_sweep(archivos_wav, ground_truths, models_dir)

    # --- Guardar JSON ---
    salida = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": {
            "plataforma": platform.machine(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_mb": fmt_mb(mem.total),
            "ram_disponible_mb": fmt_mb(mem.available),
        },
        "num_clips": len(archivos_wav),
        "n_reps": N_REPS,
        "pesos_scoring": {"alpha": pesos[0], "beta": pesos[1], "gamma": pesos[2]},
        "resultados": [r for r in resultados if r is not None],
        "ranking": ranking_data,
        "sweep_beams": sweep_beams,
        "sweep_vad": sweep_vad,
        "chunks_sweep": vosk_chunks if args.sweep else None,
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en: {output}")

    # --- Gráficas ---
    generar_graficas(salida, graficas_dir, ranking_data)

    # --- Informe de parámetros ---
    generar_informe_parametros(salida, ranking_data, informe_path)


if __name__ == "__main__":
    main()
