#!/usr/bin/env python3
"""Benchmark TTS edge v2 — RPi4: Piper vs Coqui TTS vs KittenTTS.

Comparativa exhaustiva de tres motores TTS optimizados para edge computing.
Mide: TTFB (solo Piper, streaming real), RTF ponderado, P50/P95/P99 de latencia,
RAM pico, CPU% pico/medio, temperatura CPU, throttling RPi4, WER (Whisper tiny),
UTMOS opcional, y genera gráficas + informe Markdown.

Motores:
  - Piper TTS: referencia de velocidad, streaming real (TTFB real), ONNX Runtime.
  - Coqui TTS: VITS ES (css10) no-streaming / XTTS-v2 con fallback automático si RAM<2.5GB.
  - KittenTTS: API no-streaming, barrido voz × speed × temperature.

Metodología:
  - N_REPS repeticiones por frase; la primera se descarta como warmup (N_REPS-1 válidas).
  - RTF ponderado: Σtiempos / Σduraciones (no media de medias).
  - std calculado sobre TODAS las observaciones individuales (no media de stds por frase).
  - TTFB real solo en Piper; Coqui/KittenTTS muestran latencia total (tiempo_sintesis_s).
  - Corpus 50 frases agrupadas por longitud: corta/media/larga — RTF reportado por grupo.
  - Cooldown térmico entre configs (espera <55°C) para comparación justa en RPi4.
  - Throttling RPi4 registrado antes/después de cada config via sysfs get_throttled.
  - WER calculado con Whisper tiny ES (cargado una sola vez tras liberar motores TTS).
  - UTMOS: stub (pendiente cálculo offline con los WAVs generados).

Uso:
    python3 bench_tts_edgetts_v2.py [--quick] [--no-quality] [--output-dir DIR]

Flags:
    --quick        Reduce configs y reps para pruebas rápidas (N_REPS=3).
    --no-quality   Omite evaluación WER con Whisper (ahorra RAM).
    --output-dir   Directorio de salida (default: resultados/).
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
import statistics
import sys
import threading
import time
import unicodedata
import wave

try:
    import psutil
except ImportError:
    print("[ERROR] psutil no instalado. Instala con: pip install psutil")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CORPUS
# ---------------------------------------------------------------------------
# 50 frases organizadas en 3 grupos de longitud para detectar si un motor
# penaliza desproporcionadamente frases largas o tiene un coste fijo de arranque.
#   corta : ≤ 5 palabras
#   media : 6–12 palabras
#   larga : > 12 palabras

FRASES_POR_GRUPO = {
    "corta": [
        "Hola.",
        "Finalizado.",
        "Giro completado.",
        "Entendido, avanzando.",
        "Ruta recalculada.",
        "Modo pausa activado.",
        "Conexión perdida.",
        "Sistema listo.",
        "He detectado tres objetos.",
        "Navegando hacia la cocina.",
        "Estoy acoplada en la base.",
        "No puedo hacer eso ahora.",
        "Obstáculo detectado, deteniendo.",
        "Sensor de distancia activo.",
        "Batería cargada al cien por cien.",
    ],
    "media": [
        "No entiendo, por favor repite el comando.",
        "Veo una persona y una silla delante.",
        "Hola, soy Ana, tu robot asistente personal.",
        "Tiempo agotado, por favor intenta de nuevo.",
        "Iniciando secuencia de navegación autónoma.",
        "¿Puedes repetirlo? No te he oído bien.",
        "El robot TB4 usa ROS 2 Humble sobre Ubuntu.",
        "Error: batería al quince por ciento, volviendo a la base.",
        "Por favor, despeja el camino para que pueda pasar.",
        "Hay tres personas, dos sillas y una mesa en la habitación.",
        "Son las catorce horas y la temperatura es de veintidós grados.",
        "He completado la tarea de inspección del pasillo norte.",
        "El mapa ha sido actualizado con los nuevos obstáculos detectados.",
        "La velocidad máxima ha sido reducida al cincuenta por ciento.",
        "No se encontró ningún objeto en el área de búsqueda.",
        "Esperando instrucciones en el punto de espera designado.",
        "Iniciando protocolo de emergencia, avisando al operador.",
        "¿Deseas que repita el mensaje o continúo con la tarea?",
        "Temperatura del motor dentro de los límites normales de operación.",
        "Cargando el modelo de lenguaje, esto puede tardar unos segundos.",
    ],
    "larga": [
        "He detectado un obstáculo inesperado en mi ruta y estoy buscando un camino alternativo para llegar al destino.",
        "La batería se encuentra al doce por ciento, te recomiendo llevarme a la estación de carga lo antes posible.",
        "He completado el reconocimiento de la planta baja y no encontré ninguna anomalía en las áreas inspeccionadas.",
        "Para ejecutar este comando necesito que confirmes la operación, ya que implica mover objetos en el entorno.",
        "El sensor de ultrasonidos ha detectado un objeto a treinta centímetros de distancia en el lateral derecho.",
        "Actualmente estoy procesando el mapa del entorno y calculando la ruta óptima hacia el punto de destino.",
        "No he podido completar la tarea porque la puerta de acceso a la sala estaba bloqueada desde el interior.",
        "El sistema de visión ha identificado con un noventa y cinco por ciento de confianza que el objeto es una persona.",
        "Se ha producido un error en el módulo de planificación de rutas y el sistema ha reiniciado automáticamente.",
        "Por favor, asegúrate de que el área esté libre de personas antes de iniciar la secuencia de movimiento.",
        "He registrado una anomalía en el sensor de temperatura del motor izquierdo, te recomiendo una revisión técnica.",
        "La tarea de entrega ha sido completada con éxito y el paquete depositado en el lugar indicado.",
        "No puedo continuar la navegación porque el nivel de batería es insuficiente para completar el trayecto previsto.",
        "He actualizado el inventario de la sala de almacén y detecté que faltan tres artículos respecto al registro anterior.",
        "El módulo de reconocimiento de voz no interpretó tu comando, por favor habla más despacio y con claridad.",
    ],
}

# Lista plana (orden: cortas → medias → largas) para iterar en el benchmark
FRASES = [f for grupo in ("corta", "media", "larga") for f in FRASES_POR_GRUPO[grupo]]

# Mapa inverso frase → grupo para etiquetar cada medida
FRASES_GRUPO_MAP = {
    f: grupo
    for grupo, lista in FRASES_POR_GRUPO.items()
    for f in lista
}

N_REPS = 5  # la primera rep se descarta como warmup; 4 medidas válidas mínimo

# ---------------------------------------------------------------------------
# CONFIGURACIONES
# ---------------------------------------------------------------------------

# Piper: 5 param-sets, cada uno se combinará con las voces disponibles
PIPER_PARAM_SETS = [
    {"length_scale": 0.7, "noise_scale": 0.667, "noise_w": 0.8, "tag": "fast"},
    {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8, "tag": "default"},
    {"length_scale": 1.3, "noise_scale": 0.667, "noise_w": 0.8, "tag": "slow"},
    {"length_scale": 1.0, "noise_scale": 0.3,   "noise_w": 0.3, "tag": "lowvar"},
    {"length_scale": 1.0, "noise_scale": 1.0,   "noise_w": 1.0, "tag": "highvar"},
]

# Coqui TTS: VITS español con variaciones de length_scale (equivalente a speed)
# XTTS-v2 se intenta si hay suficiente RAM, sino se usa VITS como fallback
COQUI_CONFIGS = [
    {
        "nombre": "coqui_vits_speed08",
        "model": "tts_models/es/css10/vits",
        "length_scale": 1.25,  # inverso de speed 0.8
        "descripcion": "VITS ES speed=0.8 (más lento)",
    },
    {
        "nombre": "coqui_vits_speed10",
        "model": "tts_models/es/css10/vits",
        "length_scale": 1.0,
        "descripcion": "VITS ES speed=1.0 (default)",
    },
    {
        "nombre": "coqui_vits_speed12",
        "model": "tts_models/es/css10/vits",
        "length_scale": 0.833,  # inverso de speed 1.2
        "descripcion": "VITS ES speed=1.2 (más rápido)",
    },
    {
        "nombre": "coqui_xtts_v2",
        "model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "length_scale": 1.0,
        "descripcion": "XTTS v2 multilingual (~1.8 GB) — fallback a VITS si RAM insuficiente",
    },
]

# KittenTTS: 2 voces × 3 speeds × 3 temperatures = 18 configs
# En modo --quick: solo 1 voz, 1 speed, 1 temp (3 configs)
KITTEN_VOICES = ["es_female", "es_male"]
KITTEN_SPEEDS = [0.9, 1.0, 1.1]
KITTEN_TEMPERATURES = [0.5, 0.8, 1.0]


def _build_kitten_configs(quick=False):
    configs = []
    voices = KITTEN_VOICES[:1] if quick else KITTEN_VOICES
    speeds = [1.0] if quick else KITTEN_SPEEDS
    temps = [0.8] if quick else KITTEN_TEMPERATURES
    for voz in voices:
        for spd in speeds:
            for tmp in temps:
                nombre = f"kitten_{voz}_s{str(spd).replace('.','')}_t{str(tmp).replace('.','')}"
                configs.append({
                    "nombre": nombre,
                    "voice": voz,
                    "speed": spd,
                    "temperature": tmp,
                })
    return configs


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------

class MonitorRAM:
    """Muestrea RSS del proceso cada 50 ms para capturar el pico."""

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
            try:
                self.pico_rss = max(self.pico_rss, proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(0.05)

    def detener(self):
        self._corriendo = False
        if self._hilo:
            self._hilo.join(timeout=1)
        return self.pico_rss


class MonitorCPU:
    """Muestrea CPU% del proceso cada 50 ms para capturar pico y media."""

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
        proc = psutil.Process()
        proc.cpu_percent()  # descarta la primera lectura (siempre 0.0)
        while self._corriendo:
            try:
                pct = proc.cpu_percent()
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
    """Lee temperatura CPU desde sysfs (Linux/RPi4). Retorna °C o None."""
    ruta = "/sys/class/thermal/thermal_zone0/temp"
    try:
        with open(ruta) as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        return None


def leer_throttling_rpi():
    """Lee el bitmask de throttling del RPi4 vía sysfs. Retorna int o None.

    Bit 0: bajo voltaje actual. Bit 1: limitación de frecuencia activa.
    Bit 2: throttling activo. Bit 16-18: han ocurrido desde el último reset.
    Valor 0x0 indica sistema sano.
    """
    ruta = "/sys/devices/platform/soc/soc:firmware/get_throttled"
    try:
        with open(ruta) as f:
            return int(f.read().strip(), 16)
    except Exception:
        return None


def esperar_enfriamiento(temp_objetivo=55.0, timeout_s=60):
    """Espera hasta que la CPU baje de temp_objetivo°C o se alcance timeout_s."""
    temp = leer_temperatura_cpu()
    if temp is None or temp <= temp_objetivo:
        return
    print(f"  [COOL] Temperatura {temp}°C > {temp_objetivo}°C — esperando enfriamiento...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(5)
        temp = leer_temperatura_cpu()
        if temp is None or temp <= temp_objetivo:
            print(f"  [COOL] Temperatura OK ({temp}°C) tras {time.time()-t0:.0f}s")
            return
    print(f"  [WARN] Timeout enfriamiento ({timeout_s}s) — temperatura: {leer_temperatura_cpu()}°C")


def fmt_mb(b):
    return round(b / (1024 * 1024), 1)


def duracion_wav(ruta):
    try:
        with wave.open(ruta, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _std(valores):
    n = len(valores)
    if n < 2:
        return 0.0
    m = sum(valores) / n
    return round(math.sqrt(sum((v - m) ** 2 for v in valores) / (n - 1)), 4)


def _percentil(valores, p):
    """Percentil p (0-100) sin numpy."""
    if not valores:
        return 0.0
    s = sorted(valores)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 4)


def _normalizar_texto(texto):
    """Minúsculas, sin puntuación, normalización unicode para WER."""
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _wer_simple(ref, hyp):
    """Word Error Rate clásico (edit distance a nivel palabra)."""
    ref_words = ref.split()
    hyp_words = hyp.split()
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
# BENCHMARK PIPER
# ---------------------------------------------------------------------------

def _encontrar_modelos_piper(piper_dir):
    """Retorna lista de rutas .onnx en piper_dir."""
    if not os.path.isdir(piper_dir):
        return []
    return [
        os.path.join(piper_dir, f)
        for f in sorted(os.listdir(piper_dir))
        if f.endswith(".onnx") and not f.endswith(".onnx.json")
    ]


def benchmark_piper(frases, output_dir, models_dir, quick=False):
    """Benchmark Piper TTS con barrido de params × voces disponibles."""
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig
    except ImportError:
        print("  [SKIP] piper-tts no instalado")
        return []

    piper_dir = os.path.join(models_dir, "piper")
    modelos = _encontrar_modelos_piper(piper_dir)
    if not modelos:
        print(f"  [SKIP] No se encontraron modelos .onnx en {piper_dir}")
        return []

    param_sets = PIPER_PARAM_SETS[:2] if quick else PIPER_PARAM_SETS
    resultados = []

    for modelo_onnx in modelos:
        modelo_nombre = os.path.splitext(os.path.basename(modelo_onnx))[0]
        print(f"\n  Modelo Piper: {modelo_nombre}")

        # Cargar voz una sola vez por modelo
        monitor = MonitorRAM()
        monitor.iniciar()
        t0 = time.perf_counter()
        try:
            voice = PiperVoice.load(modelo_onnx)
        except Exception as e:
            monitor.detener()
            print(f"  [ERROR] No se pudo cargar {modelo_onnx}: {e}")
            continue
        t_carga = time.perf_counter() - t0
        ram_carga = monitor.detener()
        print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

        for params in param_sets:
            nombre = f"piper_{modelo_nombre}_{params['tag']}"
            print(f"\n    --- {nombre} ---")

            temp_inicio = leer_temperatura_cpu()
            resultado = {
                "motor": "piper",
                "nombre": nombre,
                "modelo": modelo_nombre,
                "config": {k: v for k, v in params.items() if k != "tag"},
                "tiempo_carga_s": round(t_carga, 2),
                "ram_carga_mb": fmt_mb(ram_carga),
                "temp_inicio_c": temp_inicio,
                "n_reps": N_REPS,
                "frases": [],
            }

            subdir = os.path.join(output_dir, nombre)
            os.makedirs(subdir, exist_ok=True)

            syn_config = SynthesisConfig(
                length_scale=params["length_scale"],
                noise_scale=params["noise_scale"],
                noise_w_scale=params["noise_w"],
            )

            todos_tiempos = []
            todos_duraciones = []
            temp_pico = temp_inicio or 0.0
            throttle_inicio = leer_throttling_rpi()

            for i, frase in enumerate(frases):
                wav_path = os.path.join(subdir, f"frase_{i:02d}.wav")
                tiempos, ttfbs = [], []
                cpu_picos, cpu_medios, ram_picos = [], [], []
                warmup_t = 0.0

                for rep in range(N_REPS):
                    mon_ram = MonitorRAM()
                    mon_cpu = MonitorCPU()
                    mon_ram.iniciar()
                    mon_cpu.iniciar()

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
                        mon_ram.detener()
                        mon_cpu.detener()
                        print(f"      [ERROR rep {rep}]: {e}")
                        break

                    t_sint = time.perf_counter() - t0
                    ttfb = (t_first - t0) if t_first else t_sint
                    ram_pico = mon_ram.detener()
                    cpu_pico, cpu_medio = mon_cpu.detener()

                    t_cpu = leer_temperatura_cpu()
                    if t_cpu and t_cpu > temp_pico:
                        temp_pico = t_cpu

                    if rep == 0:
                        warmup_t = t_sint
                    else:
                        tiempos.append(t_sint)
                        ttfbs.append(ttfb)
                        ram_picos.append(ram_pico)
                        cpu_picos.append(cpu_pico)
                        cpu_medios.append(cpu_medio)

                if not tiempos:
                    continue

                dur = duracion_wav(wav_path)
                rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
                media_t = sum(tiempos) / len(tiempos)
                media_rtf = sum(rtfs) / len(rtfs)
                media_ttfb = sum(ttfbs) / len(ttfbs)
                todos_tiempos.extend(tiempos)
                todos_duraciones.extend([dur] * len(tiempos))

                resultado["frases"].append({
                    "frase": frase,
                    "grupo": FRASES_GRUPO_MAP.get(frase, "media"),
                    "wav": wav_path,
                    "tiempo_sintesis_s": round(media_t, 3),
                    "std_tiempo_s": _std(tiempos),
                    "warmup_s": round(warmup_t, 3),
                    "ttfb_s": round(media_ttfb, 4),
                    "duracion_audio_s": round(dur, 2),
                    "rtf": round(media_rtf, 3),
                    "std_rtf": _std(rtfs),
                    "ram_pico_mb": fmt_mb(max(ram_picos)),
                    "cpu_pct_pico": round(max(cpu_picos), 1) if cpu_picos else 0.0,
                    "cpu_pct_medio": round(sum(cpu_medios) / len(cpu_medios), 1) if cpu_medios else 0.0,
                    "wer": None,
                    "utmos": None,
                })
                print(f"      [{i+1}/{len(frases)}] [{FRASES_GRUPO_MAP.get(frase,'?')[0].upper()}] \"{frase[:35]}\" | "
                      f"{media_t:.2f}±{_std(tiempos):.3f}s | TTFB={media_ttfb:.3f}s | "
                      f"RTF={media_rtf:.2f} | CPU={round(max(cpu_picos),0) if cpu_picos else '?'}%")

            resultado["temp_pico_c"] = temp_pico if temp_pico else None
            resultado["temp_fin_c"] = leer_temperatura_cpu()
            resultado["throttling_inicio"] = throttle_inicio
            resultado["throttling_fin"] = leer_throttling_rpi()

            datos = resultado["frases"]
            if datos:
                resultado["promedios"] = _calcular_promedios(
                    datos, todos_tiempos, todos_duraciones, streaming=True
                )
                resultado["promedios_por_grupo"] = _calcular_promedios_por_grupo(datos)

            resultados.append(resultado)
            esperar_enfriamiento()

        del voice
        gc.collect()
        time.sleep(1)

    return resultados


# ---------------------------------------------------------------------------
# BENCHMARK COQUI
# ---------------------------------------------------------------------------

def benchmark_coqui(frases, output_dir, quick=False):
    """Benchmark Coqui TTS. XTTS-v2 con fallback automático a VITS si RAM<2500 MB."""
    try:
        from TTS.api import TTS
    except ImportError:
        print("  [SKIP] Coqui TTS no instalado. pip install TTS")
        return []

    configs = COQUI_CONFIGS[:2] if quick else COQUI_CONFIGS
    resultados = []

    for config in configs:
        nombre = config["nombre"]
        model = config["model"]

        # Fallback XTTS → VITS si RAM insuficiente
        ram_libre = psutil.virtual_memory().available / (1024 * 1024)
        if "xtts" in model.lower() and ram_libre < 2500:
            print(f"\n  [FALLBACK] {nombre}: RAM libre={ram_libre:.0f} MB < 2500 → usando VITS ES")
            model = "tts_models/es/css10/vits"
            nombre = nombre.replace("xtts_v2", "vits_fallback")

        print(f"\n  --- {nombre} ({config['descripcion']}) ---")

        resultado = {
            "motor": "coqui",
            "nombre": nombre,
            "config": {k: v for k, v in config.items() if k != "descripcion"},
            "modelo_efectivo": model,
            "n_reps": N_REPS,
            "nota": "TTFB = tiempo total de síntesis (API no-streaming).",
            "frases": [],
        }

        monitor = MonitorRAM()
        monitor.iniciar()
        temp_inicio = leer_temperatura_cpu()
        resultado["temp_inicio_c"] = temp_inicio

        t0 = time.perf_counter()
        try:
            tts = TTS(model_name=model, progress_bar=False, gpu=False)
        except Exception as e:
            monitor.detener()
            print(f"  [ERROR] {e}")
            resultado["error"] = str(e)
            resultados.append(resultado)
            continue

        t_carga = time.perf_counter() - t0
        ram_carga = monitor.detener()
        resultado["tiempo_carga_s"] = round(t_carga, 2)
        resultado["ram_carga_mb"] = fmt_mb(ram_carga)
        print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

        subdir = os.path.join(output_dir, nombre)
        os.makedirs(subdir, exist_ok=True)

        todos_tiempos = []
        todos_duraciones = []
        temp_pico = temp_inicio or 0.0
        length_scale = config.get("length_scale", 1.0)
        es_multilingual = "multilingual" in model or "xtts" in model.lower()
        throttle_inicio = leer_throttling_rpi()

        for i, frase in enumerate(frases):
            wav_path = os.path.join(subdir, f"frase_{i:02d}.wav")
            tiempos = []
            cpu_picos, cpu_medios, ram_picos = [], [], []
            warmup_t = 0.0

            for rep in range(N_REPS):
                mon_ram = MonitorRAM()
                mon_cpu = MonitorCPU()
                mon_ram.iniciar()
                mon_cpu.iniciar()

                t0 = time.perf_counter()
                try:
                    kwargs = {}
                    if length_scale != 1.0:
                        kwargs["length_scale"] = length_scale
                    if es_multilingual:
                        tts.tts_to_file(text=frase, file_path=wav_path, language="es", **kwargs)
                    else:
                        tts.tts_to_file(text=frase, file_path=wav_path, **kwargs)
                except Exception as e:
                    mon_ram.detener()
                    mon_cpu.detener()
                    print(f"    [ERROR rep {rep}]: {e}")
                    break

                t_sint = time.perf_counter() - t0
                ram_pico = mon_ram.detener()
                cpu_pico, cpu_medio = mon_cpu.detener()

                t_cpu = leer_temperatura_cpu()
                if t_cpu and t_cpu > temp_pico:
                    temp_pico = t_cpu

                if rep == 0:
                    warmup_t = t_sint
                else:
                    tiempos.append(t_sint)
                    ram_picos.append(ram_pico)
                    cpu_picos.append(cpu_pico)
                    cpu_medios.append(cpu_medio)

            if not tiempos:
                continue

            dur = duracion_wav(wav_path)
            rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
            media_t = sum(tiempos) / len(tiempos)
            media_rtf = sum(rtfs) / len(rtfs)
            todos_tiempos.extend(tiempos)
            todos_duraciones.extend([dur] * len(tiempos))

            resultado["frases"].append({
                "frase": frase,
                "grupo": FRASES_GRUPO_MAP.get(frase, "media"),
                "wav": wav_path,
                "tiempo_sintesis_s": round(media_t, 3),
                "std_tiempo_s": _std(tiempos),
                "warmup_s": round(warmup_t, 3),
                "ttfb_s": None,  # API no-streaming: no hay TTFB real
                "duracion_audio_s": round(dur, 2),
                "rtf": round(media_rtf, 3),
                "std_rtf": _std(rtfs),
                "ram_pico_mb": fmt_mb(max(ram_picos)),
                "cpu_pct_pico": round(max(cpu_picos), 1) if cpu_picos else 0.0,
                "cpu_pct_medio": round(sum(cpu_medios) / len(cpu_medios), 1) if cpu_medios else 0.0,
                "wer": None,
                "utmos": None,
            })
            print(f"    [{i+1}/{len(frases)}] [{FRASES_GRUPO_MAP.get(frase,'?')[0].upper()}] \"{frase[:35]}\" | "
                  f"{media_t:.2f}±{_std(tiempos):.3f}s | RTF={media_rtf:.2f} | "
                  f"CPU={round(max(cpu_picos),0) if cpu_picos else '?'}%")

        resultado["temp_pico_c"] = temp_pico if temp_pico else None
        resultado["temp_fin_c"] = leer_temperatura_cpu()
        resultado["throttling_inicio"] = throttle_inicio
        resultado["throttling_fin"] = leer_throttling_rpi()

        datos = resultado["frases"]
        if datos:
            resultado["promedios"] = _calcular_promedios(
                datos, todos_tiempos, todos_duraciones, streaming=False
            )
            resultado["promedios_por_grupo"] = _calcular_promedios_por_grupo(datos)

        del tts
        gc.collect()
        esperar_enfriamiento()
        resultados.append(resultado)

    return resultados


# ---------------------------------------------------------------------------
# BENCHMARK KITTENTTS
# ---------------------------------------------------------------------------

def benchmark_kitten(frases, output_dir, quick=False):
    """Benchmark KittenTTS. TTFB = tiempo total (API no-streaming)."""
    try:
        from kittentts import KittenTTS  # noqa: F401
        kitten_disponible = True
    except ImportError:
        print("  [SKIP] KittenTTS no instalado. pip install kittentts")
        return []

    configs = _build_kitten_configs(quick=quick)
    resultados = []

    for config in configs:
        nombre = config["nombre"]
        print(f"\n  --- {nombre} (voice={config['voice']}, speed={config['speed']}, "
              f"temp={config['temperature']}) ---")

        resultado = {
            "motor": "kitten",
            "nombre": nombre,
            "config": config.copy(),
            "n_reps": N_REPS,
            "nota": "TTFB = tiempo total de síntesis (API no-streaming).",
            "frases": [],
        }

        # Intentar cargar modelo
        monitor = MonitorRAM()
        monitor.iniciar()
        temp_inicio = leer_temperatura_cpu()
        resultado["temp_inicio_c"] = temp_inicio

        t0 = time.perf_counter()
        try:
            model = KittenTTS(voice=config["voice"])
        except TypeError:
            try:
                model = KittenTTS()
            except Exception as e:
                monitor.detener()
                print(f"  [ERROR] No se pudo instanciar KittenTTS: {e}")
                resultado["error"] = str(e)
                resultados.append(resultado)
                continue
        except Exception as e:
            monitor.detener()
            print(f"  [ERROR] {e}")
            resultado["error"] = str(e)
            resultados.append(resultado)
            continue

        t_carga = time.perf_counter() - t0
        ram_carga = monitor.detener()
        resultado["tiempo_carga_s"] = round(t_carga, 2)
        resultado["ram_carga_mb"] = fmt_mb(ram_carga)
        print(f"  Cargado en {t_carga:.1f}s | RAM: {fmt_mb(ram_carga)} MB")

        subdir = os.path.join(output_dir, nombre)
        os.makedirs(subdir, exist_ok=True)

        todos_tiempos = []
        todos_duraciones = []
        temp_pico = temp_inicio or 0.0
        throttle_inicio = leer_throttling_rpi()

        for i, frase in enumerate(frases):
            wav_path = os.path.join(subdir, f"frase_{i:02d}.wav")
            tiempos = []
            cpu_picos, cpu_medios, ram_picos = [], [], []
            warmup_t = 0.0

            for rep in range(N_REPS):
                mon_ram = MonitorRAM()
                mon_cpu = MonitorCPU()
                mon_ram.iniciar()
                mon_cpu.iniciar()

                t0 = time.perf_counter()
                generated = False
                # Intentar varias firmas de API posibles
                for fn in [
                    lambda: model.generate(
                        frase, voice=config["voice"],
                        speed=config["speed"], temperature=config["temperature"],
                        output=wav_path),
                    lambda: model.synthesize(
                        frase, speed=config["speed"],
                        temperature=config["temperature"],
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
                ram_pico = mon_ram.detener()
                cpu_pico, cpu_medio = mon_cpu.detener()

                t_cpu = leer_temperatura_cpu()
                if t_cpu and t_cpu > temp_pico:
                    temp_pico = t_cpu

                if not generated:
                    break

                if rep == 0:
                    warmup_t = t_sint
                else:
                    tiempos.append(t_sint)
                    ram_picos.append(ram_pico)
                    cpu_picos.append(cpu_pico)
                    cpu_medios.append(cpu_medio)

            if not tiempos:
                continue

            dur = duracion_wav(wav_path)
            rtfs = [t / dur for t in tiempos] if dur > 0 else [0.0] * len(tiempos)
            media_t = sum(tiempos) / len(tiempos)
            media_rtf = sum(rtfs) / len(rtfs)
            todos_tiempos.extend(tiempos)
            todos_duraciones.extend([dur] * len(tiempos))

            resultado["frases"].append({
                "frase": frase,
                "grupo": FRASES_GRUPO_MAP.get(frase, "media"),
                "wav": wav_path,
                "tiempo_sintesis_s": round(media_t, 3),
                "std_tiempo_s": _std(tiempos),
                "warmup_s": round(warmup_t, 3),
                "ttfb_s": None,  # API no-streaming: no hay TTFB real
                "duracion_audio_s": round(dur, 2),
                "rtf": round(media_rtf, 3),
                "std_rtf": _std(rtfs),
                "ram_pico_mb": fmt_mb(max(ram_picos)),
                "cpu_pct_pico": round(max(cpu_picos), 1) if cpu_picos else 0.0,
                "cpu_pct_medio": round(sum(cpu_medios) / len(cpu_medios), 1) if cpu_medios else 0.0,
                "wer": None,
                "utmos": None,
            })
            print(f"    [{i+1}/{len(frases)}] [{FRASES_GRUPO_MAP.get(frase,'?')[0].upper()}] \"{frase[:35]}\" | "
                  f"{media_t:.2f}±{_std(tiempos):.3f}s | RTF={media_rtf:.2f} | "
                  f"CPU={round(max(cpu_picos),0) if cpu_picos else '?'}%")

        resultado["temp_pico_c"] = temp_pico if temp_pico else None
        resultado["temp_fin_c"] = leer_temperatura_cpu()
        resultado["throttling_inicio"] = throttle_inicio
        resultado["throttling_fin"] = leer_throttling_rpi()

        datos = resultado["frases"]
        if datos:
            resultado["promedios"] = _calcular_promedios(
                datos, todos_tiempos, todos_duraciones, streaming=False
            )
            resultado["promedios_por_grupo"] = _calcular_promedios_por_grupo(datos)

        del model
        gc.collect()
        esperar_enfriamiento()
        resultados.append(resultado)

    return resultados


# ---------------------------------------------------------------------------
# PROMEDIOS Y PERCENTILES
# ---------------------------------------------------------------------------

def _calcular_promedios(datos, todos_tiempos, todos_duraciones, streaming=True):
    """Calcula promedios y percentiles.

    RTF ponderado: sum(synthesis_times) / sum(audio_durations), no media de medias.
    std_tiempo_s: desviación estándar de TODAS las observaciones individuales (no media de std).
    ttfb_s: solo válido para motores con streaming real (Piper). None para los demás.
    """
    n = len(datos)

    # RTF ponderado por duración (correcto)
    total_dur = sum(todos_duraciones)
    rtf_global = round(sum(todos_tiempos) / total_dur, 3) if total_dur > 0 else 0.0

    # std sobre todas las observaciones individuales (correcto)
    std_global = _std(todos_tiempos)

    # TTFB solo para motores con streaming real
    ttfbs_validos = [d["ttfb_s"] for d in datos if d.get("ttfb_s") is not None]
    ttfb_medio = round(sum(ttfbs_validos) / len(ttfbs_validos), 4) if ttfbs_validos else None

    return {
        "tiempo_sintesis_s": round(sum(d["tiempo_sintesis_s"] for d in datos) / n, 3),
        "std_tiempo_s": round(std_global, 4),
        "ttfb_s": ttfb_medio,
        "es_streaming": streaming,
        "duracion_audio_s": round(sum(d["duracion_audio_s"] for d in datos) / n, 2),
        "rtf": rtf_global,
        "std_rtf": _std([d["rtf"] for d in datos]),
        "ram_pico_mb": max(d["ram_pico_mb"] for d in datos),
        "cpu_pct_pico": max(d.get("cpu_pct_pico", 0) for d in datos),
        "cpu_pct_medio": round(sum(d.get("cpu_pct_medio", 0) for d in datos) / n, 1),
        "p50_sintesis_s": _percentil(todos_tiempos, 50),
        "p95_sintesis_s": _percentil(todos_tiempos, 95),
        "p99_sintesis_s": _percentil(todos_tiempos, 99),
        "wer_medio": None,
        "utmos_medio": None,
    }


def _calcular_promedios_por_grupo(datos):
    """Calcula RTF ponderado y percentiles por grupo de longitud (corta/media/larga).

    Usa tiempo_sintesis_s (media de reps válidas por frase) para RTF y percentiles.
    No tenemos acceso a las observaciones individuales por rep aquí, pero con
    N_REPS=5 (4 válidas) la media por frase es suficientemente estable para P50/P95.
    """
    grupos = {}
    for g in ("corta", "media", "larga"):
        subset = [d for d in datos if d.get("grupo") == g]
        if not subset:
            continue
        tiempos_g = [d["tiempo_sintesis_s"] for d in subset]
        durs_g = [d["duracion_audio_s"] for d in subset]
        total_dur = sum(durs_g)
        rtf_g = round(sum(tiempos_g) / total_dur, 3) if total_dur > 0 else 0.0
        grupos[g] = {
            "n_frases": len(subset),
            "rtf": rtf_g,
            "p50_s": _percentil(tiempos_g, 50),
            "p95_s": _percentil(tiempos_g, 95),
            "tiempo_medio_s": round(sum(tiempos_g) / len(tiempos_g), 3),
            "std_s": _std(tiempos_g),
        }
    return grupos


# ---------------------------------------------------------------------------
# EVALUACIÓN DE CALIDAD: WER
# ---------------------------------------------------------------------------

def evaluar_calidad(resultados, no_quality=False):
    """Añade WER a cada frase usando Whisper tiny. Carga el modelo una sola vez."""
    if no_quality:
        print("\n  [INFO] Evaluación de calidad omitida (--no-quality).")
        return

    try:
        import whisper
    except ImportError:
        print("\n  [SKIP] openai-whisper no instalado. pip install openai-whisper")
        return

    print("\n" + "=" * 50)
    print("EVALUACIÓN WER — Whisper tiny")
    print("=" * 50)

    ram_libre = psutil.virtual_memory().available / (1024 * 1024)
    if ram_libre < 500:
        print(f"  [SKIP] RAM libre insuficiente: {ram_libre:.0f} MB")
        return

    print("  Cargando Whisper tiny...")
    try:
        modelo_whisper = whisper.load_model("tiny")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    for resultado in resultados:
        if resultado is None or "frases" not in resultado:
            continue
        nombre = resultado.get("nombre", "?")
        wers = []
        for frase_data in resultado["frases"]:
            wav_path = frase_data.get("wav", "")
            ref = frase_data.get("frase", "")
            if not wav_path or not os.path.exists(wav_path):
                continue
            try:
                out = modelo_whisper.transcribe(wav_path, language="es", fp16=False)
                hyp = out.get("text", "")
                wer = _wer_simple(_normalizar_texto(ref), _normalizar_texto(hyp))
                frase_data["wer"] = wer
                wers.append(wer)
            except Exception as e:
                print(f"    [WARN] WER frase error: {e}")

        if wers and "promedios" in resultado:
            media_wer = round(sum(wers) / len(wers), 4)
            resultado["promedios"]["wer_medio"] = media_wer
            print(f"  {nombre[:40]:<40} WER medio = {media_wer:.4f}")

    del modelo_whisper
    gc.collect()


# ---------------------------------------------------------------------------
# UTMOS (stub offline)
# ---------------------------------------------------------------------------

def calcular_utmos(resultados):
    """Intenta calcular UTMOS si torch_utmos está disponible. Stub en caso contrario."""
    try:
        import utmos  # noqa: F401
        utmos_disponible = True
    except ImportError:
        utmos_disponible = False

    if not utmos_disponible:
        print("\n  [INFO] UTMOS no disponible (pip install utmos opcional). "
              "Calcula offline con los WAVs generados.")
        return

    # Si el paquete está disponible, intentar scoring básico
    for resultado in resultados:
        if resultado is None or "frases" not in resultado:
            continue
        scores = []
        for frase_data in resultado["frases"]:
            wav_path = frase_data.get("wav", "")
            if not wav_path or not os.path.exists(wav_path):
                continue
            try:
                score = utmos.predict(wav_path)
                frase_data["utmos"] = round(float(score), 3)
                scores.append(float(score))
            except Exception:
                pass
        if scores and "promedios" in resultado:
            resultado["promedios"]["utmos_medio"] = round(sum(scores) / len(scores), 3)


# ---------------------------------------------------------------------------
# VISUALIZACIÓN
# ---------------------------------------------------------------------------

def generar_graficas(resultados, graficas_dir):
    """Genera barras RTF/TTFB, boxplot de latencias y radar normalizado."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib no instalado. pip install matplotlib")
        return []

    os.makedirs(graficas_dir, exist_ok=True)
    archivos = []

    datos_validos = [r for r in resultados if r and "promedios" in r]
    if not datos_validos:
        return archivos

    nombres = [r["nombre"][:25] for r in datos_validos]
    motores = [r.get("motor", "?") for r in datos_validos]
    colores_motor = {"piper": "#2196F3", "coqui": "#4CAF50", "kitten": "#FF9800"}
    colores = [colores_motor.get(m, "#9E9E9E") for m in motores]

    # --- 1. Barras RTF y TTFB ---
    rtfs = [r["promedios"].get("rtf", 0) for r in datos_validos]
    # Latencia representativa: TTFB real para streaming, tiempo_sintesis para no-streaming
    latencias = [
        r["promedios"].get("ttfb_s") or r["promedios"].get("tiempo_sintesis_s", 0)
        for r in datos_validos
    ]
    latencia_labels = [
        "TTFB" if r["promedios"].get("es_streaming") else "Latencia"
        for r in datos_validos
    ]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(nombres) * 0.8), 6))
    fig.suptitle("Comparativa TTS — RTF y Latencia por configuración", fontsize=13)

    x = range(len(nombres))
    axes[0].bar(x, rtfs, color=colores)
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1, label="RTF=1 (tiempo real)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(nombres, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("RTF (menor = mejor)")
    axes[0].set_title("Real-Time Factor (ponderado)")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, latencias, color=colores)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(nombres, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Latencia (s) — menor = mejor")
    axes[1].set_title("Latencia (TTFB real si streaming, total si no)")

    # Leyenda motores
    from matplotlib.patches import Patch
    leyenda = [Patch(color=c, label=m) for m, c in colores_motor.items()]
    fig.legend(handles=leyenda, loc="upper right", fontsize=8)

    fig.tight_layout()
    ruta = os.path.join(graficas_dir, "rtf_ttfb_barras.png")
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    archivos.append(ruta)
    print(f"  Gráfica guardada: {ruta}")

    # --- 2. Boxplot de latencias por motor ---
    from collections import defaultdict
    latencias_por_motor = defaultdict(list)
    for r in datos_validos:
        motor = r.get("motor", "?")
        for fd in r.get("frases", []):
            if fd.get("tiempo_sintesis_s"):
                latencias_por_motor[motor].append(fd["tiempo_sintesis_s"])

    if latencias_por_motor:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        motor_labels = list(latencias_por_motor.keys())
        motor_data = [latencias_por_motor[m] for m in motor_labels]
        bp = ax2.boxplot(motor_data, labels=motor_labels, patch_artist=True)
        for patch, motor in zip(bp["boxes"], motor_labels):
            patch.set_facecolor(colores_motor.get(motor, "#9E9E9E"))
        ax2.set_ylabel("Tiempo síntesis (s)")
        ax2.set_title("Distribución de latencia por motor")
        fig2.tight_layout()
        ruta2 = os.path.join(graficas_dir, "boxplot_latencias.png")
        fig2.savefig(ruta2, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        archivos.append(ruta2)
        print(f"  Gráfica guardada: {ruta2}")

    # --- 3. Radar normalizado (top-5 configs por RTF) ---
    metricas_radar = ["RTF", "Latencia", "WER", "RAM_MB", "CPU%pico"]
    top = sorted(datos_validos, key=lambda r: r["promedios"].get("rtf", 999))[:5]

    if len(top) >= 2:
        valores_raw = {
            "RTF":      [r["promedios"].get("rtf", 0) for r in top],
            "Latencia": [
                r["promedios"].get("ttfb_s") or r["promedios"].get("tiempo_sintesis_s", 0)
                for r in top
            ],
            "WER":      [r["promedios"].get("wer_medio") or 1.0 for r in top],
            "RAM_MB":   [r["promedios"].get("ram_pico_mb", 0) for r in top],
            "CPU%pico": [r["promedios"].get("cpu_pct_pico", 0) for r in top],
        }

        def _norm_inv(vals):
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return [0.5] * len(vals)
            return [1 - (v - mn) / (mx - mn) for v in vals]  # invertido: menor=mejor→mayor score

        norm = {k: _norm_inv(v) for k, v in valores_raw.items()}

        N_ejes = len(metricas_radar)
        angulos = [n / float(N_ejes) * 2 * math.pi for n in range(N_ejes)]
        angulos += angulos[:1]

        fig3, ax3 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax3.set_theta_offset(math.pi / 2)
        ax3.set_theta_direction(-1)
        ax3.set_xticks(angulos[:-1])
        ax3.set_xticklabels(metricas_radar, fontsize=9)

        for idx, r in enumerate(top):
            vals = [norm[m][idx] for m in metricas_radar]
            vals += vals[:1]
            color = colores_motor.get(r.get("motor", "?"), "#9E9E9E")
            ax3.plot(angulos, vals, linewidth=1.5, color=color)
            ax3.fill(angulos, vals, alpha=0.1, color=color)

        # Leyenda
        handles = [plt.Line2D([0], [0], color=colores_motor.get(r.get("motor"), "#9E9E9E"),
                               linewidth=2, label=r["nombre"][:20]) for r in top]
        ax3.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
        ax3.set_title("Radar: métricas normalizadas (mayor = mejor)", fontsize=11, pad=20)

        fig3.tight_layout()
        ruta3 = os.path.join(graficas_dir, "radar_metricas.png")
        fig3.savefig(ruta3, dpi=120, bbox_inches="tight")
        plt.close(fig3)
        archivos.append(ruta3)
        print(f"  Gráfica guardada: {ruta3}")

    return archivos


# ---------------------------------------------------------------------------
# INFORME MARKDOWN
# ---------------------------------------------------------------------------

def generar_informe_md(resultados, archivos_graficas, out_path):
    """Genera informe Markdown con tabla resumen y gráficos incrustados."""
    lineas = [
        "# Informe Benchmark TTS Edge v2",
        "",
        f"**Fecha:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Plataforma:** {platform.machine()} — {platform.system()} {platform.release()}  ",
        f"**CPU cores:** {psutil.cpu_count(logical=True)}  ",
        f"**RAM total:** {fmt_mb(psutil.virtual_memory().total)} MB  ",
        "",
        "## Metodología",
        "",
        f"- **N_REPS:** {N_REPS} (primera repetición descartada como warmup; {N_REPS-1} medidas válidas)",
        f"- **Corpus:** {len(FRASES)} frases — "
        f"{len(FRASES_POR_GRUPO['corta'])} cortas (≤5 pal.), "
        f"{len(FRASES_POR_GRUPO['media'])} medias (6-12 pal.), "
        f"{len(FRASES_POR_GRUPO['larga'])} largas (>12 pal.).",
        "- **RTF:** ponderado por duración — `Σsynthesis_times / Σaudio_durations` (no media de medias).",
        "- **std:** desviación estándar de TODAS las observaciones individuales (no media de stds por frase).",
        "- **TTFB:** Time to First Byte — real solo en Piper (streaming). "
        "Coqui/Kitten no tienen streaming: se muestra `-` y se usa `tiempo_sintesis_s` como latencia.",
        "- **CPU%:** monitorizado continuamente cada 50 ms durante la síntesis.",
        "- **Throttling (RPi4):** bitmask de `/sys/.../get_throttled` registrado antes/después de cada config.",
        "- **WER:** calculado con Whisper tiny ES (normalización unicode + edit-distance).",
        "- **UTMOS:** pendiente cálculo offline con los WAVs generados si `utmos` no disponible.",
        "",
        "## Tabla resumen",
        "",
    ]

    # Cabecera tabla
    cols = ["Config", "Motor", "Carga(s)", "RAM(MB)", "RTF±std", "TTFB(s)*",
            "P50(s)", "P95(s)", "WER", "CPU%pico", "Throttle", "Temp_pico(°C)"]
    lineas.append("| " + " | ".join(cols) + " |")
    lineas.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for r in resultados:
        if not r or "promedios" not in r:
            continue
        p = r["promedios"]

        def _v(x):
            return str(x) if x is not None else "N/A"

        rtf_str = f"{p.get('rtf', 'N/A')}±{p.get('std_rtf', 'N/A')}"
        ttfb_str = _v(p.get("ttfb_s")) if p.get("es_streaming") else "-"
        throttle_inicio = r.get("throttling_inicio")
        throttle_fin = r.get("throttling_fin")
        throttle_str = (
            f"0x{throttle_fin:x}" if throttle_fin is not None else "N/A"
        )

        fila = [
            r["nombre"][:30],
            r.get("motor", "?"),
            _v(r.get("tiempo_carga_s")),
            _v(p.get("ram_pico_mb")),
            rtf_str,
            ttfb_str,
            _v(p.get("p50_sintesis_s")),
            _v(p.get("p95_sintesis_s")),
            _v(p.get("wer_medio")),
            _v(p.get("cpu_pct_pico")),
            throttle_str,
            _v(r.get("temp_pico_c")),
        ]
        lineas.append("| " + " | ".join(fila) + " |")

    lineas += [
        "",
        "> \\* TTFB real solo para motores con streaming (Piper). "
        "Para Coqui/KittenTTS el tiempo de latencia es `tiempo_sintesis_s` (ver P50/P95).",
        "",
        "## Gráficas",
        "",
    ]
    for ruta in archivos_graficas:
        nombre_img = os.path.basename(ruta)
        ruta_rel = os.path.relpath(ruta, os.path.dirname(out_path))
        lineas.append(f"### {nombre_img}")
        lineas.append(f"![{nombre_img}]({ruta_rel})")
        lineas.append("")

    # --- Tabla por grupo de longitud ---
    lineas += [
        "## RTF por grupo de longitud de frase",
        "",
        "Detecta si un motor penaliza desproporcionadamente frases largas "
        "o tiene un coste fijo de arranque alto.",
        "",
    ]

    for r in resultados:
        if not r or "promedios_por_grupo" not in r:
            continue
        ppg = r["promedios_por_grupo"]
        if not ppg:
            continue
        lineas.append(f"### {r['nombre'][:40]} ({r.get('motor','?')})")
        lineas.append("")
        lineas.append("| Grupo | N frases | RTF | T.medio±std (s) | P50(s) | P95(s) |")
        lineas.append("| --- | --- | --- | --- | --- | --- |")
        for g in ("corta", "media", "larga"):
            if g not in ppg:
                continue
            gd = ppg[g]
            lineas.append(
                f"| {g} | {gd['n_frases']} | {gd['rtf']} | "
                f"{gd['tiempo_medio_s']}±{gd['std_s']} | {gd['p50_s']} | {gd['p95_s']} |"
            )
        lineas.append("")

    lineas += [
        "## Notas de implementación",
        "",
        "- **RTF ponderado:** `Σtiempos / Σduraciones` — no afectado por el peso de frases cortas.",
        "- **std global:** calculado sobre todas las observaciones individuales, no promediando stds.",
        "- **WER propio vs jiwer:** implementación simple sin dependencia extra.",
        "- **UTMOS diferido:** torch+modelo pesado para RPi4; se deja gancho para offline.",
        "- **TTFB real solo en Piper:** Coqui/Kitten no exponen streaming estable.",
        "- **Fallback XTTS→VITS:** automático si RAM libre < 2.5 GB.",
        "- **Throttling:** bitmask 0x0 = sistema sano; >0 indica bajo voltaje o limitación de frecuencia.",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print(f"  Informe Markdown: {out_path}")


# ---------------------------------------------------------------------------
# EXPORTAR CSV
# ---------------------------------------------------------------------------

def exportar_csv(resultados, out_path):
    """Exporta tabla plana (config, frase, métricas) en CSV."""
    campos = [
        "motor", "nombre", "frase", "tiempo_sintesis_s", "std_tiempo_s",
        "ttfb_s", "warmup_s", "duracion_audio_s", "rtf", "std_rtf",
        "ram_pico_mb", "cpu_pct_pico", "cpu_pct_medio", "wer", "utmos",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        for r in resultados:
            if not r or "frases" not in r:
                continue
            for fd in r["frases"]:
                fila = {
                    "motor": r.get("motor", "?"),
                    "nombre": r.get("nombre", "?"),
                }
                fila.update(fd)
                writer.writerow(fila)
    print(f"  CSV exportado: {out_path}")


# ---------------------------------------------------------------------------
# TABLA CONSOLA
# ---------------------------------------------------------------------------

def imprimir_tabla_consola(resultados):
    print("\n" + "=" * 120)
    print("RESUMEN COMPARATIVO TTS — BENCHMARK EDGE v2")
    print("=" * 120)

    enc = ["Config", "Motor", "Carga(s)", "RAM(MB)", "RTF±std", "TTFB(s)*",
           "P50(s)", "P95(s)", "WER", "CPU%pico", "Throttle", "Tcpu(°C)"]
    filas = []
    for r in resultados:
        if not r or "promedios" not in r:
            continue
        p = r["promedios"]
        rtf_str = f"{p.get('rtf','N/A')}±{p.get('std_rtf','N/A')}"
        ttfb_str = str(p.get("ttfb_s", "N/A")) if p.get("es_streaming") else "-"
        throttle_fin = r.get("throttling_fin")
        throttle_str = f"0x{throttle_fin:x}" if throttle_fin is not None else "N/A"
        filas.append([
            r["nombre"][:30],
            r.get("motor", "?"),
            r.get("tiempo_carga_s", "N/A"),
            p.get("ram_pico_mb", "N/A"),
            rtf_str,
            ttfb_str,
            p.get("p50_sintesis_s", "N/A"),
            p.get("p95_sintesis_s", "N/A"),
            p.get("wer_medio", "N/A"),
            p.get("cpu_pct_pico", "N/A"),
            throttle_str,
            r.get("temp_pico_c", "N/A"),
        ])

    if not filas:
        print("  Sin resultados.")
        return

    anchos = [max(len(str(f[i])) for f in [enc] + filas) for i in range(len(enc))]
    fmt = " | ".join(f"{{:<{a}}}" for a in anchos)
    print(fmt.format(*enc))
    print("-+-".join("-" * a for a in anchos))
    for fila in filas:
        print(fmt.format(*[str(c) for c in fila]))

    print(f"\nN_REPS={N_REPS} ({N_REPS-1} válidas) | RTF<1.0 → más rápido que tiempo real | "
          "* TTFB real solo para Piper (streaming) | WAVs en resultados/<config>/frase_*.wav")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark TTS Edge v2 — RPi4")
    parser.add_argument("--quick", action="store_true",
                        help="Modo rápido: subset de configs y N_REPS=2")
    parser.add_argument("--no-quality", action="store_true",
                        help="Omitir evaluación WER con Whisper (ahorra RAM)")
    parser.add_argument("--output-dir", default=None,
                        help="Directorio de salida (default: <script>/resultados)")
    args = parser.parse_args()

    global N_REPS
    if args.quick:
        N_REPS = 3  # mínimo 2 medidas válidas en modo rápido
        print("[MODO QUICK] N_REPS=3 (2 válidas), configs reducidas")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(base_dir, "resultados")
    graficas_dir = os.path.join(output_dir, "graficas_tts_v2")
    json_path = os.path.join(output_dir, "bench_tts_edgetts_v2.json")
    csv_path = os.path.join(output_dir, "bench_tts_edgetts_v2.csv")
    md_path = os.path.join(output_dir, "informe_tts_edgetts_v2.md")

    os.makedirs(output_dir, exist_ok=True)

    # Buscar modelos primero dentro del proyecto, luego 4 niveles arriba (RPi4)
    models_dir_local = os.path.join(base_dir, "models")
    models_dir_global = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "..", "models"))
    models_dir = models_dir_local if os.path.isdir(models_dir_local) else models_dir_global

    mem = psutil.virtual_memory()
    print("=" * 70)
    print("BENCHMARK TTS EDGE v2 — RPi4")
    print("=" * 70)
    print(f"Hardware: {platform.machine()} | {psutil.cpu_count()} cores | "
          f"RAM: {fmt_mb(mem.total)} MB total, {fmt_mb(mem.available)} MB libre")
    print(f"Frases: {len(FRASES)} | N_REPS: {N_REPS} (1ª=warmup)")
    temp_sys = leer_temperatura_cpu()
    if temp_sys:
        print(f"Temperatura inicial CPU: {temp_sys}°C")
    throttle_sys = leer_throttling_rpi()
    if throttle_sys is not None:
        estado = "OK (0x0)" if throttle_sys == 0 else f"ALERTA 0x{throttle_sys:x}"
        print(f"Throttling RPi4: {estado}")
    print()

    resultados = []

    # --- PIPER ---
    print("=" * 50)
    print("PIPER — Variación params × voces")
    print("=" * 50)
    try:
        r_piper = benchmark_piper(FRASES, output_dir, models_dir, quick=args.quick)
        resultados.extend(r_piper)
    except Exception as e:
        print(f"  [ERROR] Piper benchmark: {e}")

    # --- COQUI ---
    print("\n" + "=" * 50)
    print("COQUI TTS — VITS ES / XTTS-v2")
    print("=" * 50)
    try:
        r_coqui = benchmark_coqui(FRASES, output_dir, quick=args.quick)
        resultados.extend(r_coqui)
    except Exception as e:
        print(f"  [ERROR] Coqui benchmark: {e}")

    # --- KITTEN ---
    print("\n" + "=" * 50)
    print("KITTENTTS — Barrido voz × speed × temperature")
    print("=" * 50)
    try:
        r_kitten = benchmark_kitten(FRASES, output_dir, quick=args.quick)
        resultados.extend(r_kitten)
    except Exception as e:
        print(f"  [ERROR] KittenTTS benchmark: {e}")

    # --- CALIDAD WER ---
    evaluar_calidad(resultados, no_quality=args.no_quality)

    # --- UTMOS ---
    calcular_utmos(resultados)

    # --- TABLA CONSOLA ---
    imprimir_tabla_consola(resultados)

    # --- GRÁFICAS ---
    print("\n" + "=" * 50)
    print("GENERANDO GRÁFICAS")
    print("=" * 50)
    try:
        archivos_graficas = generar_graficas(resultados, graficas_dir)
    except Exception as e:
        print(f"  [WARN] Gráficas: {e}")
        archivos_graficas = []

    # --- INFORME MD ---
    try:
        generar_informe_md(resultados, archivos_graficas, md_path)
    except Exception as e:
        print(f"  [WARN] Informe MD: {e}")

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
            "num_frases": len(FRASES),
            "quick": args.quick,
            "no_quality": args.no_quality,
        },
        "resultados": [r for r in resultados if r is not None],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nResultados JSON: {json_path}")

    # --- CSV ---
    try:
        exportar_csv(resultados, csv_path)
    except Exception as e:
        print(f"  [WARN] CSV: {e}")

    print("\nBenchmark completado.")


if __name__ == "__main__":
    main()
