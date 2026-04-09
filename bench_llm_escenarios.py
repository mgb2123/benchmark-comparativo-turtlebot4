#!/usr/bin/env python3
"""Benchmark LLM avanzado: barrido paramétrico con gráficas comparativas.

Varía max_tokens, n_ctx y n_threads para cada modelo LLM y genera:
  1. Gráficas comparativas (PNG) en resultados/graficas/
  2. Documento detallado con todas las respuestas en resultados/informe_llm_escenarios.txt
  3. JSON completo en resultados/bench_llm_escenarios.json

Escenarios:
  A) max_tokens variable (20, 40, 80) — mide cómo escala la generación
  B) n_ctx variable (128, 256, 512) — impacto del tamaño de contexto
  C) n_threads variable (1, 2, 4) — escalabilidad multi-hilo

Además integra datos de STT y TTS para análisis de pipeline end-to-end.

Uso:
    python3 bench_llm_escenarios.py
    python3 bench_llm_escenarios.py --quick          # 2 prompts por escenario
    python3 bench_llm_escenarios.py --skip-phi2      # omitir Phi-2 (ahorra RAM)
    python3 bench_llm_escenarios.py --models-dir /ruta/a/modelos
"""

import argparse
import datetime
import gc
import json
import os
import platform
import sys
import threading
import time
import textwrap

try:
    import psutil
except ImportError:
    print("[ERROR] psutil no instalado. pip install psutil")
    sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError:
    print("[ERROR] llama-cpp-python no instalado. pip install llama-cpp-python")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Sin display
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib no instalado. Se omitirán gráficas.")

# ─── Configuración base (producción) ───
BASE_N_CTX = 256
BASE_N_THREADS = 2
BASE_MAX_TOKENS = 80
N_GPU_LAYERS = 0
N_BATCH = 512
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.1
SYSTEM_PROMPT = "Eres Ana, un robot asistente. Responde SOLO en español, maximo 5 palabras."

RAM_MINIMA_MB = 400

# ─── Prompts representativos ───
PROMPTS = [
    "qué hora es",
    "cómo te llamas",
    "cuántos objetos ves",
    "llévame a la cocina",
    "qué hay delante de ti",
    "dime un chiste corto",
    "qué tiempo hace hoy",
    "quién te creó",
    "estoy perdido ayúdame",
    "cuál es tu función",
]

# ─── Modelos ───
MODELOS = [
    {
        "nombre": "Qwen2.5-0.5B",
        "nombre_corto": "Qwen-0.5B",
        "archivo": "qwen2.5-0.5b-instruct-q5_k_m.gguf",
        "tipo": "chat",
        "color": "#2196F3",
        "params_b": 0.5,
    },
    {
        "nombre": "TinyLlama-1.1B",
        "nombre_corto": "TinyLlama-1.1B",
        "archivo": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "tipo": "chat",
        "color": "#FF9800",
        "params_b": 1.1,
    },
    {
        "nombre": "Phi-2 2.7B",
        "nombre_corto": "Phi-2-2.7B",
        "archivo": "phi-2.Q4_K_M.gguf",
        "tipo": "completion",
        "color": "#4CAF50",
        "params_b": 2.7,
    },
]

# ─── Escenarios de barrido ───
ESCENARIOS = {
    "max_tokens": {
        "variable": "max_tokens",
        "valores": [10, 20, 40, 60, 80, 120],
        "fijos": {"n_ctx": BASE_N_CTX, "n_threads": BASE_N_THREADS},
        "titulo": "Impacto de max_tokens en rendimiento",
        "xlabel": "max_tokens",
    },
    "n_ctx": {
        "variable": "n_ctx",
        "valores": [128, 192, 256, 384, 512],
        "fijos": {"max_tokens": BASE_MAX_TOKENS, "n_threads": BASE_N_THREADS},
        "titulo": "Impacto de n_ctx (tamaño contexto) en rendimiento",
        "xlabel": "n_ctx (tokens de contexto)",
    },
    "n_threads": {
        "variable": "n_threads",
        "valores": [1, 2, 3, 4],
        "fijos": {"max_tokens": BASE_MAX_TOKENS, "n_ctx": BASE_N_CTX},
        "titulo": "Impacto de n_threads en rendimiento",
        "xlabel": "n_threads (hilos CPU)",
    },
}


class MonitorRAM:
    """Monitorea RSS pico del proceso en un hilo daemon."""

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


def formatear_mb(bytes_val):
    return round(bytes_val / (1024 * 1024), 1)


def ram_disponible_mb():
    return psutil.virtual_memory().available / (1024 * 1024)


def info_hardware():
    mem = psutil.virtual_memory()
    return {
        "plataforma": platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_total_mb": formatear_mb(mem.total),
        "ram_disponible_mb": formatear_mb(mem.available),
        "sistema": platform.platform(),
    }


def inferir_prompt(llm, tipo, prompt, max_tokens):
    """Ejecuta inferencia y retorna (texto, tokens_gen, t_total, t_primer_token)."""
    proc = psutil.Process()
    proc.cpu_percent()

    t0 = time.perf_counter()

    if tipo == "chat":
        respuesta = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_penalty=REPEAT_PENALTY,
        )
        t_total = time.perf_counter() - t0
        texto = respuesta["choices"][0]["message"]["content"].strip()
        tokens_gen = respuesta.get("usage", {}).get("completion_tokens", 0)
    else:
        prompt_completo = (
            f"Instruct: {SYSTEM_PROMPT}\n"
            f"Pregunta: {prompt}\n"
            f"Output:"
        )
        respuesta = llm.create_completion(
            prompt=prompt_completo,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_penalty=REPEAT_PENALTY,
            stop=["\n\n", "Instruct:", "Pregunta:"],
        )
        t_total = time.perf_counter() - t0
        texto = respuesta["choices"][0]["text"].strip()
        tokens_gen = respuesta.get("usage", {}).get("completion_tokens", 0)

    # Latencia primer token (streaming corto)
    t0s = time.perf_counter()
    t_primer_token = None

    if tipo == "chat":
        for chunk in llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=3,
            temperature=TEMPERATURE,
            stream=True,
        ):
            if chunk["choices"][0].get("delta", {}).get("content"):
                t_primer_token = time.perf_counter() - t0s
                break
    else:
        for chunk in llm.create_completion(
            prompt=f"Instruct: {SYSTEM_PROMPT}\nPregunta: {prompt}\nOutput:",
            max_tokens=3,
            temperature=TEMPERATURE,
            stream=True,
        ):
            if chunk["choices"][0].get("text", "").strip():
                t_primer_token = time.perf_counter() - t0s
                break

    if t_primer_token is None:
        t_primer_token = t_total

    cpu_pct = proc.cpu_percent()

    return {
        "texto": texto,
        "tokens_gen": tokens_gen,
        "t_total": round(t_total, 3),
        "t_primer_token": round(t_primer_token, 3),
        "tps": round(tokens_gen / t_total, 2) if t_total > 0 and tokens_gen > 0 else 0,
        "cpu_pct": round(cpu_pct, 1),
    }


def ejecutar_escenario(modelo_info, models_dir, prompts, escenario_cfg):
    """Ejecuta un escenario completo para un modelo, variando el parámetro indicado."""
    ruta = os.path.join(models_dir, modelo_info["archivo"])

    if not os.path.exists(ruta):
        print(f"    [SKIP] No encontrado: {modelo_info['archivo']}")
        return None

    tam_mb = os.path.getsize(ruta) / (1024 * 1024)
    ram_libre = ram_disponible_mb()
    if ram_libre < tam_mb + RAM_MINIMA_MB:
        print(f"    [SKIP] RAM insuficiente ({ram_libre:.0f} MB libre, "
              f"necesita ~{tam_mb + RAM_MINIMA_MB:.0f} MB)")
        return None

    variable = escenario_cfg["variable"]
    valores = escenario_cfg["valores"]
    fijos = escenario_cfg["fijos"]

    resultado_modelo = {
        "modelo": modelo_info["nombre"],
        "nombre_corto": modelo_info["nombre_corto"],
        "archivo_mb": round(tam_mb, 1),
        "puntos": [],
    }

    for val in valores:
        # Construir parámetros para esta iteración
        params = dict(fijos)
        params[variable] = val

        n_ctx = params.get("n_ctx", BASE_N_CTX)
        n_threads = params.get("n_threads", BASE_N_THREADS)
        max_tokens = params.get("max_tokens", BASE_MAX_TOKENS)

        # Asegurar que max_tokens < n_ctx
        if max_tokens >= n_ctx:
            max_tokens = n_ctx - 32
            if max_tokens <= 0:
                print(f"    [SKIP] {variable}={val}: max_tokens no cabe en n_ctx")
                continue

        print(f"    {variable}={val} (n_ctx={n_ctx}, threads={n_threads}, "
              f"max_tok={max_tokens})")

        # Cargar modelo con estos parámetros
        monitor = MonitorRAM()
        monitor.iniciar()

        t0 = time.perf_counter()
        try:
            llm = Llama(
                model_path=ruta,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=N_GPU_LAYERS,
                n_batch=N_BATCH,
                verbose=False,
            )
        except Exception as e:
            monitor.detener()
            print(f"      [ERROR carga] {e}")
            continue

        t_carga = time.perf_counter() - t0
        ram_carga = monitor.detener()

        # Inferir cada prompt
        resultados_prompts = []
        for prompt in prompts:
            monitor = MonitorRAM()
            monitor.iniciar()

            try:
                res = inferir_prompt(llm, modelo_info["tipo"], prompt, max_tokens)
                ram_pico = monitor.detener()
                res["ram_pico_mb"] = formatear_mb(ram_pico)
                res["prompt"] = prompt
                resultados_prompts.append(res)
                print(f"      \"{prompt[:25]}\" → {res['tps']} tok/s | "
                      f"{res['t_total']}s | \"{res['texto'][:40]}\"")
            except Exception as e:
                monitor.detener()
                print(f"      [ERROR] {prompt[:25]}: {e}")
                resultados_prompts.append({
                    "prompt": prompt, "error": str(e),
                    "texto": "", "tokens_gen": 0, "t_total": 0,
                    "t_primer_token": 0, "tps": 0, "cpu_pct": 0,
                    "ram_pico_mb": 0,
                })

        # Liberar modelo
        del llm
        gc.collect()
        time.sleep(0.5)

        # Calcular promedios del punto
        validos = [r for r in resultados_prompts if "error" not in r]
        if validos:
            punto = {
                "valor": val,
                "params": {"n_ctx": n_ctx, "n_threads": n_threads, "max_tokens": max_tokens},
                "tiempo_carga_s": round(t_carga, 2),
                "ram_carga_mb": formatear_mb(ram_carga),
                "promedio_tps": round(sum(r["tps"] for r in validos) / len(validos), 2),
                "promedio_t_total": round(sum(r["t_total"] for r in validos) / len(validos), 3),
                "promedio_t_primer_token": round(
                    sum(r["t_primer_token"] for r in validos) / len(validos), 3),
                "promedio_cpu_pct": round(sum(r["cpu_pct"] for r in validos) / len(validos), 1),
                "ram_pico_mb": max(r["ram_pico_mb"] for r in validos),
                "prompts": resultados_prompts,
            }
        else:
            punto = {
                "valor": val,
                "params": {"n_ctx": n_ctx, "n_threads": n_threads, "max_tokens": max_tokens},
                "tiempo_carga_s": round(t_carga, 2),
                "ram_carga_mb": formatear_mb(ram_carga),
                "promedio_tps": 0, "promedio_t_total": 0,
                "promedio_t_primer_token": 0, "promedio_cpu_pct": 0,
                "ram_pico_mb": 0,
                "prompts": resultados_prompts,
            }

        resultado_modelo["puntos"].append(punto)

    return resultado_modelo


# ─── Gráficas ───

def generar_graficas(datos_escenarios, graficas_dir):
    """Genera gráficas comparativas para cada escenario."""
    if not HAS_MPL:
        print("[WARN] Sin matplotlib, no se generan gráficas.")
        return []

    os.makedirs(graficas_dir, exist_ok=True)
    archivos = []

    for esc_nombre, esc_cfg in ESCENARIOS.items():
        datos_esc = datos_escenarios.get(esc_nombre, [])
        modelos_con_datos = [d for d in datos_esc if d and d["puntos"]]

        if not modelos_con_datos:
            continue

        # Métricas a graficar
        metricas = [
            ("promedio_tps", "Tokens/segundo", "tok/s"),
            ("promedio_t_total", "Tiempo total generación (s)", "segundos"),
            ("promedio_t_primer_token", "Latencia primer token (s)", "segundos"),
            ("ram_pico_mb", "RAM pico (MB)", "MB"),
            ("promedio_cpu_pct", "Uso CPU (%)", "%"),
            ("tiempo_carga_s", "Tiempo de carga modelo (s)", "segundos"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(esc_cfg["titulo"], fontsize=14, fontweight='bold', y=0.98)
        axes = axes.flatten()

        for idx, (key, titulo_met, unidad) in enumerate(metricas):
            ax = axes[idx]

            for modelo_data in modelos_con_datos:
                nombre = modelo_data["nombre_corto"]
                # Buscar color del modelo
                color = "#333333"
                for m in MODELOS:
                    if m["nombre_corto"] == nombre:
                        color = m["color"]
                        break

                valores_x = [p["valor"] for p in modelo_data["puntos"]]
                valores_y = [p[key] for p in modelo_data["puntos"]]

                ax.plot(valores_x, valores_y, 'o-', label=nombre,
                        color=color, linewidth=2, markersize=8)

                # Anotar valores
                for x, y in zip(valores_x, valores_y):
                    ax.annotate(f'{y}', (x, y), textcoords="offset points",
                                xytext=(0, 8), ha='center', fontsize=7,
                                color=color)

            ax.set_xlabel(esc_cfg["xlabel"], fontsize=9)
            ax.set_ylabel(f"{titulo_met} ({unidad})", fontsize=9)
            ax.set_title(titulo_met, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(esc_cfg["valores"])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        ruta_png = os.path.join(graficas_dir, f"escenario_{esc_nombre}.png")
        fig.savefig(ruta_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        archivos.append(ruta_png)
        print(f"  Gráfica guardada: {ruta_png}")

    # ─── Gráfica resumen: potencia modelo (params) vs tok/s vs latencia ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Potencia del modelo vs Rendimiento (config base)", fontsize=13,
                 fontweight='bold')

    # Recoger datos de la config base (max_tokens escenario, valor=80)
    datos_base = datos_escenarios.get("max_tokens", [])
    modelos_base = [d for d in datos_base if d and d["puntos"]]

    for modelo_data in modelos_base:
        nombre = modelo_data["nombre_corto"]
        color = "#333"
        params_b = 0
        for m in MODELOS:
            if m["nombre_corto"] == nombre:
                color = m["color"]
                params_b = m["params_b"]
                break

        # Buscar el punto con max_tokens=80 (o el más cercano a config base)
        punto_base = None
        for p in modelo_data["puntos"]:
            if p["params"]["max_tokens"] == 80 or p["valor"] == 80:
                punto_base = p
                break
        if not punto_base and modelo_data["puntos"]:
            punto_base = modelo_data["puntos"][-1]

        if punto_base:
            ax1.bar(nombre, punto_base["promedio_tps"], color=color, alpha=0.85,
                    edgecolor='black', linewidth=0.5)
            ax1.text(nombre, punto_base["promedio_tps"] + 0.1,
                     f'{punto_base["promedio_tps"]} tok/s\n({params_b}B params)',
                     ha='center', fontsize=8)

            ax2.bar(nombre, punto_base["promedio_t_total"], color=color, alpha=0.85,
                    edgecolor='black', linewidth=0.5)
            ax2.text(nombre, punto_base["promedio_t_total"] + 0.2,
                     f'{punto_base["promedio_t_total"]}s',
                     ha='center', fontsize=8)

    ax1.set_ylabel("Tokens/segundo")
    ax1.set_title("Velocidad de generación")
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.set_ylabel("Tiempo total (s)")
    ax2.set_title("Tiempo medio de respuesta")
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    ruta_resumen = os.path.join(graficas_dir, "resumen_potencia_vs_rendimiento.png")
    fig.savefig(ruta_resumen, dpi=150, bbox_inches='tight')
    plt.close(fig)
    archivos.append(ruta_resumen)
    print(f"  Gráfica guardada: {ruta_resumen}")

    return archivos


def generar_grafica_pipeline(datos_pipeline, graficas_dir):
    """Gráfica de pipeline end-to-end: STT + LLM + TTS."""
    if not HAS_MPL or not datos_pipeline:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Pipeline End-to-End: STT → LLM → TTS", fontsize=13,
                 fontweight='bold')

    nombres = [d["nombre"] for d in datos_pipeline]
    stt_times = [d["stt_s"] for d in datos_pipeline]
    llm_times = [d["llm_s"] for d in datos_pipeline]
    tts_times = [d["tts_s"] for d in datos_pipeline]
    totales = [d["total_s"] for d in datos_pipeline]

    # Stacked bar chart
    x = range(len(nombres))
    bars1 = ax1.bar(x, stt_times, label='STT', color='#2196F3', alpha=0.85)
    bars2 = ax1.bar(x, llm_times, bottom=stt_times, label='LLM', color='#FF9800',
                    alpha=0.85)
    bottoms_tts = [s + l for s, l in zip(stt_times, llm_times)]
    bars3 = ax1.bar(x, tts_times, bottom=bottoms_tts, label='TTS', color='#4CAF50',
                    alpha=0.85)

    # Anotar total
    for i, total in enumerate(totales):
        ax1.text(i, total + 0.2, f'{total:.1f}s', ha='center', fontsize=9,
                 fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres, rotation=15, fontsize=8)
    ax1.set_ylabel("Tiempo (segundos)")
    ax1.set_title("Latencia total por combinación")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # RAM stacked
    stt_ram = [d["stt_ram_mb"] for d in datos_pipeline]
    llm_ram = [d["llm_ram_mb"] for d in datos_pipeline]
    tts_ram = [d["tts_ram_mb"] for d in datos_pipeline]

    ax2.bar(x, stt_ram, label='STT', color='#2196F3', alpha=0.85)
    ax2.bar(x, llm_ram, bottom=stt_ram, label='LLM', color='#FF9800', alpha=0.85)
    bottoms_tts_ram = [s + l for s, l in zip(stt_ram, llm_ram)]
    ax2.bar(x, tts_ram, bottom=bottoms_tts_ram, label='TTS', color='#4CAF50',
            alpha=0.85)

    for i in range(len(nombres)):
        total_ram = stt_ram[i] + llm_ram[i] + tts_ram[i]
        ax2.text(i, total_ram + 10, f'{total_ram:.0f} MB', ha='center', fontsize=9,
                 fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(nombres, rotation=15, fontsize=8)
    ax2.set_ylabel("RAM pico (MB)")
    ax2.set_title("Consumo RAM por componente (estimado)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    ruta = os.path.join(graficas_dir, "pipeline_end_to_end.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Gráfica guardada: {ruta}")
    return ruta


# ─── Informe de texto ───

def generar_informe(datos_escenarios, datos_pipeline, hw, ruta_informe):
    """Genera documento de texto con todas las respuestas y análisis."""
    lineas = []

    def w(texto=""):
        lineas.append(texto)

    w("=" * 80)
    w("INFORME DETALLADO: COMPARATIVA LLM — BARRIDO PARAMÉTRICO")
    w(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w("=" * 80)
    w()
    w("HARDWARE")
    w("-" * 40)
    w(f"  Plataforma:  {hw['plataforma']}")
    w(f"  CPU cores:   {hw['cpu_cores']}")
    w(f"  RAM total:   {hw['ram_total_mb']} MB")
    w(f"  RAM libre:   {hw['ram_disponible_mb']} MB")
    w(f"  Sistema:     {hw['sistema']}")
    w()

    for esc_nombre, esc_cfg in ESCENARIOS.items():
        datos_esc = datos_escenarios.get(esc_nombre, [])
        modelos_con_datos = [d for d in datos_esc if d and d["puntos"]]

        if not modelos_con_datos:
            continue

        w()
        w("=" * 80)
        w(f"ESCENARIO: {esc_cfg['titulo']}")
        w(f"Variable: {esc_cfg['variable']} = {esc_cfg['valores']}")
        w(f"Parámetros fijos: {esc_cfg['fijos']}")
        w("=" * 80)

        for modelo_data in modelos_con_datos:
            w()
            w(f"  ┌─ MODELO: {modelo_data['modelo']} ({modelo_data['archivo_mb']} MB)")
            w(f"  │")

            for punto in modelo_data["puntos"]:
                w(f"  ├─ {esc_cfg['variable']} = {punto['valor']}")
                w(f"  │  Params: n_ctx={punto['params']['n_ctx']}, "
                  f"threads={punto['params']['n_threads']}, "
                  f"max_tokens={punto['params']['max_tokens']}")
                w(f"  │  Carga: {punto['tiempo_carga_s']}s | "
                  f"RAM carga: {punto['ram_carga_mb']} MB | "
                  f"RAM pico: {punto['ram_pico_mb']} MB")
                w(f"  │  Promedios: {punto['promedio_tps']} tok/s | "
                  f"{punto['promedio_t_total']}s total | "
                  f"{punto['promedio_t_primer_token']}s 1er token | "
                  f"CPU {punto['promedio_cpu_pct']}%")
                w(f"  │")
                w(f"  │  Respuestas:")

                for r in punto["prompts"]:
                    if "error" in r:
                        w(f"  │    P: \"{r['prompt']}\"")
                        w(f"  │    → [ERROR] {r['error']}")
                    else:
                        w(f"  │    P: \"{r['prompt']}\"")
                        w(f"  │    R: \"{r['texto']}\"")
                        w(f"  │       ({r['tokens_gen']} tok, {r['t_total']}s, "
                          f"{r['tps']} tok/s)")
                    w(f"  │")

            w(f"  └─────────────────────────────────────")

    # Tabla resumen por escenario
    for esc_nombre, esc_cfg in ESCENARIOS.items():
        datos_esc = datos_escenarios.get(esc_nombre, [])
        modelos_con_datos = [d for d in datos_esc if d and d["puntos"]]
        if not modelos_con_datos:
            continue

        w()
        w("─" * 80)
        w(f"TABLA RESUMEN: {esc_cfg['variable']}")
        w("─" * 80)

        # Encabezado
        enc = f"{'Modelo':<20} {'Valor':<8} {'Tok/s':<8} {'Total(s)':<10} "
        enc += f"{'1erTok(s)':<10} {'RAM(MB)':<8} {'CPU%':<8} {'Carga(s)':<8}"
        w(enc)
        w("-" * len(enc))

        for modelo_data in modelos_con_datos:
            for punto in modelo_data["puntos"]:
                fila = (f"{modelo_data['nombre_corto']:<20} "
                        f"{punto['valor']:<8} "
                        f"{punto['promedio_tps']:<8} "
                        f"{punto['promedio_t_total']:<10} "
                        f"{punto['promedio_t_primer_token']:<10} "
                        f"{punto['ram_pico_mb']:<8} "
                        f"{punto['promedio_cpu_pct']:<8} "
                        f"{punto['tiempo_carga_s']:<8}")
                w(fila)
            w()

    # Pipeline end-to-end
    if datos_pipeline:
        w()
        w("=" * 80)
        w("ANÁLISIS PIPELINE END-TO-END: STT → LLM → TTS")
        w("=" * 80)
        w()
        w("Combinaciones analizadas (tiempos estimados de cada etapa):")
        w()

        enc = (f"{'Combinación':<35} {'STT(s)':<8} {'LLM(s)':<8} {'TTS(s)':<8} "
               f"{'Total(s)':<9} {'RAM tot':<8}")
        w(enc)
        w("-" * len(enc))

        for d in datos_pipeline:
            total_ram = d["stt_ram_mb"] + d["llm_ram_mb"] + d["tts_ram_mb"]
            fila = (f"{d['nombre']:<35} {d['stt_s']:<8.2f} {d['llm_s']:<8.2f} "
                    f"{d['tts_s']:<8.2f} {d['total_s']:<9.2f} {total_ram:<8.0f}")
            w(fila)

        w()
        w("NOTA: Los tiempos STT y TTS son promedios de los benchmarks individuales.")
        w("      El tiempo LLM corresponde a max_tokens=80, n_ctx=256, n_threads=2.")
        w("      RAM total es la suma de picos individuales (en la práctica pueden")
        w("      coexistir parcialmente en memoria compartida).")

    # Conclusiones automáticas
    w()
    w("=" * 80)
    w("OBSERVACIONES AUTOMÁTICAS")
    w("=" * 80)
    w()

    # Encontrar el modelo más rápido en config base
    datos_base = datos_escenarios.get("max_tokens", [])
    if datos_base:
        mejor_tps = 0
        mejor_modelo = ""
        mejor_latencia = 999
        mejor_lat_modelo = ""

        for d in datos_base:
            if not d or not d["puntos"]:
                continue
            for p in d["puntos"]:
                if p["valor"] == 80 or p == d["puntos"][-1]:
                    if p["promedio_tps"] > mejor_tps:
                        mejor_tps = p["promedio_tps"]
                        mejor_modelo = d["nombre_corto"]
                    if p["promedio_t_primer_token"] < mejor_latencia:
                        mejor_latencia = p["promedio_t_primer_token"]
                        mejor_lat_modelo = d["nombre_corto"]
                    break

        w(f"  - Modelo más rápido (tok/s): {mejor_modelo} ({mejor_tps} tok/s)")
        w(f"  - Menor latencia primer token: {mejor_lat_modelo} ({mejor_latencia}s)")

    datos_threads = datos_escenarios.get("n_threads", [])
    if datos_threads:
        w()
        w("  Escalabilidad con hilos:")
        for d in datos_threads:
            if not d or not d["puntos"]:
                continue
            tps_vals = {p["valor"]: p["promedio_tps"] for p in d["puntos"]}
            if 1 in tps_vals and 4 in tps_vals and tps_vals[1] > 0:
                speedup = tps_vals[4] / tps_vals[1]
                w(f"    {d['nombre_corto']}: {tps_vals[1]} → {tps_vals[4]} tok/s "
                  f"(×{speedup:.2f} con 4 hilos vs 1)")

    if datos_pipeline:
        w()
        mejor_pipe = min(datos_pipeline, key=lambda x: x["total_s"])
        w(f"  - Pipeline más rápido: {mejor_pipe['nombre']} ({mejor_pipe['total_s']:.1f}s total)")

    w()
    w("─" * 80)
    w("Fin del informe")
    w(f"Generado: {datetime.datetime.now().isoformat()}")

    os.makedirs(os.path.dirname(ruta_informe), exist_ok=True)
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    print(f"  Informe guardado: {ruta_informe}")


# ─── Pipeline end-to-end ───

def calcular_pipeline(datos_escenarios, resultados_dir):
    """Combina datos de STT, LLM y TTS para análisis de pipeline."""
    # Cargar resultados previos de STT y TTS
    stt_json = os.path.join(resultados_dir, "bench_stt.json")
    tts_json = os.path.join(resultados_dir, "bench_tts.json")

    stt_data = None
    tts_data = None

    if os.path.exists(stt_json):
        with open(stt_json, "r") as f:
            stt_data = json.load(f)
        print("  Datos STT cargados desde bench_stt.json")
    else:
        print("  [INFO] bench_stt.json no encontrado, usando estimaciones")

    if os.path.exists(tts_json):
        with open(tts_json, "r") as f:
            tts_data = json.load(f)
        print("  Datos TTS cargados desde bench_tts.json")
    else:
        print("  [INFO] bench_tts.json no encontrado, usando estimaciones")

    # Extraer promedios de STT
    stt_opciones = []
    if stt_data and "resultados" in stt_data:
        for r in stt_data["resultados"]:
            prom = r.get("promedios", {})
            stt_opciones.append({
                "nombre": r["motor"],
                "tiempo_s": prom.get("tiempo_transcripcion_s", 2.0),
                "ram_mb": prom.get("ram_pico_mb", 200),
            })
    else:
        stt_opciones = [
            {"nombre": "faster-whisper", "tiempo_s": 1.5, "ram_mb": 180},
        ]

    # Extraer promedios de TTS
    tts_opciones = []
    if tts_data and "resultados" in tts_data:
        for r in tts_data["resultados"]:
            prom = r.get("promedios", {})
            tts_opciones.append({
                "nombre": r["motor"],
                "tiempo_s": prom.get("tiempo_sintesis_s", 0.5),
                "ram_mb": prom.get("ram_pico_mb", 50),
            })
    else:
        tts_opciones = [
            {"nombre": "pyttsx3", "tiempo_s": 0.3, "ram_mb": 30},
        ]

    # Extraer LLM de escenario max_tokens (valor=80, config base)
    llm_opciones = []
    datos_base = datos_escenarios.get("max_tokens", [])
    for d in datos_base:
        if not d or not d["puntos"]:
            continue
        for p in d["puntos"]:
            if p["valor"] == 80 or p == d["puntos"][-1]:
                llm_opciones.append({
                    "nombre": d["nombre_corto"],
                    "tiempo_s": p["promedio_t_total"],
                    "ram_mb": p["ram_pico_mb"],
                })
                break

    # Generar combinaciones
    pipeline_datos = []
    for stt in stt_opciones:
        for llm in llm_opciones:
            for tts in tts_opciones:
                nombre = f"{stt['nombre']} + {llm['nombre']} + {tts['nombre']}"
                total = stt["tiempo_s"] + llm["tiempo_s"] + tts["tiempo_s"]
                pipeline_datos.append({
                    "nombre": nombre,
                    "stt": stt["nombre"],
                    "llm": llm["nombre"],
                    "tts": tts["nombre"],
                    "stt_s": stt["tiempo_s"],
                    "llm_s": llm["tiempo_s"],
                    "tts_s": tts["tiempo_s"],
                    "total_s": round(total, 2),
                    "stt_ram_mb": stt["ram_mb"],
                    "llm_ram_mb": llm["ram_mb"],
                    "tts_ram_mb": tts["ram_mb"],
                })

    # Ordenar por tiempo total
    pipeline_datos.sort(key=lambda x: x["total_s"])
    return pipeline_datos


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM avanzado con barrido paramétrico y gráficas"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Modo rápido (2 prompts por escenario)")
    parser.add_argument("--skip-phi2", action="store_true",
                        help="Omitir Phi-2 2.7B (ahorra RAM y tiempo)")
    parser.add_argument("--models-dir",
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "..", "..", "..", "models"),
                        help="Directorio de modelos GGUF")
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(__file__), "resultados"),
                        help="Directorio de salida")
    args = parser.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    output_dir = os.path.abspath(args.output_dir)
    graficas_dir = os.path.join(output_dir, "graficas")

    prompts = PROMPTS[:2] if args.quick else PROMPTS
    modelos = [m for m in MODELOS if not (args.skip_phi2 and "Phi" in m["nombre"])]

    hw = info_hardware()

    print("=" * 65)
    print("BENCHMARK LLM AVANZADO — BARRIDO PARAMÉTRICO")
    print("=" * 65)
    print(f"Modelos: {', '.join(m['nombre'] for m in modelos)}")
    print(f"Prompts/escenario: {len(prompts)}")
    print(f"Escenarios: {', '.join(ESCENARIOS.keys())}")
    print(f"Hardware: {hw['plataforma']} | {hw['cpu_cores']} cores | "
          f"RAM: {hw['ram_total_mb']} MB total, {hw['ram_disponible_mb']} MB libre")
    print(f"Modo: {'rápido' if args.quick else 'completo'}")

    total_runs = len(modelos) * sum(len(e["valores"]) for e in ESCENARIOS.values())
    print(f"Total de combinaciones modelo×parámetro: {total_runs}")
    print()

    # Ejecutar escenarios
    datos_escenarios = {}

    for esc_nombre, esc_cfg in ESCENARIOS.items():
        print(f"\n{'='*65}")
        print(f"ESCENARIO: {esc_cfg['titulo']}")
        print(f"Variable: {esc_cfg['variable']} = {esc_cfg['valores']}")
        print(f"{'='*65}")

        datos_esc = []
        for modelo in modelos:
            print(f"\n  ── {modelo['nombre']} ──")
            resultado = ejecutar_escenario(modelo, models_dir, prompts, esc_cfg)
            datos_esc.append(resultado)

        datos_escenarios[esc_nombre] = datos_esc

    # Pipeline end-to-end
    print(f"\n{'='*65}")
    print("ANÁLISIS PIPELINE END-TO-END")
    print(f"{'='*65}")
    datos_pipeline = calcular_pipeline(datos_escenarios, output_dir)

    if datos_pipeline:
        print("\n  Combinaciones (ordenadas por latencia total):")
        for d in datos_pipeline:
            print(f"    {d['nombre']}: {d['total_s']}s "
                  f"(STT={d['stt_s']:.1f} + LLM={d['llm_s']:.1f} + TTS={d['tts_s']:.1f})")

    # Generar gráficas
    print(f"\n{'='*65}")
    print("GENERANDO GRÁFICAS")
    print(f"{'='*65}")
    archivos_graficas = generar_graficas(datos_escenarios, graficas_dir)
    ruta_pipeline_png = generar_grafica_pipeline(datos_pipeline, graficas_dir)
    if ruta_pipeline_png:
        archivos_graficas.append(ruta_pipeline_png)

    # Generar informe de texto
    print(f"\n{'='*65}")
    print("GENERANDO INFORME")
    print(f"{'='*65}")
    ruta_informe = os.path.join(output_dir, "informe_llm_escenarios.txt")
    generar_informe(datos_escenarios, datos_pipeline, hw, ruta_informe)

    # Guardar JSON completo
    salida = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": hw,
        "modo": "quick" if args.quick else "completo",
        "num_prompts": len(prompts),
        "escenarios": {},
        "pipeline": datos_pipeline,
    }

    for esc_nombre, datos_esc in datos_escenarios.items():
        salida["escenarios"][esc_nombre] = {
            "config": ESCENARIOS[esc_nombre],
            "resultados": [d for d in datos_esc if d is not None],
        }

    ruta_json = os.path.join(output_dir, "bench_llm_escenarios.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*65}")
    print("COMPLETADO")
    print(f"{'='*65}")
    print(f"  JSON:     {ruta_json}")
    print(f"  Informe:  {ruta_informe}")
    print(f"  Gráficas: {graficas_dir}/ ({len(archivos_graficas)} archivos)")
    print()


if __name__ == "__main__":
    main()
