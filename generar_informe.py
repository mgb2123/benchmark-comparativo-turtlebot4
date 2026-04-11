#!/usr/bin/env python3
"""Genera gráficos PNG de los benchmarks TTS y STT."""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), "resultados")


def cargar_json(nombre):
    ruta = os.path.join(RESULTADOS_DIR, nombre)
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# GRÁFICOS TTS
# ============================================================
def graficos_tts(datos):
    resultados = datos["resultados"]

    piper = [r for r in resultados if r["motor"] == "piper"]
    coqui = [r for r in resultados if r["motor"] == "coqui"]

    COLOR_PIPER = "#2196F3"
    COLOR_COQUI = "#FF5722"

    def color_motor(r):
        if r["motor"] == "piper":
            return COLOR_PIPER
        if r["motor"] == "coqui":
            return COLOR_COQUI
        return "#9E9E9E"

    # --- Fig 1: Comparativa general RTF + RAM ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Benchmark TTS — RPi4 (aarch64, 4GB RAM)", fontsize=14, fontweight="bold")

    nombres = [r["nombre"].replace("piper_", "piper\n").replace("coqui_", "coqui\n")
               for r in resultados]
    colores = [color_motor(r) for r in resultados]

    # RTF con barras de error (std)
    ax = axes[0]
    rtfs = [r["promedios"]["rtf"] for r in resultados]
    stds_rtf = [r["promedios"].get("std_rtf", 0) for r in resultados]
    bars = ax.bar(range(len(nombres)), rtfs, color=colores,
                  yerr=stds_rtf, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, fontsize=7, rotation=0)
    ax.set_ylabel("RTF (menor = mejor)")
    ax.set_title("Real-Time Factor")
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Tiempo real")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, rtfs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', fontsize=7)

    # RAM
    ax = axes[1]
    rams = [r["promedios"]["ram_pico_mb"] for r in resultados]
    bars = ax.bar(range(len(nombres)), rams, color=colores)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, fontsize=7)
    ax.set_ylabel("RAM pico (MB)")
    ax.set_title("Consumo de memoria")
    for bar, val in zip(bars, rams):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}", ha='center', fontsize=7)

    # Tiempo de síntesis con barras de error
    ax = axes[2]
    tiempos = [r["promedios"]["tiempo_sintesis_s"] for r in resultados]
    stds_t = [r["promedios"].get("std_tiempo_s", 0) for r in resultados]
    bars = ax.bar(range(len(nombres)), tiempos, color=colores,
                  yerr=stds_t, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, fontsize=7)
    ax.set_ylabel("Tiempo síntesis (s)")
    ax.set_title("Latencia de síntesis (media ± std)")
    for bar, val in zip(bars, tiempos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha='center', fontsize=7)

    # Leyenda manual
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLOR_PIPER, label='Piper'),
                       Patch(facecolor=COLOR_COQUI, label='Coqui TTS')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    ruta = os.path.join(RESULTADOS_DIR, "tts_comparativa_general.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  [OK] {ruta}")

    # --- Fig 2: Piper parámetros ---
    if piper:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Piper TTS — Impacto de parámetros", fontsize=14, fontweight="bold")

        nombres_p = [r["nombre"].replace("piper_", "") for r in piper]
        rtfs_p = [r["promedios"]["rtf"] for r in piper]
        dur_p = [r["promedios"]["duracion_audio_s"] for r in piper]
        sint_p = [r["promedios"]["tiempo_sintesis_s"] for r in piper]

        ax = axes[0]
        x = np.arange(len(nombres_p))
        bars = ax.bar(x, rtfs_p, color='#2196F3')
        ax.set_xticks(x)
        ax.set_xticklabels(nombres_p)
        ax.set_ylabel("RTF")
        ax.set_title("RTF por configuración")
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Tiempo real")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, rtfs_p):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha='center', fontsize=9)

        ax = axes[1]
        width = 0.35
        bars1 = ax.bar(x - width/2, sint_p, width, label='Síntesis', color='#FF5722')
        bars2 = ax.bar(x + width/2, dur_p, width, label='Dur. Audio', color='#8BC34A')
        ax.set_xticks(x)
        ax.set_xticklabels(nombres_p)
        ax.set_ylabel("Tiempo (s)")
        ax.set_title("Síntesis vs Duración audio")
        ax.legend()

        plt.tight_layout()
        ruta = os.path.join(RESULTADOS_DIR, "tts_piper_params.png")
        plt.savefig(ruta, dpi=150)
        plt.close()
        print(f"  [OK] {ruta}")

    # --- Fig 3: Piper vs Coqui — comparativa directa ---
    if piper and coqui:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Piper vs Coqui TTS — Comparativa directa", fontsize=14, fontweight="bold")

        # Mejor Piper (menor RTF) vs cada Coqui
        mejor_piper = min(piper, key=lambda r: r["promedios"]["rtf"])
        comparados = [mejor_piper] + coqui
        nombres_c = [r["nombre"].replace("piper_", "Piper\n").replace("coqui_", "Coqui\n")
                     for r in comparados]
        colores_c = [COLOR_PIPER] + [COLOR_COQUI] * len(coqui)

        ax = axes[0]
        rtfs_c = [r["promedios"]["rtf"] for r in comparados]
        stds_c = [r["promedios"].get("std_rtf", 0) for r in comparados]
        ax.bar(range(len(nombres_c)), rtfs_c, color=colores_c,
               yerr=stds_c, capsize=5)
        ax.set_xticks(range(len(nombres_c)))
        ax.set_xticklabels(nombres_c, fontsize=9)
        ax.set_ylabel("RTF (menor = mejor)")
        ax.set_title("Real-Time Factor")
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Tiempo real")
        ax.legend(fontsize=8)

        ax = axes[1]
        rams_c = [r["promedios"]["ram_pico_mb"] for r in comparados]
        ax.bar(range(len(nombres_c)), rams_c, color=colores_c)
        ax.set_xticks(range(len(nombres_c)))
        ax.set_xticklabels(nombres_c, fontsize=9)
        ax.set_ylabel("RAM pico (MB)")
        ax.set_title("Consumo de memoria")

        plt.tight_layout()
        ruta = os.path.join(RESULTADOS_DIR, "tts_piper_vs_coqui.png")
        plt.savefig(ruta, dpi=150)
        plt.close()
        print(f"  [OK] {ruta}")


# ============================================================
# GRÁFICOS STT
# ============================================================
def graficos_stt(datos):
    resultados = [r for r in datos["resultados"] if "skip" not in r and "error" not in r]

    whisper = [r for r in resultados if r["motor"] == "faster-whisper"]
    vosk = [r for r in resultados if r["motor"] == "vosk"]

    todos = whisper + vosk

    # --- Fig 1: Comparativa general WER + RTF + RAM ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Benchmark STT — RPi4 (aarch64, 4GB RAM)", fontsize=14, fontweight="bold")

    nombres = []
    for r in todos:
        n = r["nombre"]
        n = n.replace("whisper-", "w-").replace("_int8", "\nint8").replace("_float32", "\nfp32")
        n = n.replace("_beam", "\nbeam").replace("_best", " best")
        nombres.append(n)

    colores = ["#FF9800" if r["motor"] == "faster-whisper" else "#9C27B0" for r in todos]

    # WER
    ax = axes[0]
    wers = [r["promedios"]["wer"] for r in todos]
    bars = ax.bar(range(len(nombres)), wers, color=colores)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, fontsize=6, rotation=0)
    ax.set_ylabel("WER (menor = mejor)")
    ax.set_title("Word Error Rate")
    for bar, val in zip(bars, wers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha='center', fontsize=6)

    # RTF con barras de error (std)
    ax = axes[1]
    rtfs = [r["promedios"]["rtf"] for r in todos]
    stds_rtf = [r["promedios"].get("std_rtf", 0) for r in todos]
    bars = ax.bar(range(len(nombres)), rtfs, color=colores,
                  yerr=stds_rtf, capsize=3, error_kw={"elinewidth": 1})
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, fontsize=6)
    ax.set_ylabel("RTF (menor = mejor)")
    ax.set_title("Real-Time Factor (media ± std)")
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Tiempo real")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, rtfs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha='center', fontsize=6)

    # RAM
    ax = axes[2]
    rams = [r["promedios"]["ram_pico_mb"] for r in todos]
    bars = ax.bar(range(len(nombres)), rams, color=colores)
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres, fontsize=6)
    ax.set_ylabel("RAM pico (MB)")
    ax.set_title("Consumo de memoria")
    for bar, val in zip(bars, rams):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}", ha='center', fontsize=6)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FF9800', label='faster-whisper'),
                       Patch(facecolor='#9C27B0', label='Vosk')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    ruta = os.path.join(RESULTADOS_DIR, "stt_comparativa_general.png")
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  [OK] {ruta}")

    # --- Fig 2: Whisper beam_size impact (tiny int8) ---
    tiny_int8 = [r for r in whisper if "tiny" in r["nombre"] and "int8" in r["nombre"]]
    if len(tiny_int8) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("faster-whisper tiny int8 — Impacto del beam_size", fontsize=14, fontweight="bold")

        beams = [r["config"]["beam_size"] for r in tiny_int8]
        wers_b = [r["promedios"]["wer"] for r in tiny_int8]
        rtfs_b = [r["promedios"]["rtf"] for r in tiny_int8]

        ax = axes[0]
        ax.plot(beams, wers_b, 'o-', color='#FF9800', linewidth=2, markersize=8)
        ax.set_xlabel("beam_size")
        ax.set_ylabel("WER")
        ax.set_title("beam_size vs Precisión (WER)")
        ax.grid(True, alpha=0.3)
        for x, y in zip(beams, wers_b):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), fontsize=10, ha='center')

        ax = axes[1]
        ax.plot(beams, rtfs_b, 'o-', color='#F44336', linewidth=2, markersize=8)
        ax.set_xlabel("beam_size")
        ax.set_ylabel("RTF")
        ax.set_title("beam_size vs Velocidad (RTF)")
        ax.grid(True, alpha=0.3)
        for x, y in zip(beams, rtfs_b):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), fontsize=10, ha='center')

        plt.tight_layout()
        ruta = os.path.join(RESULTADOS_DIR, "stt_whisper_beam_size.png")
        plt.savefig(ruta, dpi=150)
        plt.close()
        print(f"  [OK] {ruta}")

    # --- Fig 3: tiny vs base ---
    tiny_b1 = [r for r in whisper if "tiny" in r["nombre"] and "beam1" in r["nombre"] and "int8" in r["nombre"]]
    base_b1 = [r for r in whisper if "base" in r["nombre"] and "beam1" in r["nombre"] and "int8" in r["nombre"]]
    if tiny_b1 and base_b1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig.suptitle("tiny vs base (int8, beam=1) — Trade-off WER vs RTF", fontsize=14, fontweight="bold")

        for r, color, marker in [(tiny_b1[0], '#FF9800', 'o'), (base_b1[0], '#E91E63', 's')]:
            ax.scatter(r["promedios"]["rtf"], r["promedios"]["wer"],
                      s=r["promedios"]["ram_pico_mb"], color=color, marker=marker,
                      alpha=0.7, edgecolors='black', linewidth=1)
            ax.annotate(r["nombre"].replace("whisper-", ""),
                       (r["promedios"]["rtf"], r["promedios"]["wer"]),
                       textcoords="offset points", xytext=(10, 5), fontsize=10)

        if vosk:
            v = vosk[0]
            ax.scatter(v["promedios"]["rtf"], v["promedios"]["wer"],
                      s=v["promedios"]["ram_pico_mb"], color='#9C27B0', marker='^',
                      alpha=0.7, edgecolors='black', linewidth=1)
            ax.annotate("vosk", (v["promedios"]["rtf"], v["promedios"]["wer"]),
                       textcoords="offset points", xytext=(10, 5), fontsize=10)

        ax.set_xlabel("RTF (menor = más rápido)")
        ax.set_ylabel("WER (menor = más preciso)")
        ax.set_title("Tamaño burbuja = RAM pico (MB)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        ruta = os.path.join(RESULTADOS_DIR, "stt_tradeoff.png")
        plt.savefig(ruta, dpi=150)
        plt.close()
        print(f"  [OK] {ruta}")

    # --- Fig 4: VAD impact ---
    vad_results = [r for r in whisper if "promedios_vad" in r]
    if vad_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Efecto del VAD filter (beam_size=1)", fontsize=14, fontweight="bold")

        nombres_v = [r["nombre"].replace("whisper-", "") for r in vad_results]
        x = np.arange(len(nombres_v))
        width = 0.35

        wer_sin = [r["promedios"]["wer"] for r in vad_results]
        wer_con = [r["promedios_vad"]["wer"] for r in vad_results]

        ax = axes[0]
        ax.bar(x - width/2, wer_sin, width, label="Sin VAD", color="#FF9800")
        ax.bar(x + width/2, wer_con, width, label="Con VAD", color="#4CAF50")
        ax.set_xticks(x)
        ax.set_xticklabels(nombres_v, fontsize=8)
        ax.set_ylabel("WER")
        ax.set_title("WER: Sin VAD vs Con VAD")
        ax.legend()

        rtf_sin = [r["promedios"]["rtf"] for r in vad_results]
        rtf_con = [r["promedios_vad"]["rtf"] for r in vad_results]

        ax = axes[1]
        ax.bar(x - width/2, rtf_sin, width, label="Sin VAD", color="#FF9800")
        ax.bar(x + width/2, rtf_con, width, label="Con VAD", color="#4CAF50")
        ax.set_xticks(x)
        ax.set_xticklabels(nombres_v, fontsize=8)
        ax.set_ylabel("RTF")
        ax.set_title("RTF: Sin VAD vs Con VAD")
        ax.legend()

        plt.tight_layout()
        ruta = os.path.join(RESULTADOS_DIR, "stt_vad_impact.png")
        plt.savefig(ruta, dpi=150)
        plt.close()
        print(f"  [OK] {ruta}")


def main():
    print("Generando gráficas de benchmarks...")

    datos_tts = cargar_json("bench_tts_exhaustivo.json")
    datos_stt = cargar_json("bench_stt_exhaustivo.json")

    if not datos_tts and not datos_stt:
        print("[ERROR] No se encontraron resultados JSON en", RESULTADOS_DIR)
        sys.exit(1)

    if datos_tts:
        graficos_tts(datos_tts)

    if datos_stt:
        graficos_stt(datos_stt)

    print("\nListo. Gráficas guardadas en", RESULTADOS_DIR)


if __name__ == "__main__":
    main()
