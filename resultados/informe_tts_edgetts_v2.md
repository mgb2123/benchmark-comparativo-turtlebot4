# Informe Benchmark TTS Edge v2

**Fecha:** 2026-04-16 14:25:59  
**Plataforma:** AMD64 — Windows 10  
**CPU cores:** 8  
**RAM total:** 16080.3 MB  

## Metodología

- **N_REPS:** 5 (primera repetición descartada como warmup; 4 medidas válidas)
- **Corpus:** 50 frases — 15 cortas (≤5 pal.), 20 medias (6-12 pal.), 15 largas (>12 pal.).
- **RTF:** ponderado por duración — `Σsynthesis_times / Σaudio_durations` (no media de medias).
- **std:** desviación estándar de TODAS las observaciones individuales (no media de stds por frase).
- **TTFB:** Time to First Byte — real solo en Piper (streaming). Coqui/Kitten no tienen streaming: se muestra `-` y se usa `tiempo_sintesis_s` como latencia.
- **CPU%:** monitorizado continuamente cada 50 ms durante la síntesis.
- **Throttling (RPi4):** bitmask de `/sys/.../get_throttled` registrado antes/después de cada config.
- **WER:** calculado con Whisper tiny ES (normalización unicode + edit-distance).
- **UTMOS:** pendiente cálculo offline con los WAVs generados si `utmos` no disponible.

## Tabla resumen

| Config | Motor | Carga(s) | RAM(MB) | RTF±std | TTFB(s)* | P50(s) | P95(s) | WER | CPU%pico | Throttle | Temp_pico(°C) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piper_es_ES-davefx-medium_fast | piper | 2.13 | 343.8 | 0.059±0.0044 | 0.1505 | 0.1363 | 0.2842 | N/A | 543.5 | N/A | N/A |
| piper_es_ES-davefx-medium_defa | piper | 2.13 | 397.3 | 0.063±0.0025 | 0.2045 | 0.1863 | 0.3786 | N/A | 565.2 | N/A | N/A |
| piper_es_ES-davefx-medium_slow | piper | 2.13 | 413.9 | 0.068±0.0034 | 0.2612 | 0.2369 | 0.4967 | N/A | 565.2 | N/A | N/A |
| piper_es_ES-davefx-medium_lowv | piper | 2.13 | 414.9 | 0.069±0.0089 | 0.2224 | 0.2048 | 0.4005 | N/A | 565.2 | N/A | N/A |
| piper_es_ES-davefx-medium_high | piper | 2.13 | 415.7 | 0.069±0.0057 | 0.23 | 0.2125 | 0.431 | N/A | 565.2 | N/A | N/A |
| piper_es_ES-mls_10246-low_fast | piper | 2.08 | 523.4 | 0.05±0.0118 | 0.253 | 0.257 | 0.3925 | N/A | 565.2 | N/A | N/A |
| piper_es_ES-mls_10246-low_defa | piper | 2.08 | 683.9 | 0.045±0.0111 | 0.3245 | 0.3188 | 0.6099 | N/A | 577.4 | N/A | N/A |
| piper_es_ES-mls_10246-low_slow | piper | 2.08 | 826.7 | 0.046±0.0131 | 0.3881 | 0.3694 | 0.7065 | N/A | 577.4 | N/A | N/A |
| piper_es_ES-mls_10246-low_lowv | piper | 2.08 | 825.8 | 0.05±0.0068 | 0.2135 | 0.206 | 0.3203 | N/A | 565.2 | N/A | N/A |
| piper_es_ES-mls_10246-low_high | piper | 2.08 | 877.2 | 0.047±0.0187 | 0.4261 | 0.3817 | 0.7844 | N/A | 611.4 | N/A | N/A |
| piper_es_ES-sharvard-medium_fa | piper | 1.99 | 363.0 | 0.071±0.0049 | 0.1884 | 0.1708 | 0.3495 | N/A | 565.2 | N/A | N/A |
| piper_es_ES-sharvard-medium_de | piper | 1.99 | 426.2 | 0.068±0.0034 | 0.2324 | 0.2069 | 0.4322 | N/A | 565.2 | N/A | N/A |
| piper_es_ES-sharvard-medium_sl | piper | 1.99 | 465.2 | 0.07±0.0077 | 0.2907 | 0.2665 | 0.5587 | N/A | 577.4 | N/A | N/A |
| piper_es_ES-sharvard-medium_lo | piper | 1.99 | 466.2 | 0.076±0.0053 | 0.2559 | 0.2329 | 0.4712 | N/A | 577.4 | N/A | N/A |
| piper_es_ES-sharvard-medium_hi | piper | 1.99 | 467.7 | 0.077±0.0059 | 0.2665 | 0.2431 | 0.4959 | N/A | 577.4 | N/A | N/A |

> \* TTFB real solo para motores con streaming (Piper). Para Coqui/KittenTTS el tiempo de latencia es `tiempo_sintesis_s` (ver P50/P95).

## Gráficas

### rtf_ttfb_barras.png
![rtf_ttfb_barras.png](graficas_tts_v2\rtf_ttfb_barras.png)

### boxplot_latencias.png
![boxplot_latencias.png](graficas_tts_v2\boxplot_latencias.png)

### radar_metricas.png
![radar_metricas.png](graficas_tts_v2\radar_metricas.png)

## RTF por grupo de longitud de frase

Detecta si un motor penaliza desproporcionadamente frases largas o tiene un coste fijo de arranque alto.

### piper_es_ES-davefx-medium_fast (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.059 | 0.064 | 0.0825 | 0.062 |
| media | 20 | 0.059 | 0.137 | 0.1663 | 0.134 |
| larga | 15 | 0.059 | 0.266 | 0.291 | 0.266 |

### piper_es_ES-davefx-medium_default (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.061 | 0.087 | 0.1243 | 0.084 |
| media | 20 | 0.063 | 0.1855 | 0.223 | 0.183 |
| larga | 15 | 0.063 | 0.358 | 0.3863 | 0.361 |

### piper_es_ES-davefx-medium_slow (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.065 | 0.108 | 0.1507 | 0.109 |
| media | 20 | 0.067 | 0.2375 | 0.2955 | 0.232 |
| larga | 15 | 0.068 | 0.47 | 0.5152 | 0.462 |

### piper_es_ES-davefx-medium_lowvar (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.079 | 0.097 | 0.1679 | 0.109 |
| media | 20 | 0.071 | 0.204 | 0.2381 | 0.204 |
| larga | 15 | 0.066 | 0.362 | 0.4235 | 0.37 |

### piper_es_ES-davefx-medium_highvar (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.072 | 0.104 | 0.1369 | 0.101 |
| media | 20 | 0.07 | 0.2105 | 0.2592 | 0.208 |
| larga | 15 | 0.068 | 0.394 | 0.439 | 0.396 |

### piper_es_ES-mls_10246-low_fast (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.049 | 0.224 | 0.4055 | 0.233 |
| media | 20 | 0.048 | 0.235 | 0.3299 | 0.249 |
| larga | 15 | 0.053 | 0.291 | 0.3148 | 0.288 |

### piper_es_ES-mls_10246-low_default (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.042 | 0.272 | 0.6493 | 0.331 |
| media | 20 | 0.046 | 0.3065 | 0.4209 | 0.316 |
| larga | 15 | 0.049 | 0.343 | 0.3768 | 0.341 |

### piper_es_ES-mls_10246-low_slow (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.04 | 0.347 | 0.7098 | 0.394 |
| media | 20 | 0.049 | 0.3935 | 0.5669 | 0.401 |
| larga | 15 | 0.048 | 0.389 | 0.4104 | 0.384 |

### piper_es_ES-mls_10246-low_lowvar (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.046 | 0.136 | 0.2891 | 0.16 |
| media | 20 | 0.05 | 0.183 | 0.2796 | 0.199 |
| larga | 15 | 0.054 | 0.297 | 0.3073 | 0.295 |

### piper_es_ES-mls_10246-low_highvar (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.043 | 0.408 | 0.766 | 0.458 |
| media | 20 | 0.048 | 0.405 | 0.6134 | 0.453 |
| larga | 15 | 0.051 | 0.377 | 0.4199 | 0.379 |

### piper_es_ES-sharvard-medium_fast (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.075 | 0.087 | 0.1192 | 0.079 |
| media | 20 | 0.071 | 0.167 | 0.2031 | 0.171 |
| larga | 15 | 0.07 | 0.328 | 0.3488 | 0.328 |

### piper_es_ES-sharvard-medium_default (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.07 | 0.097 | 0.1436 | 0.096 |
| media | 20 | 0.068 | 0.204 | 0.2482 | 0.209 |
| larga | 15 | 0.067 | 0.408 | 0.4396 | 0.407 |

### piper_es_ES-sharvard-medium_slow (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.07 | 0.124 | 0.1683 | 0.116 |
| media | 20 | 0.072 | 0.2615 | 0.3436 | 0.267 |
| larga | 15 | 0.069 | 0.515 | 0.572 | 0.507 |

### piper_es_ES-sharvard-medium_lowvar (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.077 | 0.104 | 0.1518 | 0.104 |
| media | 20 | 0.077 | 0.231 | 0.2937 | 0.235 |
| larga | 15 | 0.074 | 0.447 | 0.485 | 0.443 |

### piper_es_ES-sharvard-medium_highvar (piper)

| Grupo | N frases | RTF | P50(s) | P95(s) | T.medio(s) |
| --- | --- | --- | --- | --- | --- |
| corta | 15 | 0.082 | 0.119 | 0.1784 | 0.113 |
| media | 20 | 0.079 | 0.2375 | 0.3054 | 0.248 |
| larga | 15 | 0.075 | 0.459 | 0.496 | 0.457 |

## Notas de implementación

- **RTF ponderado:** `Σtiempos / Σduraciones` — no afectado por el peso de frases cortas.
- **std global:** calculado sobre todas las observaciones individuales, no promediando stds.
- **WER propio vs jiwer:** implementación simple sin dependencia extra.
- **UTMOS diferido:** torch+modelo pesado para RPi4; se deja gancho para offline.
- **TTFB real solo en Piper:** Coqui/Kitten no exponen streaming estable.
- **Fallback XTTS→VITS:** automático si RAM libre < 2.5 GB.
- **Throttling:** bitmask 0x0 = sistema sano; >0 indica bajo voltaje o limitación de frecuencia.
