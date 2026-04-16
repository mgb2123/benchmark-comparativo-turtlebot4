# Benchmark STT — Documentación de Parámetros y Metodología

> Generado automáticamente el 2026-04-15T22:11:03.901128

> Hardware: AMD64 | 8 cores | RAM 16080.3 MB


## 1. Objetivo del Benchmark

Comparar motores de reconocimiento de voz (STT) para su uso como interfaz de lenguaje natural del robot TurtleBot4 con Raspberry Pi 4. El objetivo es identificar el motor que mejor equilibre **precisión de transcripción**, **latencia en tiempo real** y **huella de memoria**, dado que el hardware tiene recursos limitados (4 cores ARM Cortex-A72, 4 GB RAM).

## 2. Motores Evaluados

### faster-whisper

Implementación optimizada de OpenAI Whisper basada en CTranslate2. Whisper es un modelo Transformer encoder-decoder entrenado con 680.000 horas de audio multilingüe supervisado. El encoder procesa espectrogramas log-mel; el decoder genera tokens de texto autoregressivamente. CTranslate2 permite cuantización en int8 reduciendo uso de memoria y latencia con pérdida mínima de WER.

### Vosk (Kaldi)

Motor basado en Kaldi, que usa modelos acústicos DNN-HMM y modelos de lenguaje de n-gramas. Diseñado para reconocimiento en streaming (chunk por chunk), con bajo uso de RAM y latencia predecible. El modelo `vosk-model-small-es-0.42` está optimizado para español en hardware embebido.

## 3. Parámetros del Experimento

### 3.1 Parámetros de faster-whisper

| Parámetro | Valores testados | Significado | Efecto esperado |
|-----------|-----------------|-------------|-----------------|
| `modelo` | `tiny`, `base` | Tamaño de la arquitectura Transformer (tiny: 39M params, base: 74M params) | Más grande → mejor WER, más RAM, más latencia |
| `compute_type` | `int8`, `float32` | Precisión numérica de los pesos durante la inferencia | int8: ~2-4× más rápido, ~40% menos RAM; pérdida de WER mínima (<2 pp) |
| `beam_size` | 1, 2, 3, 4, 5 | Amplitud del haz de búsqueda del decodificador: nº de hipótesis mantenidas simultáneamente | Más alto → mejor WER (exploración más amplia), más latencia (O(beam×L) tokens) |
| `best_of` | 1, 3 | Nº de hipótesis independientes generadas (requiere temperatura > 0) | Solo efectivo con temperatura > 0; aumenta WER a costa de latencia |
| `temperature` | 0.0, 0.2 | Aleatoriedad del muestreo del decodificador (0 = greedy determinista) | 0.0 = reproducible y estable; > 0 = estocástico, puede mejorar WER con `best_of` > 1 |
| `vad_filter` | True, False | Filtro de actividad de voz (WebRTC VAD) aplicado antes de la transcripción | Elimina segmentos de silencio; puede reducir WER en frases cortas y latencia en silencios largos |
| `min_silence_duration_ms` | 200–1000 ms | Duración mínima de silencio para que el VAD corte un segmento | Más corto: más reactivo, riesgo de cortar palabras; más largo: más conservador |

### 3.2 Parámetros de Vosk

| Parámetro | Valores testados | Significado | Efecto esperado |
|-----------|-----------------|-------------|-----------------|
| `chunk_frames` | 500, 1000, 2000, 4000, 8000 | Nº de muestras de audio enviadas al reconocedor por iteración (a 16 kHz: 31–500 ms de audio por chunk) | Chunks pequeños: mayor latencia por overhead de llamadas; chunks grandes: menor overhead pero mayor buffer |

### 3.3 Parámetros del Experimento

| Parámetro | Default | Flag CLI | Significado |
|-----------|---------|----------|-------------|
| `N_REPS` | 3 | `--n-reps N` | Repeticiones por clip; la 1ª siempre se descarta como warmup para eliminar sesgos de inicialización JIT |
| `alpha` | 0.4 | `--weights α β γ` | Peso de la precisión `(1-WER)` en el score compuesto |
| `beta` | 0.4 | `--weights α β γ` | Peso de la velocidad `(1-RTF_norm)` en el score compuesto |
| `gamma` | 0.2 | `--weights α β γ` | Peso del consumo de RAM `(1-RAM_norm)` en el score compuesto |

## 4. Métricas Recogidas

### Word Error Rate (WER)

Mide la distancia de edición a nivel de palabras entre la hipótesis (transcripción STT) y la referencia (ground truth), normalizada por la longitud de la referencia:

```
WER = (S + D + I) / N
```

donde S = sustituciones, D = borrados, I = inserciones, N = palabras en la referencia. Un WER de 0.0 es perfecto; valores > 1.0 indican más inserciones que palabras de referencia. **Normalización aplicada**: minúsculas, eliminación de diacríticos (NFD), solo alfanuméricos. Esto hace que `"qué"` = `"que"` y `"llévame"` = `"llevame"`, haciéndola más robusta ante variaciones ortográficas entre motores.

### Real-Time Factor (RTF)

Ratio entre el tiempo de inferencia y la duración del audio:

```
RTF = t_inferencia / t_audio
```

- **RTF < 1.0**: el motor transcribe más rápido que el audio en tiempo real → viable para asistente de voz
- **RTF > 1.0**: el motor no puede seguir el ritmo → inviable para uso en tiempo real
- En RPi4, se busca RTF < 0.5 para dejar margen a otros procesos del sistema

### RAM pico (`ram_pico_mb`)

Diferencia de RSS (Resident Set Size) del proceso antes y durante la transcripción, medida cada 100 ms con `psutil`. No incluye memoria compartida de bibliotecas. Relevante porque el RPi4 tiene 4 GB compartidos con el SO y otros procesos del robot.

### CPU (`cpu_pct`)

Media de `psutil.cpu_percent()` durante la inferencia. faster-whisper usa todos los cores disponibles vía OpenMP; Vosk es principalmente single-core. En RPi4, un CPU% alto puede interferir con otros nodos ROS2.

### Warmup (`warmup_s`)

Tiempo de la primera transcripción, excluida de los promedios. La primera inferencia suele ser más lenta por inicialización de buffers internos, caché JIT (ONNX/PyTorch) y carga de pesos a la caché de CPU. Reportarlo permite identificar configuraciones con overhead de inicialización alto.

## 5. Corpus de Prueba

- **19 frases** de comandos de voz en español para robótica
- Duración media: ~2.7 s por frase
- Formato: WAV, mono, 16 kHz, PCM 16-bit
- **Fuente**: audio sintético generado con Piper TTS (voz `es_ES-davefice-medium`). Para resultados representativos del uso real se recomienda audio humano grabado en condiciones similares al entorno de despliegue del robot.

| # | Frase de referencia |
|---|---------------------|
| 00 | `para el robot` |
| 01 | `gira a la derecha` |
| 02 | `vuelve a la base` |
| 03 | `navega hasta la cocina y espera mis instrucciones allí` |
| 04 | `busca el objeto rojo que está encima de la mesa` |
| 05 | `toma una foto del pasillo y guárdala en memoria` |
| 06 | `avanza hasta el salón, gira noventa grados a la derecha y para cuando llegues a la pared del fondo` |
| 07 | `localiza a la persona que está en la habitación y cuando la encuentres emite una señal sonora para avisarme` |
| 08 | `detecta si hay obstáculos en el pasillo y si los hay traza una ruta alternativa hacia el destino principal` |

## 6. Metodología de Scoring Compuesto

Para elegir objetivamente el mejor modelo se utiliza un **score compuesto** que pondera las tres dimensiones clave:

```
Score = α·max(0, 1-WER) + β·(1-RTF_norm) + γ·(1-RAM_norm)
```

donde RTF_norm y RAM_norm están normalizados min-max sobre todos los modelos testados (0 = peor, 1 = mejor en esa métrica). Los pesos usados en esta ejecución: **α=0.4** (precisión), **β=0.4** (velocidad), **γ=0.2** (RAM).

Los pesos por defecto (0.4, 0.4, 0.2) reflejan el criterio de que para un asistente de voz robótico la velocidad de respuesta y la precisión son igualmente críticas, mientras que el consumo de RAM es una restricción más blanda (siempre que quepa en 4 GB). Se pueden ajustar con `--weights α β γ`.

## 7. Intervalo de Confianza Wilson al 95%

El WER medio sobre 10 clips es una estimación puntual con incertidumbre. Se aplica el **intervalo de Wilson** sobre la proporción total de errores de palabras (total_errores / total_palabras_referencia). Este método es más robusto que el intervalo normal de Wald para n pequeño y proporciones cercanas a 0 o 1.

Un intervalo estrecho indica resultados más estables y reproducibles. Configuraciones con CI amplio pueden variar significativamente entre ejecuciones.

## 8. Flags CLI Disponibles

```bash
# Ejecución normal (benchmark + ranking + gráficas):
python bench_stt_exhaustivo.py

# Solo regenerar gráficas e informe desde JSON existente (sin re-ejecutar):
python bench_stt_exhaustivo.py --plot-only

# Más repeticiones para mayor rigor estadístico:
python bench_stt_exhaustivo.py --n-reps 5

# Barrido paramétrico completo (beam_size, VAD threshold, chunk, temperatura):
python bench_stt_exhaustivo.py --sweep

# Cambiar pesos del score (priorizar precisión sobre velocidad):
python bench_stt_exhaustivo.py --weights 0.6 0.2 0.2
```

## 9. Resultados — Veredicto

```
Mejor global (score=0.666): whisper-tiny_int8_beam1_best1
Mejor velocidad (RTF=0.195):       whisper-tiny_int8_beam1_best1
Mejor precision (WER=0.632):       whisper-base_int8_beam5_best3
Pesos usados: alpha=0.4 (precision), beta=0.4 (velocidad), gamma=0.2 (RAM)
```

## 10. Ranking Completo

| Pos | Configuración | Score | WER | CI 95% | RTF | RAM (MB) |
|-----|--------------|-------|-----|--------|-----|----------|
| 1 | `whisper-tiny_int8_beam1_best1` | 0.6664 | 0.834 | [0.684, 0.806] | 0.195 | 623.9 |
| 2 | `whisper-tiny_int8_beam3_best1` | 0.6365 | 0.881 | [0.684, 0.806] | 0.205 | 629.1 |
| 3 | `whisper-tiny_int8_beam5_best3` | 0.6302 | 0.825 | [0.657, 0.782] | 0.230 | 655.0 |
| 4 | `whisper-base_int8_beam3_best1` | 0.5809 | 0.638 | [0.518, 0.656] | 0.338 | 768.4 |
| 5 | `whisper-base_int8_beam5_best3` | 0.5725 | 0.632 | [0.502, 0.641] | 0.347 | 786.2 |
| 6 | `whisper-tiny_float32_beam1_best1` | 0.5646 | 0.800 | [0.679, 0.801] | 0.294 | 754.5 |
| 7 | `whisper-base_int8_beam1_best1` | 0.5621 | 0.651 | [0.534, 0.671] | 0.351 | 762.3 |
| 8 | `whisper-tiny_float32_beam3_best1` | 0.5205 | 0.830 | [0.646, 0.773] | 0.323 | 766.5 |
| 9 | `vosk-small-es_c8000` | 0.5203 | 0.964 | [0.913, 0.975] | 0.278 | 692.3 |
| 10 | `vosk-small-es_c1000` | 0.5183 | 0.964 | [0.913, 0.975] | 0.278 | 723.2 |
| 11 | `vosk-small-es_c2000` | 0.5066 | 0.964 | [0.913, 0.975] | 0.291 | 686.9 |
| 12 | `vosk-small-es_c4000` | 0.4803 | 0.959 | [0.907, 0.972] | 0.317 | 689.2 |
| 13 | `whisper-base_float32_beam1_best1` | 0.3538 | 0.705 | [0.571, 0.705] | 0.510 | 992.1 |
| 14 | `vosk-large-es_c8000` | 0.0599 | 0.918 | [0.851, 0.936] | 0.540 | 3736.5 |
| 15 | `vosk-large-es_c1000` | 0.0434 | 0.932 | [0.875, 0.952] | 0.550 | 3738.0 |
| 16 | `vosk-large-es_c2000` | 0.0328 | 0.924 | [0.857, 0.940] | 0.563 | 3733.8 |
| 17 | `vosk-large-es_c4000` | 0.0306 | 0.924 | [0.863, 0.944] | 0.565 | 3734.7 |
