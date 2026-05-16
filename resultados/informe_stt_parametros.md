# Benchmark STT — Documentación de Parámetros y Metodología

> Generado automáticamente el 2026-05-16T00:15:46.913998

> Hardware: x86_64 | 8 cores | RAM 15687.3 MB


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

- **36 frases** de comandos de voz en español para robótica
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
Mejor global (score=0.800): whisper-tiny_int8_beam1_best1
Mejor velocidad (RTF=0.052):       whisper-tiny_int8_beam1_best1
Mejor precision (WER=0.287):       vosk-small-es_c1000
Pesos usados: alpha=0.4 (precision), beta=0.4 (velocidad), gamma=0.2 (RAM)
```

## 10. Ranking Completo

| Pos | Configuración | Score | WER | CI 95% | RTF | RAM (MB) |
|-----|--------------|-------|-----|--------|-----|----------|
| 1 | `whisper-tiny_int8_beam1_best1` | 0.7996 | 0.501 | [0.343, 0.440] | 0.052 | 881.8 |
| 2 | `whisper-tiny_int8_beam3_best1` | 0.7755 | 0.462 | [0.290, 0.385] | 0.061 | 917.2 |
| 3 | `vosk-small-es_c2000` | 0.7696 | 0.287 | [0.144, 0.221] | 0.069 | 1059.6 |
| 4 | `vosk-small-es_c4000` | 0.7684 | 0.289 | [0.147, 0.224] | 0.069 | 1060.6 |
| 5 | `vosk-small-es_c8000` | 0.7662 | 0.287 | [0.144, 0.221] | 0.070 | 1060.8 |
| 6 | `vosk-small-es_c1000` | 0.7642 | 0.287 | [0.144, 0.221] | 0.071 | 1058.2 |
| 7 | `whisper-tiny_int8_beam5_best3` | 0.7150 | 0.486 | [0.321, 0.416] | 0.077 | 926.8 |
| 8 | `whisper-tiny_float32_beam3_best1` | 0.6275 | 0.481 | [0.316, 0.411] | 0.088 | 1082.0 |
| 9 | `whisper-tiny_float32_beam1_best1` | 0.6259 | 0.511 | [0.351, 0.448] | 0.089 | 1045.4 |
| 10 | `whisper-tiny_float32_beam5_best3` | 0.5974 | 0.468 | [0.293, 0.387] | 0.098 | 1097.4 |
| 11 | `whisper-base_int8_beam3_best1` | 0.5799 | 0.380 | [0.197, 0.282] | 0.118 | 1079.6 |
| 12 | `whisper-base_int8_beam1_best1` | 0.5706 | 0.391 | [0.217, 0.304] | 0.120 | 1076.7 |
| 13 | `whisper-base_int8_beam5_best3` | 0.5135 | 0.397 | [0.217, 0.304] | 0.125 | 1185.4 |
| 14 | `whisper-base_float32_beam3_best1` | 0.3140 | 0.400 | [0.212, 0.298] | 0.166 | 1394.8 |
| 15 | `whisper-base_float32_beam1_best1` | 0.2892 | 0.388 | [0.207, 0.293] | 0.172 | 1426.8 |
| 16 | `whisper-base_float32_beam5_best3` | 0.2669 | 0.389 | [0.202, 0.287] | 0.187 | 1365.5 |
