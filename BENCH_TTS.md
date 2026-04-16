# Benchmark TTS Edge v2 — RPi4

**Script:** `bench_tts_edgetts_v2.py`  
**Hardware objetivo:** Raspberry Pi 4, 4 GB RAM, ARM Cortex-A72, sin GPU  
**Propósito:** Comparar los motores TTS más ligeros disponibles para edge computing, midiendo su viabilidad en el asistente TurtleBot4: velocidad de síntesis, calidad de la voz, consumo de recursos y comportamiento térmico en ejecución prolongada.

---

## Motores comparados

### Piper TTS

Motor TTS offline basado en ONNX Runtime, desarrollado por Rhasspy. Diseñado específicamente para dispositivos embebidos. Usa redes VITS ligeras exportadas como grafos ONNX.

| Parámetro | Detalle |
|---|---|
| Arquitectura | VITS + ONNX Runtime |
| Voces disponibles (en benchmark) | 3 voces españolas |
| Tamaño por voz | 28–63 MB |
| RAM típica RPi4 | 150–250 MB |
| Streaming | Sí — genera audio en chunks (TTFB real medible) |
| GPU requerida | No |

**Voces incluidas:**

| Archivo | Calidad | Característica |
|---|---|---|
| `es_ES-davefx-medium` | medium | Voz masculina adulta, muy natural |
| `es_ES-mls_10246-low` | low | Voz masculina, modelo más ligero (~28 MB) |
| `es_ES-sharvard-medium` | medium | Voz femenina |

- **Ventaja principal:** es el motor más rápido en RPi4 (RTF típico ~0.15–0.30), tiene TTFB medible porque genera audio en streaming real, y sus modelos son muy pequeños.
- **Limitación:** las voces suenan ligeramente sintéticas comparadas con XTTS-v2; la variabilidad prosódica depende de `noise_scale` y `noise_w`.

---

### Coqui TTS

Motor TTS de código abierto con modelos de mayor calidad. Dos variantes:

#### VITS ES (css10)

| Parámetro | Detalle |
|---|---|
| Modelo | `tts_models/es/css10/vits` |
| Dataset | CSS10 (lectura de libros, español) |
| Tamaño | ~80 MB |
| RAM típica RPi4 | ~400 MB |
| Streaming | No — genera el WAV completo y lo devuelve |

#### XTTS-v2

| Parámetro | Detalle |
|---|---|
| Modelo | `tts_models/multilingual/multi-dataset/xtts_v2` |
| Calidad | Alta — voz muy natural, clonación de voz opcional |
| Tamaño | ~1.8 GB |
| RAM RPi4 | ~2.5 GB (requiere ≥2.5 GB libres) |
| Streaming | No |
| Fallback automático | Si RAM < 2500 MB → usa VITS en su lugar |

- **Ventaja:** XTTS-v2 produce la voz más natural del benchmark; VITS es un buen equilibrio calidad/RAM.
- **Limitación:** no tiene streaming real, por lo que la latencia medida es el tiempo total de síntesis. XTTS-v2 puede ser inviable en RPi4 con otros procesos activos.

---

### KittenTTS

Motor TTS ligero con API local no-streaming. Permite ajustar voz, velocidad y temperatura (variabilidad expresiva).

| Parámetro | Detalle |
|---|---|
| Voces | `es_female`, `es_male` |
| Streaming | No — API no-streaming |
| Instalación | `pip install kittentts` |
| Modelos locales | No requiere ficheros en `models/` |

- **Ventaja:** muy fácil de usar, sin modelos locales que gestionar.
- **Limitación:** si no está disponible en pip o la API falla, el benchmark lo omite automáticamente con `[SKIP]`.

---

## Parámetros que se pueden cambiar

### Piper — param sets (5 configuraciones por voz)

Cada voz Piper se prueba con 5 conjuntos de parámetros:

| Tag | `length_scale` | `noise_scale` | `noise_w` | Efecto |
|---|---|---|---|---|
| `fast` | 0.7 | 0.667 | 0.8 | Habla más rápida (~30%) |
| `default` | 1.0 | 0.667 | 0.8 | Configuración base |
| `slow` | 1.3 | 0.667 | 0.8 | Habla más lenta (~30%) |
| `lowvar` | 1.0 | 0.3 | 0.3 | Poca variabilidad tonal — voz más monótona, predecible |
| `highvar` | 1.0 | 1.0 | 1.0 | Alta variabilidad — voz más expresiva pero menos estable |

**Significado de los parámetros:**
- `length_scale`: escala de duración de los fonemas. < 1.0 = más rápido, > 1.0 = más lento.
- `noise_scale`: variabilidad en la curva de tono (pitch). Afecta la naturalidad.
- `noise_w` (noise_w_scale): variabilidad en la duración de los fonemas. Afecta el ritmo.

Combinando las 3 voces × 5 param sets = **15 configuraciones Piper**.

---

### Coqui TTS — configuraciones

| Nombre | Modelo | `length_scale` | Descripción |
|---|---|---|---|
| `coqui_vits_speed08` | VITS ES | 1.25 | Habla más lenta (speed=0.8) |
| `coqui_vits_speed10` | VITS ES | 1.0 | Default |
| `coqui_vits_speed12` | VITS ES | 0.833 | Habla más rápida (speed=1.2) |
| `coqui_xtts_v2` | XTTS-v2 | 1.0 | Alta calidad — fallback a VITS si RAM < 2.5 GB |

> En Coqui `length_scale` es el inverso de `speed`: `length_scale = 1 / speed`.

---

### KittenTTS — configuraciones

Barrido completo: 2 voces × 3 speeds × 3 temperatures = **18 configuraciones**  
Modo `--quick`: 1 voz × 1 speed × 1 temperature = **3 configuraciones**

| Parámetro | Valores | Descripción |
|---|---|---|
| `voice` | `es_female`, `es_male` | Voz del sintetizador |
| `speed` | 0.9, 1.0, 1.1 | Velocidad de habla |
| `temperature` | 0.5, 0.8, 1.0 | Variabilidad expresiva (0 = determinista, 1 = máxima variación) |

---

## Corpus de síntesis

50 frases en 3 grupos de longitud. El objetivo es detectar si un motor penaliza desproporcionadamente las frases largas o tiene un coste fijo de arranque relevante:

| Grupo | Nº frases | Longitud aprox. | Ejemplo |
|---|---|---|---|
| Cortas | 15 | ≤ 5 palabras | `"Finalizado."`, `"Sistema listo."` |
| Medias | 20 | 6–12 palabras | `"Hola, soy Ana, tu robot asistente personal."` |
| Largas | 15 | > 12 palabras | `"He detectado un obstáculo inesperado en mi ruta..."` |

Las frases están diseñadas para representar la voz real del asistente TurtleBot4 (respuestas del sistema, mensajes de estado, avisos).

---

## Cómo se ejecuta el benchmark

### Flujo general

```
Para cada motor (Piper, Coqui, KittenTTS):
  Para cada configuración del motor:
    1. Esperar enfriamiento CPU < 55°C
    2. Registrar throttling RPi4 (inicio)
    3. Para cada frase (50 frases):
         Para cada repetición (N_REPS=5):
           - Sintetizar → medir tiempo_s, TTFB (solo Piper), RAM pico, CPU%
           - Rep 0 = warmup (excluida)
         Guardar WAV de la última rep (para evaluación auditiva)
    4. Registrar throttling RPi4 (fin)
    5. Calcular RTF ponderado, P50/P95/P99, RAM pico, temperatura máxima

Tras liberar todos los motores TTS:
  Cargar Whisper tiny ES una sola vez
  Calcular WER de todos los WAVs generados
```

### RTF ponderado

El RTF no se calcula como la media de RTFs por frase, sino como:

```
RTF = Σ(tiempos_síntesis) / Σ(duraciones_audio)
```

Esto evita que frases muy cortas (con overhead fijo dominante) distorsionen el RTF global. Se reporta también por grupo de longitud.

### TTFB (Time To First Byte)

Solo Piper tiene TTFB real: se mide el tiempo desde que se llama a `synthesize()` hasta que llega el primer chunk de audio. Es la métrica más relevante para el tiempo de respuesta percibido en el asistente.

Coqui y KittenTTS no hacen streaming, por lo que su "TTFB" equivale al tiempo total de síntesis.

### Repeticiones y warmup

```
N_REPS = 5
  Rep 0 → warmup (guardada pero excluida de estadísticas)
  Reps 1–4 → medidas válidas (4 observaciones por frase)
```

Los percentiles (P50, P95, P99) se calculan sobre **todas las observaciones individuales** (no sobre medias de frases), lo que captura correctamente la distribución de latencias.

---

## Métricas reportadas

| Métrica | Motor | Descripción |
|---|---|---|
| `ttfb_s` | Solo Piper | Tiempo hasta el primer chunk de audio |
| `tiempo_sintesis_s` | Todos | Tiempo total de síntesis del WAV completo |
| `rtf` | Todos | RTF ponderado (ver arriba) |
| `p50_latencia_s` | Todos | Mediana de latencias |
| `p95_latencia_s` | Todos | Percentil 95 — latencia en el peor 5% de casos |
| `p99_latencia_s` | Todos | Percentil 99 — cota casi máxima |
| `ram_pico_mb` | Todos | RSS máximo durante la síntesis |
| `cpu_pico_pct` | Todos | CPU% máximo |
| `cpu_medio_pct` | Todos | CPU% medio durante la síntesis |
| `temp_cpu_c` | Todos | Temperatura CPU durante la síntesis (sysfs) |
| `throttling_hex` | Todos | Bitmask de throttling RPi4 |
| `wer` | Todos | WER medido con Whisper tiny sobre los WAVs generados |
| `tiempo_carga_s` | Todos | Tiempo de inicialización del motor/modelo |
| `ram_carga_mb` | Todos | RAM al cargar el modelo |

---

## Salidas generadas

```
resultados/
├── bench_tts_edgetts_v2.json          ← datos completos
├── bench_tts_edgetts_v2.csv           ← tabla plana (importable en Excel/pandas)
├── informe_tts_edgetts_v2.md          ← informe legible con análisis por motor
├── graficas_tts_v2/
│   ├── rtf_comparativa.png
│   ├── latencia_p50_p95.png
│   ├── ram_comparativa.png
│   ├── wer_comparativa.png
│   └── temperatura_throttling.png
├── piper_es_ES-davefx-medium_default/ ← WAVs generados por esta config
├── piper_es_ES-davefx-medium_fast/
├── ...
├── coqui_vits_speed10/
└── kitten_es_female_s10_t08/
```

Los WAVs permiten evaluación auditiva subjetiva de la calidad de voz.

---

## Comandos de ejecución

```bash
# Benchmark completo
python3 bench_tts_edgetts_v2.py

# Modo rápido (pocas configs, útil para verificar instalación)
python3 bench_tts_edgetts_v2.py --quick

# Sin evaluación WER con Whisper (ahorra ~200 MB de RAM)
python3 bench_tts_edgetts_v2.py --no-quality

# Directorio de salida personalizado
python3 bench_tts_edgetts_v2.py --output-dir /mnt/usb/resultados

# Combinaciones
python3 bench_tts_edgetts_v2.py --quick --no-quality
```

---

## Cosas a tener en cuenta en RPi4

### Temperatura y cooldown térmico

El benchmark incluye un **sistema de espera térmica automático** entre configuraciones: espera hasta que la CPU baje de **55°C** antes de iniciar la siguiente config (timeout: 60 segundos).

Esto es fundamental para comparar configuraciones en igualdad de condiciones: el throttling de la RPi4 reduce la frecuencia de CPU cuando supera ~80°C, lo que distorsiona las medidas de latencia.

- Temperatura normal en idle RPi4: 40–50°C
- Con carga TTS sostenida: 60–75°C (dependiendo del disipador)
- Con disipador activo + ventilador: se mantiene por debajo de 60°C

**Recomendación:** usa disipador con ventilador para este benchmark. Sin ventilador, el cooldown entre configs puede alargar el tiempo total del benchmark en 10–20 minutos.

### Throttling

El script registra el bitmask `/sys/devices/platform/soc/soc:firmware/get_throttled` antes y después de cada configuración. Si aparece cualquier valor distinto de `0x0`, los resultados de esa config pueden estar sesgados hacia latencias más altas.

Bits relevantes:
- `0x1` = bajo voltaje actual (fuente de alimentación insuficiente)
- `0x2` = limitación de frecuencia activa
- `0x4` = throttling térmico activo

**Usar siempre una fuente de alimentación de al menos 3A** para RPi4 bajo carga.

### XTTS-v2 y RAM

XTTS-v2 necesita ~2.5 GB de RAM libre. En RPi4 4 GB con el sistema operativo y otros procesos, esto puede no estar disponible. El script detecta la RAM libre automáticamente y hace fallback a VITS si no hay suficiente. Puedes forzar el intento con `--force-ram` (no disponible en TTS, solo en LLM), pero no está implementado aquí — si falla, fallback es automático.

Para maximizar la RAM disponible antes de ejecutar:
```bash
sudo systemctl stop bluetooth avahi-daemon cups 2>/dev/null || true
```

### WER con Whisper tiny

La evaluación de calidad (WER) usa Whisper tiny en español, cargado **una sola vez al final** después de liberar todos los motores TTS. Esto evita solapar el consumo de RAM de TTS y STT.

Con `--no-quality` se omite esta fase: útil si la RPi4 tiene poca RAM libre o si solo te interesa medir velocidad.

### Comparación justa entre motores

Al ser motores tan distintos (Piper streaming vs. Coqui/KittenTTS no-streaming), no compares TTFB directamente entre ellos. Usa el tiempo total de síntesis (`tiempo_sintesis_s`) como denominador común.

### Número total de configuraciones

Sin `--quick`:
- Piper: 3 voces × 5 param sets = 15 configs
- Coqui: 4 configs (3 VITS + 1 XTTS-v2)
- KittenTTS: 18 configs
- **Total: ~37 configs × 50 frases × 5 reps = ~9250 síntesis**

El benchmark completo puede tardar **3–6 horas** en RPi4 dependiendo del cooldown térmico.  
Usa `--quick` para validaciones previas (~30 minutos).
