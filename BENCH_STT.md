# Benchmark STT exhaustivo — RPi4

**Script:** `bench_stt_exhaustivo.py`  
**Hardware objetivo:** Raspberry Pi 4, 4 GB RAM, ARM Cortex-A72, sin GPU  
**Propósito:** Encontrar la configuración de reconocimiento de voz más eficiente para el asistente TurtleBot4, equilibrando precisión (WER), velocidad (RTF) y consumo de RAM.

---

## Motores comparados

### faster-whisper (OpenAI Whisper cuantizado)

Wrapper de CTranslate2 sobre los modelos Whisper de OpenAI. Permite inferencia en CPU con cuantización int8, lo que reduce RAM y acelera la transcripción respecto al Whisper original de Python.

| Parámetro | tiny | base |
|---|---|---|
| Parámetros | ~39 M | ~74 M |
| Tamaño en disco | ~78 MB | ~148 MB |
| RAM típica RPi4 (int8) | ~200 MB | ~350 MB |
| Idioma | multilingüe | multilingüe |
| Arquitectura | Transformer encoder | Transformer encoder |

- **Ventaja:** muy configurable (beam search, VAD, cuantización), buena precisión con `base`.
- **Limitación en RPi4:** la primera inferencia (warmup) es sensiblemente más lenta; con `float32` el consumo de RAM casi dobla al de `int8`.

---

### Vosk

Motor STT offline basado en Kaldi, diseñado específicamente para dispositivos embebidos. Usa un pipeline acústico clásico (MFCC + GMM/TDNN) con RNNLM opcional para rescoring.

| Parámetro | vosk-model-small-es-0.42 | vosk-model-es-0.42 |
|---|---|---|
| Tamaño en disco | ~39 MB | ~1.4 GB |
| RAM típica RPi4 | ~120 MB | ~800 MB |
| Latencia arranque | baja | alta |
| Calidad | media | alta |
| Streaming | sí (chunk-based) | sí (chunk-based) |

- **Ventaja:** latencia de arranque mínima, bajo consumo de RAM con el modelo small, API streaming real (procesa en chunks).
- **Limitación:** el modelo grande requiere ~800 MB de RAM libre, lo que puede ser problemático en RPi4 4 GB con otros procesos activos.

---

## Parámetros que se pueden cambiar

### Configuraciones fijas de Whisper probadas (9 en total)

```
modelo    | compute_type | beam_size | best_of
----------|--------------|-----------|--------
tiny      | int8         | 1         | 1
tiny      | int8         | 3         | 1
tiny      | int8         | 5         | 3
tiny      | float32      | 1         | 1
tiny      | float32      | 3         | 1
base      | int8         | 1         | 1
base      | int8         | 3         | 1
base      | int8         | 5         | 3
base      | float32      | 1         | 1
```

> El modelo `small` (~460 MB) está comentado por defecto; puede activarse si hay RAM suficiente.

### Parámetro: `compute_type`

| Valor | Significado | Impacto en RPi4 |
|---|---|---|
| `int8` | Pesos cuantizados a 8 bits enteros | Menor RAM, mayor velocidad |
| `float32` | Precisión completa | Mayor RAM (~2×), menor velocidad |

### Parámetro: `beam_size`

Número de hipótesis candidatas que el decodificador mantiene en paralelo. Más alto = más preciso pero más lento y con mayor uso de RAM.

- `beam_size=1` → decodificación greedy (más rápida)
- `beam_size=3` → equilibrio calidad/velocidad
- `beam_size=5` → máxima precisión (con `best_of=3`)

### Parámetro: `best_of`

Número de muestras paralelas; el decoder escoge la de mayor probabilidad. Solo relevante con `beam_size > 1`. Incrementa uso de CPU.

### Parámetro: VAD (`vad_filter`)

Silencious Voice Activity Detection integrado en faster-whisper. Filtra segmentos de silencio antes de la transcripción.

- Se prueba automáticamente con `beam_size=1` en paralelo a la configuración sin VAD.
- Puede mejorar el RTF en frases cortas al evitar que el modelo "transcriba silencio".

### Barrido paramétrico adicional

El script incluye tres barridos adicionales configurables:

```python
SWEEP_BEAM_SIZES   = [1, 2, 3, 4, 5]       # barrido de beam
SWEEP_VAD_MS       = [200, 350, 500, 750, 1000]  # umbral VAD en ms
SWEEP_CHUNK_FRAMES = [500, 1000, 2000, 4000, 8000]  # chunk size Vosk
SWEEP_TEMPERATURES = [0.0, 0.2]            # temperatura Whisper
```

### Pesos del score compuesto (ranking)

El ranking final se calcula con una función ponderada:

```
score = α × (1 - WER)  +  β × (1 - RTF_norm)  +  γ × (1 - RAM_norm)
```

Valores por defecto `α=0.4, β=0.4, γ=0.2` — precisión y velocidad igual de importantes, RAM con peso menor. Se pueden ajustar con el flag `--weights`.

---

## Corpus de audio

9 frases organizadas en 3 niveles de complejidad, representativas de comandos reales de TurtleBot:

| Grupo | Nº frases | Longitud | Ejemplo |
|---|---|---|---|
| Cortas | 3 | 3–5 palabras | `"para el robot"` |
| Medianas | 3 | 8–12 palabras | `"navega hasta la cocina y espera mis instrucciones allí"` |
| Largas | 3 | 20–30 palabras | `"avanza hasta el salón, gira noventa grados..."` |

### Audio real vs. sintético

**Recomendado: voz humana real.**  
Graba con el micrófono real del TurtleBot4 en condiciones similares al entorno de uso (ruido ambiente, distancia micrófono-hablante).

```bash
# Grabar frase por frase (formato: WAV, mono, 16 kHz, PCM 16-bit)
arecord -f S16_LE -r 16000 -c 1 audio_tests/frase_00.wav
```

**Alternativa sintética con Piper (solo si no es posible grabar):**

```bash
python3 bench_stt_exhaustivo.py --generar-audio --piper-fallback
```

> Los WER medidos con audio sintético pueden no representar el uso real; documentarlo siempre.

---

## Cómo se ejecuta el benchmark

### Flujo completo

```
Para cada configuración Whisper (9 configs):
  1. Cargar modelo → medir tiempo_carga_s y RAM inicial
  2. Para cada frase (9 WAVs):
       Para cada repetición (N_REPS=3):
         - Transcribir → medir tiempo_s, RTF, RAM pico, CPU%
         - Rep 0 = warmup (excluida del promedio)
       Calcular WER vs ground truth
       Reportar media ± std de las 2 reps válidas
  3. Repetir con vad_filter=True si beam_size=1

Para cada modelo Vosk encontrado en models/:
  1. Cargar modelo → medir tiempo_carga_s y RAM
  2. Para cada chunk_size en SWEEP_CHUNK_FRAMES:
       Para cada frase:
         N_REPS repeticiones (warmup excluido)
         Medir tiempo_s, RTF, WER, RAM, CPU
  3. Calcular promedios por config
```

### Cálculo del WER

Se eliminan diacríticos (NFD) antes de comparar: `"qué" == "que"`. Esto hace la comparación más robusta ante motores que omiten tildes, pero puede inflar/deflactar el WER real.

La incertidumbre estadística se reporta con el **intervalo de confianza de Wilson al 95%**, más robusto que el intervalo normal de Wald para corpus pequeños.

### Repeticiones y warmup

```
N_REPS = 3
  Rep 0 → warmup (guardado pero excluido de promedios)
  Rep 1, 2 → medidas válidas (media ± std)
```

La primera inferencia suele ser más lenta por caching del sistema operativo y precalentamiento del motor. Descartarla da estimaciones más representativas del uso continuo.

---

## Métricas reportadas

| Métrica | Descripción |
|---|---|
| `wer` | Word Error Rate (0 = perfecto, 1 = todo mal) |
| `wer_ci_95` | Intervalo de confianza Wilson 95% para WER |
| `tiempo_s` | Latencia de transcripción (segundos) |
| `std_tiempo_s` | Desviación estándar de la latencia |
| `rtf` | Real-Time Factor = tiempo_transcripción / duración_audio (< 1.0 = más rápido que real) |
| `std_rtf` | Desviación estándar del RTF |
| `ram_pico_mb` | RSS máximo durante la transcripción |
| `cpu_pct` | Porcentaje de CPU durante la transcripción |
| `tiempo_carga_s` | Tiempo para inicializar el modelo |
| `warmup_s` | Duración de la primera inferencia (excluida) |
| `score` | Score compuesto = α(1-WER) + β(1-RTF_norm) + γ(1-RAM_norm) |

---

## Salidas generadas

```
resultados/
├── bench_stt_exhaustivo.json      ← datos completos de todas las configs
├── informe_stt_parametros.md      ← informe legible con ranking y veredicto
└── graficas_stt/
    ├── heatmap_wer.png            ← WER por frase × configuración
    ├── rtf_comparativa.png        ← RTF por motor
    ├── ram_comparativa.png        ← RAM pico por configuración
    └── ranking_score.png          ← ranking por score compuesto
```

---

## Comandos de ejecución

```bash
# Benchmark completo
python3 bench_stt_exhaustivo.py

# Modo rápido (2 clips por config, útil para verificar que todo funciona)
python3 bench_stt_exhaustivo.py --quick

# Generar audio sintético con Piper (solo si no tienes audio real)
python3 bench_stt_exhaustivo.py --generar-audio --piper-fallback

# Especificar carpeta de modelos
python3 bench_stt_exhaustivo.py --models-dir /ruta/a/models

# Pesos personalizados del ranking (α=precisión, β=velocidad, γ=RAM)
python3 bench_stt_exhaustivo.py --weights 0.5 0.3 0.2
```

---

## Cosas a tener en cuenta en RPi4

### Temperatura y throttling

La RPi4 puede hacer **throttling térmico** si supera ~80°C: reduce la frecuencia de CPU automáticamente, lo que distorsiona las medidas de latencia. El script monitorea esto mediante el fichero sysfs `/sys/devices/platform/soc/soc:firmware/get_throttled`.

Bitmask de throttling:
- Bit 0: bajo voltaje **activo ahora**
- Bit 1: limitación de frecuencia **activa ahora**
- Bit 2: throttling **activo ahora**
- Bits 16-18: han ocurrido alguna vez desde el último reset

`0x0` = sistema sano. Cualquier otro valor indica que los resultados pueden estar sesgados.

**Recomendación:** usar disipador + ventilador en la RPi4. Ejecutar el benchmark con la RPi en reposo y temperatura < 50°C antes de empezar.

### Vosk con rutas no-ASCII

Vosk tiene un bug conocido: falla si la ruta al modelo contiene caracteres no-ASCII (tildes, ñ, espacios con codificación especial). El script detecta esto automáticamente y copia el modelo a una ruta ASCII temporal. Si tienes el proyecto en una ruta con tildes, no hay problema — el script lo gestiona.

### Modelo Vosk grande (~800 MB)

En RPi4 4 GB con procesos del sistema activos, el modelo grande puede necesitar >800 MB de RAM libre. Si el benchmark lo omite con `[SKIP]`, cierra otros procesos y vuelve a intentarlo.

### Modelo Whisper small

Está comentado por defecto (`~460 MB`). En RPi4 4 GB funciona, pero la descarga es lenta y las medidas empeoran con los otros motores activos. Descomentarlo solo si necesitas esa precisión.

### Swap

Si la RPi4 tiene poca RAM libre, aumentar el swap:
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile   # CONF_SWAPSIZE=1024
sudo dphys-swapfile setup && sudo dphys-swapfile swapon
```
El swap en tarjeta SD degrada mucho el rendimiento; usar como último recurso.

### Audio real vs. sintético

Los WER con audio sintético (Piper) son sistemáticamente más bajos que con voz humana, porque Piper genera audio "limpio" y bien articulado. Para un benchmark válido del sistema de voz del robot, **usa siempre audio grabado con el micrófono real del TurtleBot4**.
