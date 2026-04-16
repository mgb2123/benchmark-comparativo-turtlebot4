# Por qué Whisper tiene tantos parámetros y Vosk solo uno

La diferencia fundamental es **arquitectura**. Son dos filosofías de reconocimiento de voz completamente distintas.

---

## Vosk / Kaldi: pipeline clásico de etapas fijas

Vosk usa una cadena de procesado **determinista y modular**:

```
Audio → Extracción de características (MFCC) → Red neuronal acústica (DNN) → Decodificador HMM + n-grama → Texto
```

Cada etapa tiene un cometido único y no interactúa con las demás. El único parámetro que expone al usuario es **cuánto audio enviar cada vez** (`chunk_frames`), porque el modelo fue diseñado para *streaming* en tiempo real. Todo lo demás está fijado en el modelo entrenado: no puedes cambiar cómo busca, qué precisión numérica usa, ni si filtra silencio.

---

## faster-whisper: Transformer seq2seq con decodificación configurable

Whisper es un **modelo de traducción de secuencias** (como GPT pero para audio→texto). Funciona como un generador de texto: recibe el audio codificado y *genera* la transcripción token a token. Eso abre un abanico de decisiones en tiempo de inferencia que Vosk no tiene.

---

## Los parámetros en detalle

### `compute_type` — precisión numérica de los pesos del modelo

Los pesos de una red neuronal son números en coma flotante. La pregunta es: **¿con cuántos bits representamos cada número?**

| Tipo | Bits | Rango / precisión | Velocidad en CPU | RAM |
|------|------|-------------------|------------------|-----|
| `float32` | 32 | Alta | Lenta (operaciones pesadas) | ~4× más |
| `float16` | 16 | Media | Rápida en GPU (no en CPU) | ~2× más |
| `int8` | 8 | Baja (cuantizada) | Muy rápida en CPU | Mínima |

**¿Qué es cuantización int8?**
En entrenamiento, los pesos tienen valores continuos como `0.73291…`. En int8, cada peso se mapea a un entero de -128 a 127 más un factor de escala por capa. La operación matricial (el 90% del cómputo en un Transformer) se convierte en multiplicación de enteros, que la CPU ejecuta con instrucciones SIMD (AVX2) hasta **4-8× más rápido** que float32.

La pérdida de precisión en WER suele ser mínima (<1 punto porcentual absoluto) porque el modelo tiene redundancia paramétrica suficiente para absorber el ruido de cuantización.

**Para RPi4**: `int8` es casi obligatorio. El procesador ARM Cortex-A72 no tiene unidades FPU vectoriales potentes; la cuantización multiplica la velocidad por ~3-5×.

---

### `beam_size` — amplitud del haz de búsqueda en el decodificador

El decodificador genera la transcripción **token a token** (un token ≈ una sílaba/palabra). En cada paso tiene que elegir qué token viene a continuación de entre ~50.000 posibles.

**Greedy (beam_size=1):**
```
En cada paso: elige el token con mayor probabilidad y sigue adelante.
"para" → "el" → "robot"  [sigue siempre el camino más probable]
```

**Beam search (beam_size=N):**
```
Mantiene N hipótesis simultáneas en paralelo.
beam=3: mantiene los 3 caminos más prometedores y los explora en profundidad.

Paso 1: ["para", "parra", "par"]
Paso 2: ["para el", "para al", "parra el"]
Paso 3: ["para el robot", "para el rovot", "para al robot"]
Ganador: el de mayor score acumulado
```

**Trade-off:**
- `beam=1`: más rápido, puede quedar atrapado en mínimos locales (elegir "parra" en vez de "para" aunque "para el robot" sea mucho más probable en conjunto)
- `beam=5`: examina 5 veces más caminos, RTF aumenta ~linealmente, pero puede recuperar errores que greedy no detecta

**En frases cortas** ("para el robot"): la diferencia es pequeña, el contexto es corto.  
**En frases largas** ("detecta si hay obstáculos…"): beam mayor puede marcar diferencia porque el error se propaga a través de 20+ tokens.

---

### `best_of` — múltiples muestras independientes

Este parámetro solo tiene sentido con `temperature > 0`. Si el decodificador es estocástico, puedes ejecutarlo varias veces y quedarte con la mejor transcripción según su log-probabilidad.

```
best_of=3, temperature=0.2:
  Run 1: "para el robot"         (log-prob: -0.12)
  Run 2: "para el rovot"         (log-prob: -2.41)  ← error
  Run 3: "para el roboth"        (log-prob: -1.87)  ← error
  → Gana Run 1
```

Con `temperature=0.0` (determinista), `best_of` no tiene ningún efecto porque todas las ejecuciones producen el mismo resultado. Es un parámetro redundante en modo greedy/beam puro.

**Coste:** multiplica el tiempo de inferencia por `best_of`. En RPi4, `best_of=3` es prohibitivo salvo para benchmarking.

---

### `temperature` — aleatoriedad en el muestreo del decodificador

En cada paso, el decodificador calcula una distribución de probabilidad sobre todos los tokens posibles (softmax sobre logits). `temperature` escala los logits antes del softmax:

```
logits_escalados = logits / temperature

temperature = 0.0 → greedy (siempre el token de mayor probabilidad)
temperature = 0.2 → distribución casi determinista con pequeñas variaciones
temperature = 1.0 → distribución original del modelo
temperature > 1.0 → distribución más plana (más aleatoriedad, más errores)
```

**¿Para qué sirve temperature > 0 en STT?**  
Whisper tiene un mecanismo de **fallback automático**: si detecta que la transcripción tiene baja probabilidad (posible alucinación), sube la temperatura progresivamente (0.0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0) hasta que el resultado mejora. Esto se llama *temperature fallback* y está activo por defecto.

Para benchmarking fijo, usamos `temperature=0.0` para obtener resultados **reproducibles y deterministas**.

---

### `vad_filter` — filtro de actividad de voz (Voice Activity Detection)

Antes de pasarle el audio a Whisper, un preprocesador analiza qué partes del audio contienen voz real y cuáles son silencio o ruido de fondo.

**Sin VAD:**
```
[silencio 0.5s][voz 2.1s][silencio 0.8s] → Whisper procesa los 3.4s completos
```

**Con VAD (Silero VAD por defecto en faster-whisper):**
```
[silencio 0.5s][voz 2.1s][silencio 0.8s]
        ↓ VAD detecta segmentos de voz
Whisper solo procesa el segmento de 2.1s
```

**Beneficios en producción:**
1. **RTF mejora** al procesar menos audio
2. **WER mejora** porque Whisper no "alucina" texto en segmentos silenciosos (Whisper tiende a transcribir música o ruido como palabras)
3. **Menos consumo energético** (crítico en RPi4 con batería)

**`min_silence_duration_ms`** controla cuánto silencio continuo necesita el VAD para considerar que terminó un segmento de voz. Valores muy bajos cortan frases largas en medio; valores muy altos no filtran bien.

En el benchmark testamos 200 ms / 350 ms / 500 ms / 750 ms / 1000 ms para ver cómo afecta al WER de las frases largas (nivel 3).

---

## Resumen: ¿por qué Vosk no tiene estos parámetros?

| Capacidad | Whisper | Vosk |
|-----------|---------|------|
| Cuantización post-entrenamiento | Sí (int8/float16/float32) | No (modelo compilado fijo) |
| Decodificación beam search configurable | Sí | No (HMM Viterbi, algoritmo fijo) |
| Muestreo estocástico | Sí (temperature) | No (determinista siempre) |
| VAD integrado configurable | Sí | No (el usuario hace streaming, Vosk no decide) |
| Diseño | Generativo seq2seq | Discriminativo por etapas |

Vosk fue diseñado para **eficiencia en edge** con un pipeline fijo y predecible. Whisper fue diseñado como un modelo general de comprensión de audio, con toda la flexibilidad de un Transformer moderno.
