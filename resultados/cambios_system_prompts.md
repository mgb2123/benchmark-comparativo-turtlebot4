# Cambios en los System Prompts — Justificación

## Resumen

Se han eliminado las restricciones de longitud de respuesta de los system prompts.
El control de extensión pasa a ser responsabilidad exclusiva del parámetro `max_tokens`.

---

## Cambio 1 · `SYSTEM_PROMPT` (prompts funcionales)

| | Antes | Después |
|---|---|---|
| **Texto** | `"Eres Ana, un robot asistente. Responde SOLO en español, maximo 5 palabras."` | `"Eres Ana, un robot asistente. Responde SOLO en español."` |
| **Límite de palabras** | "maximo 5 palabras" | _(eliminado)_ |

### Por qué se eliminó

1. **Responsabilidad duplicada.** El benchmark ya usa `max_tokens` para acotar la longitud de generación. Poner además un límite en texto natural crea **dos restricciones que compiten** sin coordinación: el modelo podría obedecer la instrucción y cortar antes de `max_tokens`, o ignorarla y cortar por `max_tokens`, haciendo imprevisible qué mide realmente cada experimento.

2. **Distorsiona la comparativa.** Cuando `max_tokens` es grande (80, 120) el modelo debería poder generar respuestas largas para medir velocidad real de generación. Decirle "máximo 5 palabras" hace que se detenga antes, acortando artificialmente el tiempo medido y favoreciendo a modelos que interpretan mejor las instrucciones en español, no a los que generan más rápido.

3. **Sesgo de capacidad lingüística.** Modelos pequeños (Qwen-0.5B, TinyLlama) a veces no siguen instrucciones de conteo de palabras con precisión. Phi-2, al ser un modelo de completion sin entrenamiento de instrucción robusto, lo ignora directamente. El límite lingüístico mide obediencia a instrucciones, no rendimiento de inferencia.

4. **El escenario `max_tokens=10` ya cubre el caso de respuestas muy cortas.** No hace falta forzarlo desde el prompt.

---

## Cambio 2 · `SYSTEM_PROMPT_EMOCIONAL`

| | Antes | Después |
|---|---|---|
| **Frase eliminada** | `"Tus respuestas son breves pero emocionalmente significativas (máximo 3 frases)."` | _(eliminada)_ |
| **Resto** | Sin cambios | Sin cambios |

### Por qué se eliminó

1. **Contradice el objetivo de la evaluación emocional.** El propósito del bloque emocional es precisamente ver **hasta dónde se puede explayar** cada modelo cuando el contexto emocional lo requiere. Limitar a 3 frases en el propio prompt impide observar si el modelo tiene capacidad para respuestas más desarrolladas.

2. **`MAX_TOKENS_EMOCIONAL = 120` ya establece el techo.** Si el modelo quiere escribir más, `max_tokens` lo corta. Si escribe menos, esa es su respuesta natural — información válida para la evaluación subjetiva.

3. **"Máximo 3 frases" es una instrucción ambigua para LLMs pequeños.** ¿Es una frase terminada en punto? ¿En coma? Los modelos de 0.5B–2.7B no son fiables siguiendo esta meta-instrucción, lo que añade ruido sin control.

---

## Principio general adoptado

> **El system prompt define el rol y el idioma. `max_tokens` define la longitud máxima.**

Esto garantiza que cada punto del barrido paramétrico (`max_tokens` = 10, 20, 40, 60, 80, 120) mida exactamente lo que dice medir: el comportamiento del modelo a esa longitud de generación, sin interferencias del propio prompt.

---

## Impacto esperado en los resultados

| Escenario | Efecto esperado |
|-----------|-----------------|
| `max_tokens=10` | Respuestas muy cortas, como antes — sin cambio visible |
| `max_tokens=80-120` | Respuestas más largas y naturales, especialmente en Qwen y TinyLlama |
| Evaluación emocional | Respuestas potencialmente más ricas y desarrolladas |
| Phi-2 | Sin impacto adicional (ya corregido el formato de prompt por separado) |

---

*Aplicado en `bench_llm_escenarios.py` — `SYSTEM_PROMPT` y `SYSTEM_PROMPT_EMOCIONAL`*
