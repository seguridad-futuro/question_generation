# 🚀 Guía Completa: Cómo Ejecutar los Scripts de Generación

Documento detallado con todos los parámetros disponibles para ejecutar el pipeline de generación de preguntas.

---

## 📋 Índice

1. [main.py - Pipeline Completo (Z→B→C→D→E)](#mainpy)
2. [run_bcde_pipeline.py - Pipeline BCDE (sin Agent Z)](#run_bcde_pipelinepy)
3. [run_rewriter.py - Agent Z Standalone](#run_rewriterpy)
4. [Ejemplos Prácticos](#ejemplos-prácticos)
5. [Flujos Típicos](#flujos-típicos)

---

## main.py

**Descripción**: Orquestador principal que ejecuta el pipeline completo (Z→B→C→D→E).

Procesa PDFs desde `input_docs/`, los divide en chunks, genera preguntas, las valida, las persiste en SQLite y genera PDFs.

### Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--doc`, `-d` | string | N/A | Documento específico a procesar (nombre o path). Si no es absoluto, busca en `input_docs/` |
| `--questions`, `-q` | int | 5 | Número de preguntas a generar por documento |
| `--topic`, `-t` | int | 1 | ID del topic para metadata |
| `--academy`, `-a` | int | 1 | ID de la academia para metadata |
| `--list`, `-l` | flag | False | Lista documentos pendientes y sale |
| `--reset` | flag | False | Elimina `questions_done.csv` (permite reprocesar todo) |
| `--skip-agent-c` | flag | False | Salta validación del Agent C (Z→B→D→E, sin calidad) |

### Ejemplos

**Procesar todos los PDFs pendientes** (detecta automáticamente):
```bash
python main.py
```

**Procesar con parámetros específicos**:
```bash
python main.py --questions 10 --topic 101 --academy 1
```

**Procesar un archivo específico**:
```bash
python main.py --doc tema1.pdf
# o con path absoluto
python main.py --doc /ruta/completa/tema1.pdf
```

**Procesar sin validación del Agent C** (más rápido, menos confiable):
```bash
python main.py --skip-agent-c --questions 20
```

**Lista de documentos pendientes**:
```bash
python main.py --list
```

**Reset: limpiar tracking y reprocesar todo**:
```bash
python main.py --reset
# Luego procesa
python main.py
```

**Combinaciones avanzadas**:
```bash
# 15 preguntas por doc, topic 201, sin validación
python main.py --questions 15 --topic 201 --skip-agent-c

# Procesar un doc específico con validación
python main.py --doc constitucion.pdf --questions 5 --topic 101 --academy 2
```

### Logs esperados

```
█████████████████████████████████████████████████████████████████████████
█              PIPELINE DE GENERACIÓN DE PREGUNTAS                      █
█████████████████████████████████████████████████████████████████████████

📋 CONFIGURACIÓN:
   • Modelo LLM: gpt-4o
   • Topic: 101
   • Academy: 1
   • Preguntas/doc: 5
   • Agente C: ✅ ACTIVADO
   • Input dir: /ruta/input_docs
   • Database: /ruta/database/questions.db

📁 DOCUMENTOS A PROCESAR: 1
   1. tema1.pdf

█████████████████████████████████████████████████████████████████████████
█                        PROCESANDO: tema1.pdf                          █
█████████████████████████████████████████████████████████████████████████

📄 PASO 1: Generando chunks coherentes...
   ✅ 12 chunks coherentes generados (de 15 totales)

🤖 PASO 2-3: Generando y validando preguntas...
   ================================================================================
   📝 Pregunta candidata 1 | Chunks usados: chunk_001
   ================================================================================
      Pregunta: Según el Artículo 54...
      A) Opción 1
      B) Opción 2
      C) Opción 3
      D) Opción 4
      Correcta: B
      Artículo: Artículo 54 de la Constitución
      Tip: La respuesta correcta es la b, porque...
         🔍 Evaluación del Agente C...
         🧠 Agente C: auto_pass
         🔎 Scores: faithfulness=0.92, relevancy=0.90
         ✅ Agente C: Pregunta válida

💾 PASO 4: Persistiendo en SQLite...
   ✅ 5 preguntas persistidas
   ⏭️  0 duplicadas (omitidas)

📄 PASO 5: Generando PDF del examen...
   ✅ PDF generado: tema_101_exam.pdf
   📁 Ubicación: output/pdfs/tema_101/tema_101_exam.pdf

█████████████████████████████████████████████████████████████████████████
█                        RESUMEN FINAL                                  █
█████████████████████████████████████████████████████████████████████████

📊 ESTADÍSTICAS TOTALES:
   • Documentos procesados: 1
   • Chunks generados: 12
   • Preguntas generadas: 5
   • Preguntas persistidas: 5

📄 DETALLE POR DOCUMENTO:
   ✅ tema1.pdf: 5/5 preguntas

💾 ESTADO DE LA BASE DE DATOS:
   • Total preguntas: 5
   • Únicas: 5
   • Duplicadas: 0
   • Para revisión manual: 0
```

---

## run_bcde_pipeline.py

**Descripción**: Ejecuta el pipeline B→C→D→E usando chunks cacheados (salta Agent Z).

Requiere que antes hayas ejecutado `run_rewriter.py` para generar chunks cacheados en `input_docs/rewritten/`.

### Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--topic` | int | 1 | ID del topic |
| `--academy` | int | 1 | ID de la academia |
| `--doc` | string | N/A | Procesar solo este documento (stem/nombre sin extensión) |
| `--questions-per-chunk` | int | (de settings) | Preguntas a generar por chunk |
| `--skip-agent-c` | flag | False | Salta validación (B→D→E) |
| `--force` | flag | False | Ignora caché de chunks y procesa todos |
| `--reset-cache` | flag | False | Elimina caché de uso (permite reprocesar) |
| `--limit-chunks` | int | 0 | Limita chunks a procesar (0 = sin límite) |
| `--workers` | int | (de settings) | Número de agentes paralelos para procesamiento |

### Ejemplos

**Usar chunks cacheados (más rápido)**:
```bash
python run_bcde_pipeline.py --questions-per-chunk 5
```

**Procesar solo un documento**:
```bash
python run_bcde_pipeline.py --doc constitucion --questions-per-chunk 10
```

**Con paralelización (4 workers)**:
```bash
python run_bcde_pipeline.py --workers 4 --questions-per-chunk 5
```

**Sin validación del Agent C**:
```bash
python run_bcde_pipeline.py --skip-agent-c --questions-per-chunk 15
```

**Procesar solo primeros 10 chunks**:
```bash
python run_bcde_pipeline.py --limit-chunks 10 --questions-per-chunk 5
```

**Forzar reprocesamiento (ignorar caché)**:
```bash
python run_bcde_pipeline.py --force --questions-per-chunk 5
```

**Limpiar caché y reprocesar todo**:
```bash
python run_bcde_pipeline.py --reset-cache --force --questions-per-chunk 5
```

**Combinaciones avanzadas**:
```bash
# Procesar documento específico, 20 preguntas, con 2 workers paralelos
python run_bcde_pipeline.py --doc codigo_penal --questions-per-chunk 20 --workers 2

# Procesar primeros 20 chunks sin validación
python run_bcde_pipeline.py --limit-chunks 20 --skip-agent-c

# Topic 201, 4 preguntas por chunk, 8 workers paralelos
python run_bcde_pipeline.py --topic 201 --questions-per-chunk 4 --workers 8
```

### Logs esperados

```
📄 Procesando constitucion.json | chunks: 15
   ⚡ Procesando en paralelo con 4 agentes

   🧩 [W1] Chunk chunk_001 | pendientes: 5
      ✅ Pregunta 1 generada
      ✅ Pregunta 2 generada
      ...

   🧩 [W2] Chunk chunk_002 | pendientes: 5
      ✅ Pregunta 3 generada
      ...

   ⏳ Esperando a que el writer procese todos los items...
   ✅ Writer thread finalizado correctamente

💾 Persistidas: 20 | Duplicadas: 2
📄 PDF generado: tema_101_study_guide.pdf
```

### Requisitos previos

Debes haber ejecutado `run_rewriter.py` antes:
```bash
python run_rewriter.py --input input_docs/
# Esto crea caché en: input_docs/rewritten/*.json
```

---

## run_rewriter.py

**Descripción**: Ejecuta Agent Z para procesar PDFs y generar chunks coherentes cacheados.

Divide documentos en chunks semánticos significativos y los guarda en caché JSON para uso posterior.

### Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--input-dir` | string | (de settings) | Directorio con documentos de entrada |
| `--output-subdir` | string | "rewritten" | Subdirectorio de output (dentro de `input_dir`) |
| `--topic` | int | 1 | ID del topic para metadata |
| `--auto-topic` | flag | False | Infiere topic del nombre (busca "Tema-14" en nombre) |
| `--extensions` | string | "pdf,txt,md" | Extensiones a procesar (separadas por coma) |
| `--force` | flag | False | Procesa incluso si caché existe |
| `--list` | flag | False | Lista archivos pendientes y sale |
| `--limit` | int | 0 | Limita número de archivos a procesar (0 = sin límite) |
| `--limit-chunks` | int | 0 | Limita chunks por documento (0 = sin límite) |
| `--doc-timeout` | int | (de settings) | Timeout por documento en segundos (0 = sin timeout) |

### Ejemplos

**Procesar todos los PDFs pendientes**:
```bash
python run_rewriter.py
```

**Procesar con detección automática de topic**:
```bash
python run_rewriter.py --auto-topic
# Busca "Tema-14" en nombres y ajusta topic automáticamente
```

**Procesar solo primeros 5 archivos**:
```bash
python run_rewriter.py --limit 5
```

**Procesar extensiones específicas**:
```bash
python run_rewriter.py --extensions pdf,txt
# Solo PDF y TXT, ignora MD
```

**Listar archivos pendientes sin procesar**:
```bash
python run_rewriter.py --list
```

**Forzar reprocesamiento** (ignorar caché):
```bash
python run_rewriter.py --force
```

**Limitar chunks por documento**:
```bash
python run_rewriter.py --limit-chunks 20
# Genera máximo 20 chunks por documento
```

**Combinaciones avanzadas**:
```bash
# Auto-topic, solo PDFs, máximo 10 archivos
python run_rewriter.py --auto-topic --extensions pdf --limit 10

# Forzar reproceso, máximo 5 chunks por doc
python run_rewriter.py --force --limit-chunks 5

# 20 segundos de timeout, auto-topic, sin límite
python run_rewriter.py --doc-timeout 20 --auto-topic

# Solo TXT, primeros 3 archivos, 15 chunks máximo
python run_rewriter.py --extensions txt --limit 3 --limit-chunks 15
```

### Logs esperados

```
[1/5] Processing: tema1.pdf
   ✅ Procesado correctamente
   📊 Chunks generados: 12
   💾 Caché guardado: input_docs/rewritten/tema1.json

[2/5] Processing: tema2.pdf
   ✅ Procesado correctamente
   📊 Chunks generados: 15
   💾 Caché guardado: input_docs/rewritten/tema2.json

Summary:
 - processed: 5
 - failed: 0
```

---

## Ejemplos Prácticos

### Ejemplo 1: Flujo completo desde cero

```bash
# 1. Procesar PDFs → chunks cacheados
python run_rewriter.py --auto-topic

# 2. Usar chunks para generar preguntas (rápido)
python run_bcde_pipeline.py --questions-per-chunk 5 --workers 4

# 3. Ver lo que se generó
sqlite3 database/questions.db "SELECT COUNT(*) FROM questions;"
```

### Ejemplo 2: Quick test con pocos PDFs

```bash
# Procesar solo 1 PDF, solo 5 chunks
python run_rewriter.py --limit 1 --limit-chunks 5

# Generar 3 preguntas por chunk
python run_bcde_pipeline.py --questions-per-chunk 3

# Ver resultado
ls -lh output/pdfs/
```

### Ejemplo 3: Procesar documento específico completo

```bash
# Opción A: Todo en un comando
python main.py --doc constitucion.pdf --questions 10 --topic 101

# Opción B: Por pasos
python run_rewriter.py --limit 1
python run_bcde_pipeline.py --doc constitucion --questions-per-chunk 5
```

### Ejemplo 4: Reprocesar con diferentes configuraciones

```bash
# Primera pasada: pocos chunks
python run_rewriter.py --limit 2 --limit-chunks 10

# Segunda pasada: más preguntas por chunk
python run_bcde_pipeline.py --questions-per-chunk 10

# Tercera pasada: sin validación (más rápido)
python run_bcde_pipeline.py --skip-agent-c --questions-per-chunk 20
```

### Ejemplo 5: Procesar en paralelo (4 workers)

```bash
# Con 4 workers simultáneos
python run_bcde_pipeline.py --workers 4 --questions-per-chunk 5

# Verificar uso de CPU: debe estar en ~400% en 4 cores
```

### Ejemplo 6: Reset completo y reprocesar

```bash
# Limpiar todo
python main.py --reset
rm -rf input_docs/rewritten/
rm -rf output/chunks/

# Reprocesar desde cero
python run_rewriter.py --force
python run_bcde_pipeline.py --reset-cache --force --questions-per-chunk 5
```

---

## Flujos Típicos

### Flujo A: Quick Prototype (desarrollo rápido)

```bash
# 1. Procesar 1 PDF, 5 chunks máximo
python run_rewriter.py --limit 1 --limit-chunks 5

# 2. Generar preguntas rápidas (sin Agent C)
python run_bcde_pipeline.py --questions-per-chunk 2 --skip-agent-c

# 3. Ver resultado
cat output/pdfs/tema_*/tema_*.pdf
```

**Tiempo total**: ~2-3 minutos

---

### Flujo B: Producción Completa

```bash
# 1. Procesar todos los PDFs
python run_rewriter.py --auto-topic

# 2. Generar preguntas con validación
python run_bcde_pipeline.py --questions-per-chunk 5 --workers 4

# 3. Verificar base de datos
sqlite3 database/questions.db << EOF
SELECT COUNT(*) as total FROM questions;
SELECT COUNT(DISTINCT source_document) as docs FROM questions;
SELECT AVG(faithfulness_score) as avg_faithfulness FROM questions WHERE faithfulness_score IS NOT NULL;
EOF

# 4. Ver PDFs generados
ls -lh output/pdfs/*/
```

**Tiempo total**: ~30-60 minutos (depende de cantidad de PDFs)

---

### Flujo C: Iteración sobre documento específico

```bash
# 1. Procesar document específico
python run_rewriter.py --limit 1
# ✅ Genera caché para ese documento

# 2. Generar preguntas (intento 1)
python run_bcde_pipeline.py --doc documento --questions-per-chunk 3

# 3. Ver resultados, revisar...
# 4. Reprocesar con diferentes parámetros
python run_bcde_pipeline.py --doc documento --reset-cache --force --questions-per-chunk 5

# 5. Comparar resultados
sqlite3 database/questions.db "SELECT * FROM questions WHERE source_document LIKE '%documento%';"
```

---

### Flujo D: Sin validación (máxima velocidad)

```bash
# Agent Z
python run_rewriter.py

# Saltarse Agent C (Z → B → D → E)
python run_bcde_pipeline.py --skip-agent-c --workers 8 --questions-per-chunk 20

# ⚠️ Nota: Preguntas sin validación de calidad
```

**Ventaja**: 3-5x más rápido
**Desventaja**: Menor calidad de preguntas

---

## Tabla Comparativa de Velocidad

| Comando | Speed | Calidad | Caso de uso |
|---------|-------|---------|-----------|
| `main.py` (con Agent C) | Lento | ⭐⭐⭐⭐⭐ | Producción |
| `main.py --skip-agent-c` | Rápido | ⭐⭐⭐ | Draft/testing |
| `run_bcde_pipeline.py` | Rápido | ⭐⭐⭐⭐ | Iteración rápida |
| `run_bcde_pipeline.py --skip-agent-c` | Muy rápido | ⭐⭐⭐ | Prototipo |
| `run_rewriter.py` | N/A | N/A | Preparación de datos |

---

## Troubleshooting

### Error: "No cached chunk files found"

```bash
# Solución: Ejecutar Agent Z primero
python run_rewriter.py --input input_docs/
```

### Error: "No pending files to process"

```bash
# Opción 1: Archivos ya procesados, forzar reprocess
python run_rewriter.py --force

# Opción 2: No hay archivos en input_docs/
cp /ruta/a/tus/pdfs/* input_docs/
python run_rewriter.py
```

### Error: "No such table: question_options"

```bash
# Solución: Inicializar base de datos
sqlite3 database/questions.db < database/init_schema_v2.sql
```

### Programa se cuelga en Agent C

```bash
# Solución: Saltarse Agent C
python run_bcde_pipeline.py --skip-agent-c
# Luego investiga Agent C en logs
```

### Memoria muy alta con muchos workers

```bash
# Solución: Reducir número de workers
python run_bcde_pipeline.py --workers 2
# o sin paralelización
python run_bcde_pipeline.py --workers 1
```

---

## Tips de Optimización

### Para máxima velocidad:
```bash
python run_rewriter.py --limit-chunks 10
python run_bcde_pipeline.py --skip-agent-c --workers 8 --questions-per-chunk 20
```

### Para máxima calidad:
```bash
python run_rewriter.py
python run_bcde_pipeline.py --questions-per-chunk 5 --workers 1
# Sin --skip-agent-c (activado por default)
```

### Para máxima paralelización:
```bash
python run_bcde_pipeline.py --workers $(nproc) --questions-per-chunk 10
# $(nproc) = número de cores en tu máquina
```

---

## Resumen Rápido

| Necesitas... | Comando |
|--------------|---------|
| Pipeline completo (recomendado) | `python main.py` |
| Chunks → preguntas (rápido) | `python run_bcde_pipeline.py` |
| Procesar PDFs solamente | `python run_rewriter.py` |
| Listar pendientes | `python main.py --list` |
| Reprocesar todo | `python main.py --reset` |
| Sin validación (rápido) | `python main.py --skip-agent-c` |
| Con paralelización | `python run_bcde_pipeline.py --workers 4` |
| Documento específico | `python main.py --doc archivo.pdf` |
