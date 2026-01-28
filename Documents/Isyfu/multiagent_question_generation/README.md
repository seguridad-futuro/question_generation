# Sistema Multiagente para Generación de Preguntas de Tipo Test

Sistema **automatizado y especializado** para generar preguntas de tipo test de alta calidad para las oposiciones de la **Guardia Civil española**, utilizando una arquitectura multi-agente con LangGraph, LangChain y OpenAI LLMs.

## 🎯 Descripción General

Este sistema implementa un **pipeline inteligente de 5 agentes especializados** que procesan documentos legales y generan preguntas de examen con validación de calidad de dos niveles, deduplicación semántica y persistencia en SQLite.

**Capacidades clave**:
- 🤖 **5 Agentes Especializados**: Chunking coherente (Z), Generación avanzada (B), Evaluación especializada (C), Persistencia (D) y Generación PDF (E)
- 📚 **Few-Shot Learning**: 6 ejemplos reales de exámenes oficiales de Guardia Civil
- 🎯 **8 Técnicas de Distracción Documentadas**: Basadas en análisis de 105 preguntas reales
- ✅ **Validación Dual**: Criterios especializados para oposiciones + métricas Ragas
- 🔄 **Sistema de Retry Inteligente**: Regeneración automática con feedback específico
- 🔍 **Deduplicación Semántica**: Detección de duplicados usando embeddings de OpenAI
- 📊 **Base de Datos Escalable**: Schema V2 compatible con Supabase
- 📄 **Generación de PDFs Profesionales**: Múltiples formatos de exportación

---

## 🏗️ Arquitectura del Sistema

El sistema implementa un patrón **Map-Reduce** con orquestación centralizada:

```
INPUT DOCUMENTS (PDFs, TXT, Markdown)
        ↓
    [Agent Z: Rewriter]
    Procesa PDFs → chunks coherentes
    Análisis de estructura
    Reescritura para coherencia
        ↓
    [Agent B: Generator] (Map - Paralelo)
    Para cada chunk:
      • Generar N preguntas (default: 5)
      • Few-shot learning + 8 técnicas
      • Referencias legales explícitas
        ↓
    [Agent C: Evaluator] (Reduce)
    Para cada pregunta:
      • Evaluación especializada (PRINCIPAL)
      • Métricas Ragas (COMPLEMENTARIAS)
      • Clasificación: auto_pass | manual_review | auto_fail
      • Retry inteligente (max 3 intentos)
        ↓
    [Agent D: Persistence]
    • Deduplicación semántica
    • Persistencia en SQLite
    • Estadísticas de duplicados
        ↓
    [Agent E: PDF Generator]
    • Múltiples formatos
    • Agrupación por tema
    • Estadísticas de dificultad
        ↓
OUTPUT: SQLite DB + PDFs + Estadísticas
```

---

## 📁 Estructura del Proyecto

```
multiagent_question_generation/
│
├── agents/                           # 5 Agentes especializados
│   ├── agent_z_rewriter.py          # Generación coherente de chunks (StateGraph + ReAct)
│   ├── agent_b_generator.py         # Generación de preguntas avanzada (few-shot)
│   ├── agent_c_evaluator.py         # Evaluación especializada + Ragas
│   ├── agent_d_persistence.py       # Persistencia y deduplicación
│   └── agent_e_pdf_generator.py     # Generación de PDFs profesionales
│
├── config/                           # Configuración del sistema
│   ├── settings.py                  # Settings globales (Pydantic BaseSettings)
│   ├── config_models.py             # Modelos de configuración estructurada
│   ├── thresholds.py                # Thresholds de calidad y deduplicación
│   └── prompts/                     # Prompts especializados (~30 archivos)
│       ├── agent_b_generation.json          # Few-shot + 8 técnicas
│       ├── agent_b_generation_system.txt    # System prompt
│       ├── agent_b_generation_user.txt      # User prompt
│       ├── agent_c_specialized_system.txt   # System (evaluador)
│       └── agent_c_specialized_user.txt     # User (evaluador)
│
├── database/                         # SQLite + Migraciones
│   ├── questions.db                 # Base de datos principal (~1.5MB)
│   ├── init_schema_v2.sql           # Schema V2 (Supabase-compatible)
│   ├── migrate_v1_to_v2.sql         # Script de migración
│   └── repository.py                # CRUD operations
│
├── graph/                            # Definición del grafo LangGraph
│   ├── state.py                     # GraphState TypedDict
│   └── workflow.py                  # Workflow orchestration
│
├── models/                           # Modelos de datos (Pydantic)
│   ├── chunk.py                     # Chunk con metadata y embeddings
│   ├── question.py                  # Question V1 (legacy)
│   ├── question_v2.py               # Question V2 (Supabase-compatible)
│   ├── question_option.py           # QuestionOption (relación 1-N)
│   └── quality_metrics.py           # Métricas de calidad (Ragas)
│
├── services/                         # Servicios de negocio
│   └── chunk_retriever.py           # Recuperación semántica de chunks
│
├── utils/                            # Utilidades
│   ├── llm_factory.py               # Factory pattern para OpenAI
│   ├── loaders.py                   # Cargadores de documentos
│   ├── embeddings.py                # Embeddings para deduplicación
│   ├── metadata_filter.py           # Limpieza de metadata de PDFs
│   ├── prompt_loader.py             # Cargador de prompts
│   ├── pdf_visualizer.py            # Visualización de chunks
│   └── rewrite_pattern_registry.py  # Patrones regex para limpieza
│
├── input_docs/                       # Documentos de entrada
│   ├── Tema-1-*.pdf                 # 23 temas del currículo de Guardia Civil
│   └── README.md
│
├── output/                           # Outputs generados
│   ├── questions_batch_*.json
│   ├── pdfs/                         # PDFs generados
│   └── chunks/                       # Chunks cacheados
│
├── examples/                         # Datos de referencia
│   ├── preguntas_guardia_civil.csv  # 105 preguntas reales analizadas
│   └── README.md
│
├── .env.example                      # Template de variables de entorno
├── requirements.txt                  # Dependencias Python
├── main.py                           # Orquestador principal (28KB)
├── run_bcde_pipeline.py             # Pipeline BCDE (sin Agent Z)
├── run_rewriter.py                   # Agent Z (standalone)
├── extract_and_generate_pdf.py      # Utilidad PDF
├── test_agent_z.py                   # Tests de Agent Z
├── test_agent_e_pdf.py               # Tests de Agent E
└── README.md                         # Este archivo
```

---

## 🚀 Instalación y Configuración

### 1. Requisitos Previos

- **Python 3.11+**
- **API Key de OpenAI**
- **Git** (para clonar el repositorio)

### 2. Clonar el Repositorio

```bash
git clone <repo-url>
cd multiagent_question_generation
```

### 3. Crear Entorno Virtual

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 4. Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Dependencias principales**:
- `langgraph>=0.3.0` - Orquestación multi-agente
- `langchain>=0.3.0` - Abstracciones LLM
- `langchain-openai>=0.3.0` - Integración OpenAI
- `pydantic>=2.0.0` - Validación de datos
- `ragas>=0.3.0` - Métricas de calidad
- `sentence-transformers>=2.3.1` - Embeddings
- `unstructured[pdf]>=0.16.0` - Procesamiento avanzado de PDFs
- `scikit-learn>=1.5.0` - Similitud coseno
- `tenacity>=8.0.0` - Retry logic

### 5. Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env con tu editor favorito
nano .env
```

**Variables obligatorias**:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o  # o gpt-5, gpt-4o-mini

# Agent Configuration
NUM_QUESTIONS_PER_CHUNK=5
NUM_ANSWER_OPTIONS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Quality Thresholds
QUALITY_THRESHOLD_FAITHFULNESS=0.85
QUALITY_THRESHOLD_RELEVANCY=0.85
QUALITY_THRESHOLD_AUTO_FAIL_FAITHFULNESS=0.60
QUALITY_THRESHOLD_AUTO_FAIL_RELEVANCY=0.60
MAX_RETRIES=3

# Deduplication
SIMILARITY_THRESHOLD=0.85
EMBEDDING_MODEL=text-embedding-3-small

# Database
DATABASE_PATH=database/questions.db
```

### 6. Inicializar Base de Datos

```bash
# Crear schema V2
sqlite3 database/questions.db < database/init_schema_v2.sql

# Verificar tablas
sqlite3 database/questions.db "SELECT name FROM sqlite_master WHERE type='table';"
```

**Salida esperada**:
```
questions
question_options
batches
questions_with_options
questions_full
batch_quality_stats
```

---

## 🎓 Descripción de los Agentes

### Agent Z: Coherent Chunk Rewriter

**Responsabilidad**: Procesar documentos PDF en chunks coherentes y semánticamente significativos.

**Características**:
- StateGraph con ReAct (Reasoning + Acting)
- Análisis de estructura de documentos
- Reescritura para coherencia
- Generación de metadata (source, page, tokens)
- Cálculo de "substantive ratio"

**Uso**:
```bash
python run_rewriter.py --input input_docs/
```

---

### Agent B: Advanced Question Generator

**Responsabilidad**: Generar preguntas de tipo test de alta calidad usando few-shot learning.

**Características Clave**:
- **Few-Shot Learning**: 6 ejemplos reales de exámenes de Guardia Civil
- **8 Técnicas de Distracción Documentadas**:
  1. Cambio de requisitos conjuntivos/disyuntivos (Y ↔ O)
  2. Manipulación de periodos de tiempo
  3. Eliminación de excepciones legales
  4. Atribución incorrecta de funciones
  5. Confusión de nivel normativo (ley vs. reglamento)
  6. Limitación falsa del ámbito
  7. Alteración de números/cifras
  8. Contexto con cambio del final

- **Paralelización**: Procesa múltiples chunks simultáneamente
- **Referencias Legales Obligatorias**: Cita explícita de artículos/leyes
- **Feedback Estructurado**: Formato oficial de oposiciones

**Formato de Salida**:
```json
{
  "question": "Según el Artículo 54 de la Constitución...",
  "options": [
    {"answer": "...", "is_correct": true, "option_order": 1},
    {"answer": "...", "is_correct": false, "option_order": 2},
    {"answer": "...", "is_correct": false, "option_order": 3},
    {"answer": "...", "is_correct": false, "option_order": 4}
  ],
  "tip": "La respuesta correcta es la a, porque...",
  "article": "Artículo 54 de la Constitución Española",
  "distractor_techniques": ["cambio_y_o", "atribución_incorrecta"]
}
```

---

### Agent C: Specialized Quality Evaluator

**Responsabilidad**: Validar la calidad de preguntas usando criterios especializados de oposiciones.

**Validación Dual**:

#### 1. Evaluación Especializada (PRINCIPAL)

**5 Criterios Obligatorios**:

| Criterio | Severidad | Descripción |
|----------|-----------|-------------|
| Referencias legales explícitas | CRÍTICO | Debe citar artículos/leyes específicas |
| Precisión técnica absoluta | CRÍTICO | Terminología legal exacta |
| Distractores plausibles | ALTO | Usando técnicas documentadas |
| Feedback estructurado | ALTO | Formato oficial con conectores |
| Formato oficial | MEDIO | Estructura estándar de oposiciones |

**Clasificación de tres zonas**:
- **auto_pass** (≥0.85): Aprobada automáticamente
- **manual_review** (0.60-0.85): Requiere revisión humana
- **auto_fail** (<0.60): Regenerar con feedback

#### 2. Métricas Ragas (COMPLEMENTARIAS)

Si la evaluación especializada no es concluyente:
- **faithfulness**: ¿La respuesta es fiel al contexto?
- **answer_relevancy**: ¿La respuesta es relevante a la pregunta?

---

### Agent D: Persistence & Deduplication

**Responsabilidad**: Eliminar duplicados semánticos y persistir preguntas en SQLite.

**Deduplicación Semántica**:
- Modelo: OpenAI `text-embedding-3-small`
- Métrica: Similitud coseno
- Threshold: 0.85 (configurable)
- Texto comparado: `question + correct_answer`

**Schema V2** (Compatible con Supabase):
```sql
-- Tabla principal
CREATE TABLE questions (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    tip TEXT,
    article TEXT,
    topic INTEGER,
    academy_id INTEGER,
    published BOOLEAN DEFAULT TRUE,

    -- Metadata LLM
    llm_model TEXT,
    generation_method TEXT,
    distractor_techniques TEXT,

    -- Quality tracking
    faithfulness_score REAL,
    relevancy_score REAL,
    source_chunk_id TEXT,
    source_document TEXT,

    -- Deduplication
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of INTEGER,
    needs_manual_review BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de opciones (relación 1-N)
CREATE TABLE question_options (
    id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL,
    answer TEXT NOT NULL,
    is_correct BOOLEAN DEFAULT FALSE,
    option_order INTEGER NOT NULL,

    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);
```

**Ventajas**:
- ✅ Compatible con Supabase/PostgreSQL
- ✅ Flexible: 3, 4, 5+ opciones
- ✅ Normalizada: sin redundancia
- ✅ Extensible: fácil agregar campos

---

### Agent E: PDF Generator

**Responsabilidad**: Generar PDFs profesionales con preguntas y soluciones.

**Características**:
- Múltiples formatos de exportación
- Agrupación por tema/tópico
- Estadísticas de dificultad
- Formato oficial de exámenes

**Formatos soportados**:
- Exam only (preguntas + opciones)
- With solutions (respuestas + explicaciones)
- Answers only (claves de respuestas)

---

## 📊 Modelos de Datos

### Question V2 (Supabase Compatible)

```python
from models.question_v2 import Question
from models.question_option import QuestionOption

question = Question(
    question="Según el Artículo 54 de la Constitución...",
    tip="La respuesta correcta es la a, porque...",
    article="Artículo 54 de la Constitución Española",
    topic=101,
    academy_id=1,
    generation_method="advanced_prompt",
    distractor_techniques=["cambio_y_o"],
    options=[
        QuestionOption(
            answer="El alto comisionado de las Cortes",
            is_correct=True,
            option_order=1
        ),
        # ... más opciones
    ]
)

# Métodos útiles
correct_answer = question.get_correct_answer_text()
correct_order = question.get_correct_answer_order()
all_answers = question.get_all_answers()
is_high_quality = question.is_high_quality()
```

### Chunk

```python
from models.chunk import Chunk

chunk = Chunk(
    chunk_id="chunk_001",
    content="Artículo 54. El Defensor del Pueblo...",
    source_document="constitucion.pdf",
    page=15,
    token_count=450,
    metadata={"section": "Título V", "coherent": True}
)
```

### QualityMetrics

```python
from models.quality_metrics import QualityMetrics

metrics = QualityMetrics(
    question_id="q_001",
    faithfulness=0.92,
    answer_relevancy=0.88,
    feedback="Alta calidad: respuesta fiel y relevante"
)

classification = metrics.get_classification()  # "auto_pass"
```

---

## 🎯 Flujo Completo del Pipeline

```
INPUT: ["documento.pdf"], academy=1, topic=101, num_questions=30

↓ [Agent Z: Chunking]
   Output: List[Chunk] con metadata

↓ [Agent B: Generation - MAP]
   Para cada chunk en paralelo:
   Output: Dict[chunk_id, List[Question]]

↓ [Agent C: Evaluation - REDUCE]
   Para cada pregunta:
   - Evaluación especializada (PRINCIPAL)
   - Ragas metrics (si no conclusiva)
   - Retry si auto_fail (max 3 intentos)
   Output: validated_questions, manual_review_questions, failed

↓ [Agent D: Persistence]
   - Deduplicación semántica
   - Persistencia en SQLite
   Output: persisted_ids, dedup_stats

↓ [Agent E: PDF Generation]
   - Generar PDFs profesionales
   Output: PDF files + statistics

FINAL: SQLite DB + PDFs + Estadísticas completas
```

---

## 🚀 Cómo Usar

### Opción 1: Pipeline Completo (A→B→C→D→E)

```bash
python main.py
```

**Opciones disponibles**:
```bash
python main.py --doc Tema-1-*.pdf           # Documentos específicos
python main.py --questions 10 --topic 101   # Con opciones
```

### Opción 2: Pipeline BCDE (sin Agent Z, usa chunks cacheados)

```bash
python run_bcde_pipeline.py --cache-dir output/chunks --num-questions 5
```

### Opción 3: Agent Z Standalone

```bash
python run_rewriter.py --input input_docs/
```

### Opción 4: Python programático

```python
from agents.agent_z_rewriter import RewriterAgent
from agents.agent_b_generator import generate_questions_for_chunk
from agents.agent_c_evaluator import create_agent_c_evaluator
from agents.agent_e_pdf_generator import PDFGeneratorAgent, PDFFormatEnum
from pathlib import Path

# 1. Generar chunks (Agent Z)
agent_z = RewriterAgent(use_llm=True)
chunks = agent_z.create_coherent_chunks(
    Path("input_docs/constitucion_española.pdf"),
    topic=101
)

# 2. Generar preguntas (Agent B)
all_questions = []
for chunk in chunks[:10]:
    questions = generate_questions_for_chunk(
        chunk=chunk,
        num_questions=5,
        topic="Tema 101"
    )
    all_questions.extend(questions)

# 3. Evaluar preguntas (Agent C)
agent_c = create_agent_c_evaluator()
validated_questions = []
for question in all_questions:
    evaluation = agent_c.evaluate(question)
    if evaluation["classification"] == "auto_pass":
        validated_questions.append(question)

# 4. Generar PDFs (Agent E)
agent_e = PDFGeneratorAgent()
agent_e.generate_pdfs(
    questions=validated_questions,
    output_dir=Path("output/pdfs"),
    pdf_format=PDFFormatEnum.WITH_SOLUTIONS,
    topic_filter=101
)
```

---

## ⚙️ Configuración Avanzada

### Ajustar Thresholds de Calidad

```bash
# Configuración estricta (menos aprobadas automáticamente)
QUALITY_THRESHOLD_FAITHFULNESS=0.90
QUALITY_THRESHOLD_RELEVANCY=0.90

# Configuración permisiva (más aprobadas)
QUALITY_THRESHOLD_FAITHFULNESS=0.80
QUALITY_THRESHOLD_RELEVANCY=0.80

# Recomendada (balanceada)
QUALITY_THRESHOLD_FAITHFULNESS=0.85
QUALITY_THRESHOLD_RELEVANCY=0.85
```

### Cambiar Número de Opciones

```bash
# Editar .env
NUM_ANSWER_OPTIONS=3  # Por defecto: 4

# El prompt avanzado está optimizado para 4 opciones
```

### Ajustar Deduplicación

```bash
# Más estricto (menos duplicados detectados)
SIMILARITY_THRESHOLD=0.90

# Menos estricto (más duplicados)
SIMILARITY_THRESHOLD=0.75
```

---

## 🛠️ Stack Tecnológico

### Core Framework
- **LangGraph** (≥0.3.0): Orquestación con StateGraph
- **LangChain** (≥0.3.0): Abstracciones LLM
- **Pydantic** (≥2.0.0): Validación de datos

### LLM
- **OpenAI API** (≥1.99.0): GPT-5, GPT-4o
- **langchain-openai** (≥0.3.0): Integración

### Embeddings & Similarity
- **sentence-transformers** (≥2.3.1): Embeddings multilingües
- **scikit-learn** (≥1.5.0): Similitud coseno
- **FAISS-CPU** (≥1.8.0): Vector search para multi-chunk

### Quality Evaluation
- **Ragas** (≥0.3.0): Métricas de calidad
- **datasets** (≥2.0.0): Requerido por Ragas

### Document Processing
- **unstructured[pdf]** (≥0.16.0): Procesamiento avanzado
- **pypdf** (≥4.0.0): Extracción de texto
- **pdfplumber** (≥0.10.0): Extracción de tablas
- **PyMuPDF** (≥1.23.0): Manipulación de PDFs
- **ReportLab** (≥4.0.0): Generación de PDFs

### Database
- **sqlite3**: Incluido en Python (sin instalación)

### Utilities
- **python-dotenv** (≥1.0.0): Env vars
- **tenacity** (≥8.0.0): Retry logic
- **tqdm** (≥4.66.0): Progress bars

---

## 🐛 Troubleshooting

### Error: "No such table: question_options"

```bash
sqlite3 database/questions.db < database/init_schema_v2.sql
```

### Error: "OPENAI_API_KEY not found"

```bash
# Editar .env
OPENAI_API_KEY=sk-...

# O exportar directamente
export OPENAI_API_KEY=sk-...
```

### Error: Ragas falla

```bash
pip install ragas datasets --upgrade
```

### Preguntas con baja calidad constante

1. Revisar prompts en `config/prompts/`
2. Ajustar thresholds en `.env`:
   ```bash
   QUALITY_THRESHOLD_FAITHFULNESS=0.80
   ```
3. Verificar chunks: quizás demasiado pequeños

### Error: "UNIQUE constraint failed: question_options"

Asegurar que cada pregunta tenga opciones con order 1, 2, 3, 4 únicos.

---

## 📚 Documentación Adicional

- **[ADVANCED_PROMPT_GUIDE.md](ADVANCED_PROMPT_GUIDE.md)**: Guía completa del prompt avanzado
- **[RESUMEN_ACTUALIZACION_V2.md](RESUMEN_ACTUALIZACION_V2.md)**: Actualización a estructura V2
- **[database/MIGRATION_GUIDE.md](database/MIGRATION_GUIDE.md)**: Migración V1→V2

---

## 🚦 Estado del Proyecto

### ✅ Completado

- [x] 5 Agentes especializados (Z, B, C, D, E)
- [x] Modelos de datos (Question V2, QuestionOption, Chunk, QualityMetrics)
- [x] Schema SQLite V2 (compatible con Supabase)
- [x] Prompt avanzado con few-shot learning
- [x] 8 técnicas de distracción documentadas
- [x] Evaluación especializada (5 criterios)
- [x] Métricas Ragas (faithfulness, relevancy)
- [x] Sistema de retry inteligente
- [x] Deduplicación semántica
- [x] Generación de PDFs profesionales
- [x] Documentación extensiva

### ⏳ En Desarrollo

- [ ] LangGraph workflow completo
- [ ] Batch processing paralelo optimizado
- [ ] Tests unitarios y de integración
- [ ] CLI mejorado
- [ ] Dashboard de métricas

### 🔮 Futuro

- [ ] DSPy prompt optimization automática
- [ ] Guardrails validation
- [ ] Caché de embeddings
- [ ] Interface web
- [ ] Logs estructurados
- [ ] Export a múltiples formatos
- [ ] Integración directa con Supabase
- [ ] Soporte para más LLM providers

---

## 👥 Contribución

Proyecto interno de **Isyfu** para generación automatizada de preguntas de oposiciones.

Para preguntas, sugerencias o reportar bugs, contactar al equipo de desarrollo.

---

## 📄 Licencia

**Propietario** - Isyfu © 2026

Todos los derechos reservados. Este software es propiedad de Isyfu y está protegido por leyes de copyright.

---

## 🙏 Agradecimientos

- Análisis basado en **105 preguntas reales** de exámenes oficiales de Guardia Civil
- Inspirado en best practices de sistemas de evaluación educativa
- Powered by **LangGraph**, **LangChain** y **OpenAI**

---

**Última actualización**: 2026-01-28
**Versión**: 2.1
**Estado**: ✅ Todos los agentes completados - Pipeline en producción