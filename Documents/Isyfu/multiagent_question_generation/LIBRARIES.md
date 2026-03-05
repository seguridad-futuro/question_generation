# Guía de Librerías del Sistema Multiagente

Esta guía explica las librerías principales que usamos en el sistema de generación de preguntas, con ejemplos simples y prácticos.

---

## 📊 Índice

1. [LangGraph - Orquestación de Agentes](#1-langgraph---orquestación-de-agentes)
2. [LangChain - Abstracciones LLM](#2-langchain---abstracciones-llm)
3. [Ragas - Evaluación de Calidad](#3-ragas---evaluación-de-calidad)
4. [DSPy - Prompt Engineering Automático](#4-dspy---prompt-engineering-automático)
5. [Sentence Transformers - Embeddings](#5-sentence-transformers---embeddings)
6. [Unstructured - Procesamiento de PDFs](#6-unstructured---procesamiento-de-pdfs)
7. [Guardrails AI - Validación de Outputs](#7-guardrails-ai---validación-de-outputs)
8. [Pydantic - Validación de Datos](#8-pydantic---validación-de-datos)

---

## 1. LangGraph - Orquestación de Agentes

### ¿Qué es?
Framework para crear aplicaciones multi-agente usando grafos. Permite definir flujos complejos con estado compartido, loops y condicionales.

### ¿Por qué lo usamos?
- Orquestar los 4 agentes del pipeline
- Manejar retry logic automático
- Estado compartido entre agentes
- Ejecución paralela (Map-Reduce)

### Ejemplo Simple: Grafo de 2 Nodos

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. Definir el estado compartido
class State(TypedDict):
    input: str
    output: str
    count: int

# 2. Definir funciones de los nodos
def node_a(state: State) -> State:
    """Procesa el input"""
    state["output"] = state["input"].upper()
    state["count"] = len(state["input"])
    return state

def node_b(state: State) -> State:
    """Añade información adicional"""
    state["output"] = f"{state['output']} (length: {state['count']})"
    return state

# 3. Crear el grafo
workflow = StateGraph(State)

# 4. Añadir nodos
workflow.add_node("process", node_a)
workflow.add_node("format", node_b)

# 5. Definir flujo
workflow.set_entry_point("process")
workflow.add_edge("process", "format")
workflow.add_edge("format", END)

# 6. Compilar y ejecutar
graph = workflow.compile()

# Ejecutar
result = graph.invoke({
    "input": "hello world",
    "output": "",
    "count": 0
})

print(result)
# Output: {'input': 'hello world', 'output': 'HELLO WORLD (length: 11)', 'count': 11}
```

### Ejemplo con Condicional (Retry Logic)

```python
from langgraph.graph import StateGraph, END

class RetryState(TypedDict):
    number: int
    is_valid: bool
    retry_count: int
    max_retries: int

def validate_number(state: RetryState) -> RetryState:
    """Valida si el número es par"""
    state["is_valid"] = (state["number"] % 2 == 0)
    return state

def increment_number(state: RetryState) -> RetryState:
    """Incrementa el número y el contador de retry"""
    state["number"] += 1
    state["retry_count"] += 1
    return state

def should_retry(state: RetryState) -> str:
    """Decide si hacer retry o terminar"""
    if not state["is_valid"] and state["retry_count"] < state["max_retries"]:
        return "retry"
    return "end"

# Crear grafo con loop
workflow = StateGraph(RetryState)
workflow.add_node("validate", validate_number)
workflow.add_node("increment", increment_number)

workflow.set_entry_point("validate")

# Conditional edge: si no válido → retry, si válido → end
workflow.add_conditional_edges(
    "validate",
    should_retry,
    {
        "retry": "increment",
        "end": END
    }
)

workflow.add_edge("increment", "validate")  # Loop de retry

graph = workflow.compile()

# Ejecutar con número impar
result = graph.invoke({
    "number": 5,
    "is_valid": False,
    "retry_count": 0,
    "max_retries": 3
})

print(result)
# Output: {'number': 6, 'is_valid': True, 'retry_count': 1, 'max_retries': 3}
# Hizo retry una vez: 5 → 6 (par)
```

### Uso en Nuestro Proyecto

```python
# Pipeline completo
workflow = StateGraph(GraphState)

# Añadir agentes
workflow.add_node("chunking", agent_z_coherent_chunking)
workflow.add_node("generation", agent_b_generation)
workflow.add_node("quality_gate", agent_c_quality)
workflow.add_node("persistence", agent_d_persistence)

# Flujo básico
workflow.set_entry_point("chunking")
workflow.add_edge("chunking", "generation")
workflow.add_edge("generation", "quality_gate")

# Retry logic
def should_retry_questions(state: GraphState) -> str:
    if state["failed_questions"] and any(
        state["retry_counts"].get(chunk_id, 0) < state["max_retries"]
        for chunk_id in state["failed_questions"]
    ):
        return "generation"  # Retry
    return "persistence"

workflow.add_conditional_edges(
    "quality_gate",
    should_retry_questions,
    {
        "generation": "generation",
        "persistence": "persistence"
    }
)

workflow.add_edge("persistence", END)
```

---

## 2. LangChain - Abstracciones LLM

### ¿Qué es?
Framework para aplicaciones con LLMs. Proporciona abstracciones para prompts, chains, document loaders y más.

### ¿Por qué lo usamos?
- Chat con OpenAI de forma simple
- Cargar y dividir documentos (PDFs)
- Templates de prompts reutilizables
- Chains para combinar operaciones

### Ejemplo 1: Chat Básico con OpenAI

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Crear cliente LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key="sk-..."
)

# Ejecutar chat
messages = [
    SystemMessage(content="Eres un experto en generar preguntas."),
    HumanMessage(content="Genera una pregunta sobre la capital de España.")
]

response = llm.invoke(messages)
print(response.content)
# Output: "¿Cuál es la capital de España? a) Madrid b) Barcelona c) Valencia d) Sevilla"
```

### Ejemplo 2: Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Template reutilizable
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en {subject}."),
    ("user", "Genera {num} preguntas sobre: {topic}")
])

llm = ChatOpenAI(model="gpt-4")

# Chain: prompt + llm
chain = prompt | llm

# Usar con diferentes inputs
response = chain.invoke({
    "subject": "Derecho Penal",
    "num": 3,
    "topic": "delitos de homicidio"
})

print(response.content)
```

### Ejemplo 3: Document Loaders (Cargar PDFs)

```python
from langchain_community.document_loaders import PyPDFLoader

# Cargar PDF
loader = PyPDFLoader("input_docs/tema_1.pdf")
documents = loader.load()

# Ver contenido
for doc in documents[:2]:  # Primeras 2 páginas
    print(f"Página {doc.metadata['page']}:")
    print(doc.page_content[:200])  # Primeros 200 caracteres
    print("---")

# Output:
# Página 0:
# El Código Penal español regula los delitos contra las personas...
# ---
# Página 1:
# Artículo 138: El homicidio se castiga con pena de prisión...
# ---
```

### Ejemplo 4: Text Splitters (Dividir en Chunks)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Cargar documento
loader = TextLoader("input_docs/tema.txt")
documents = loader.load()

# Crear splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Tamaño del chunk
    chunk_overlap=100,     # Overlap entre chunks
    separators=["\n\n", "\n", ". ", " ", ""]  # Prioridad de separadores
)

# Dividir
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i}:")
    print(chunk.page_content)
    print(f"Metadata: {chunk.metadata}")

# Output:
# Total chunks: 12
# Chunk 0:
# El Código Penal español regula los delitos contra las personas...
# Metadata: {'source': 'tema.txt'}
```

### Uso en Nuestro Proyecto (Agente A)

```python
from pathlib import Path
from agents.agent_z_rewriter import RewriterAgent

def agent_z_coherent_chunking(state: GraphState) -> GraphState:
    """Agente Z: Genera chunks coherentes con análisis de estructura"""

    all_chunks = []

    for doc_path in state["input_docs"]:
        # Crear agente Z
        agent_z = RewriterAgent(use_llm=False)

        # Generar chunks coherentes
        chunks = agent_z.create_coherent_chunks(
            Path(doc_path),
            topic=state["metadata"].get("topic", 1)
        )
        raw_chunks = splitter.split_documents(documents)

        # Convertir a modelo Chunk
        for i, chunk in enumerate(raw_chunks):
            all_chunks.append(Chunk(
                chunk_id=f"{doc_path}__chunk_{i}",
                content=chunk.page_content,
                source_document=doc_path,
                page=chunk.metadata.get("page"),
                token_count=len(chunk.page_content) // 4
            ))

    state["chunks"] = all_chunks
    state["current_step"] = "chunking_complete"
    return state
```

---

## 3. Ragas - Evaluación de Calidad

### ¿Qué es?
Framework para evaluar calidad de sistemas RAG usando métricas automáticas basadas en LLMs.

### ¿Por qué lo usamos?
- Medir si las preguntas generadas son fieles al contexto (faithfulness)
- Medir si las respuestas son relevantes (answer relevancy)
- Clasificar automáticamente: aprobar, rechazar o revisar manualmente

### Ejemplo Simple: Evaluar Faithfulness

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Datos de evaluación
data = {
    "question": [
        "¿Qué artículo del Código Penal tipifica el homicidio?"
    ],
    "answer": [
        "El artículo 138 del Código Penal tipifica el homicidio."
    ],
    "contexts": [
        [
            "Artículo 138 del Código Penal: El que matare a otro será castigado, "
            "como reo de homicidio, con la pena de prisión de diez a quince años."
        ]
    ]
}

# Convertir a Dataset
dataset = Dataset.from_dict(data)

# Evaluar
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)

print(result)
# Output:
# {'faithfulness': 0.95, 'answer_relevancy': 0.92}
```

### Ejemplo: Clasificar según Scores

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

def classify_question(faithfulness_score, relevancy_score):
    """Clasifica pregunta según thresholds"""

    # Auto-pass: Alta calidad
    if faithfulness_score >= 0.85 and relevancy_score >= 0.85:
        return "auto_pass"

    # Auto-fail: Baja calidad → Regenerar
    elif faithfulness_score < 0.60 or relevancy_score < 0.60:
        return "auto_fail"

    # Zona gris: Revisión manual
    else:
        return "manual_review"

# Evaluar varias preguntas
questions = [
    {
        "question": "¿Qué es el homicidio?",
        "answer": "Es matar a otra persona.",
        "contexts": [["Art. 138: El que matare a otro será castigado como reo de homicidio."]]
    },
    {
        "question": "¿Cuál es la capital de Francia?",  # Irrelevante
        "answer": "París",
        "contexts": [["Art. 138: El que matare a otro será castigado como reo de homicidio."]]
    }
]

for q in questions:
    dataset = Dataset.from_dict({
        "question": [q["question"]],
        "answer": [q["answer"]],
        "contexts": q["contexts"]
    })

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    classification = classify_question(
        result["faithfulness"],
        result["answer_relevancy"]
    )

    print(f"Pregunta: {q['question']}")
    print(f"  Faithfulness: {result['faithfulness']:.2f}")
    print(f"  Relevancy: {result['answer_relevancy']:.2f}")
    print(f"  Clasificación: {classification}\n")

# Output:
# Pregunta: ¿Qué es el homicidio?
#   Faithfulness: 0.92
#   Relevancy: 0.89
#   Clasificación: auto_pass
#
# Pregunta: ¿Cuál es la capital de Francia?
#   Faithfulness: 0.15
#   Relevancy: 0.10
#   Clasificación: auto_fail
```

### Uso en Nuestro Proyecto (Agente C)

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from models.quality_metrics import QualityMetrics

def agent_c_quality(state: GraphState) -> GraphState:
    """Agente C: Evalúa calidad con Ragas"""

    for chunk_id, questions in state["generated_questions"].items():
        chunk = next(c for c in state["chunks"] if c.chunk_id == chunk_id)

        for question in questions:
            # Preparar datos para Ragas
            data = {
                "question": [question.question],
                "answer": [question.get_correct_answer()],
                "contexts": [[chunk.content]]
            }

            dataset = Dataset.from_dict(data)

            # Evaluar
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

            # Crear métricas
            metrics = QualityMetrics(
                question_id=question.source_chunk_id,
                faithfulness=result["faithfulness"],
                answer_relevancy=result["answer_relevancy"],
                context=chunk.content
            )

            # Clasificar
            classification = metrics.get_classification()

            if classification == "auto_pass":
                question.faithfulness_score = metrics.faithfulness
                question.relevancy_score = metrics.answer_relevancy
                state["validated_questions"].append(question)

            elif classification == "auto_fail":
                # Marcar para retry
                if chunk_id not in state["failed_questions"]:
                    state["failed_questions"][chunk_id] = []

                state["failed_questions"][chunk_id].append({
                    "question": question,
                    "feedback": f"Scores bajos: F={metrics.faithfulness:.2f}, R={metrics.answer_relevancy:.2f}"
                })

            else:  # manual_review
                question.needs_manual_review = True
                state["manual_review_questions"].append(question)

    return state
```

---

## 4. DSPy - Prompt Engineering Automático

### ¿Qué es?
Framework para optimizar prompts automáticamente usando ejemplos. En vez de escribir prompts manualmente, defines la tarea y DSPy genera el mejor prompt.

### ¿Por qué lo usamos?
- Optimización automática de prompts para generación de preguntas
- Few-shot learning automático
- Chain of Thought integrado
- Mejora continua con ejemplos

### Ejemplo Simple: Signature (Input/Output)

```python
import dspy

# Configurar LLM
lm = dspy.OpenAI(model='gpt-4', api_key='sk-...')
dspy.settings.configure(lm=lm)

# Definir signature: qué inputs y outputs esperas
class GenerateSummary(dspy.Signature):
    """Genera un resumen de un texto."""

    text = dspy.InputField(desc="Texto largo a resumir")
    max_words = dspy.InputField(desc="Máximo de palabras")

    summary = dspy.OutputField(desc="Resumen conciso")

# Usar predictor básico
predictor = dspy.Predict(GenerateSummary)

# Ejecutar
result = predictor(
    text="El Código Penal español es un conjunto de normas que regulan "
         "los delitos y las penas. Fue aprobado en 1995 y ha sido modificado "
         "en múltiples ocasiones.",
    max_words="15"
)

print(result.summary)
# Output: "Código Penal español: normas sobre delitos y penas, aprobado en 1995"
```

### Ejemplo: Chain of Thought (Razonamiento Paso a Paso)

```python
import dspy

lm = dspy.OpenAI(model='gpt-4')
dspy.settings.configure(lm=lm)

class AnswerQuestion(dspy.Signature):
    """Responde preguntas con razonamiento."""

    context = dspy.InputField(desc="Contexto con información")
    question = dspy.InputField(desc="Pregunta a responder")

    reasoning = dspy.OutputField(desc="Razonamiento paso a paso")
    answer = dspy.OutputField(desc="Respuesta final")

# Chain of Thought añade razonamiento intermedio
cot = dspy.ChainOfThought(AnswerQuestion)

result = cot(
    context="El artículo 138 del Código Penal tipifica el homicidio con penas de 10 a 15 años.",
    question="¿Cuál es la pena por homicidio?"
)

print("Razonamiento:", result.reasoning)
print("Respuesta:", result.answer)

# Output:
# Razonamiento: El contexto menciona el artículo 138 que regula el homicidio.
#                Indica que la pena es de 10 a 15 años de prisión.
# Respuesta: La pena por homicidio es de 10 a 15 años de prisión.
```

### Ejemplo: Optimización con Few-Shot

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

lm = dspy.OpenAI(model='gpt-4')
dspy.settings.configure(lm=lm)

# Definir módulo
class QuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("context -> question, answer1, answer2, answer3, answer4, correct")

    def forward(self, context):
        return self.generate(context=context)

# Ejemplos de entrenamiento
trainset = [
    dspy.Example(
        context="Art. 138: El homicidio se castiga con 10 a 15 años de prisión.",
        question="¿Cuál es la pena por homicidio según el CP?",
        answer1="10 a 15 años",
        answer2="5 a 10 años",
        answer3="15 a 20 años",
        answer4="20 a 25 años",
        correct="1"
    ).with_inputs("context"),

    dspy.Example(
        context="Art. 139: El asesinato se castiga con 15 a 25 años de prisión.",
        question="¿Cuál es la pena por asesinato?",
        answer1="10 a 15 años",
        answer2="15 a 25 años",
        answer3="5 a 10 años",
        answer4="25 a 30 años",
        correct="2"
    ).with_inputs("context"),
]

# Métrica de evaluación
def validate_question(example, pred, trace=None):
    return example.correct == pred.correct

# Optimizar automáticamente
optimizer = BootstrapFewShot(metric=validate_question, max_bootstrapped_demos=2)
optimized_generator = optimizer.compile(QuestionGenerator(), trainset=trainset)

# Usar versión optimizada
result = optimized_generator(
    context="Art. 140: El homicidio imprudente se castiga con 1 a 4 años."
)

print(result)
```

### Uso en Nuestro Proyecto (Agente B)

```python
import dspy

# Configurar
lm = dspy.OpenAI(model='gpt-4')
dspy.settings.configure(lm=lm)

# Signature para generación de preguntas
class GenerateTestQuestion(dspy.Signature):
    """Generar pregunta tipo test de oposiciones."""

    context = dspy.InputField(desc="Texto del tema")
    topic = dspy.InputField(desc="Tema específico")
    difficulty = dspy.InputField(desc="Nivel: fácil, medio, difícil")

    question = dspy.OutputField(desc="Pregunta formulada")
    answer1 = dspy.OutputField(desc="Opción 1")
    answer2 = dspy.OutputField(desc="Opción 2")
    answer3 = dspy.OutputField(desc="Opción 3")
    answer4 = dspy.OutputField(desc="Opción 4")
    correct = dspy.OutputField(desc="Número de respuesta correcta (1-4)")
    tip = dspy.OutputField(desc="Explicación breve")

# Módulo con Chain of Thought
class QuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateTestQuestion)

    def forward(self, context, topic, difficulty):
        return self.generate(context=context, topic=topic, difficulty=difficulty)

# Usar en el agente
def agent_b_generation(state: GraphState) -> GraphState:
    generator = QuestionGenerator()

    for chunk in state["chunks"]:
        result = generator(
            context=chunk.content,
            topic=state["metadata"]["topic"],
            difficulty="medio"
        )

        # Convertir a Question model...
        question = Question(
            question=result.question,
            answer1=result.answer1,
            # ...
        )

    return state
```

---

## 5. Sentence Transformers - Embeddings

### ¿Qué es?
Modelos preentrenados para convertir texto en vectores numéricos (embeddings) que capturan significado semántico.

### ¿Por qué lo usamos?
- Deduplicación: detectar preguntas similares/duplicadas
- Comparación semántica usando similitud coseno
- Modelos optimizados para español

### Ejemplo Simple: Generar Embeddings

```python
from sentence_transformers import SentenceTransformer

# Cargar modelo multilingüe (incluye español)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Textos a convertir
sentences = [
    "¿Cuál es la capital de España?",
    "¿Qué ciudad es la capital española?",
    "El perro come comida."
]

# Generar embeddings
embeddings = model.encode(sentences)

print(f"Shape: {embeddings.shape}")  # (3, 384) - 3 frases, 384 dimensiones
print(f"Primer embedding: {embeddings[0][:5]}...")  # Primeros 5 valores

# Output:
# Shape: (3, 384)
# Primer embedding: [0.234, -0.567, 0.123, 0.890, -0.234]...
```

### Ejemplo: Calcular Similitud

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Frases a comparar
sentence1 = "¿Cuál es la capital de España?"
sentence2 = "¿Qué ciudad es la capital española?"
sentence3 = "El perro come comida."

# Generar embeddings
embeddings = model.encode([sentence1, sentence2, sentence3])

# Calcular similitud coseno
similarity_1_2 = cosine_similarity(
    embeddings[0].reshape(1, -1),
    embeddings[1].reshape(1, -1)
)[0][0]

similarity_1_3 = cosine_similarity(
    embeddings[0].reshape(1, -1),
    embeddings[2].reshape(1, -1)
)[0][0]

print(f"Similitud entre pregunta 1 y 2: {similarity_1_2:.3f}")
print(f"Similitud entre pregunta 1 y 3: {similarity_1_3:.3f}")

# Output:
# Similitud entre pregunta 1 y 2: 0.921  (MUY similares - duplicados)
# Similitud entre pregunta 1 y 3: 0.089  (Nada similares)
```

### Ejemplo: Detectar Duplicados

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Lista de preguntas
questions = [
    "¿Qué artículo regula el homicidio?",
    "¿Cuál es el artículo sobre homicidio en el CP?",  # Duplicado de la 1
    "¿Cuántos años de prisión por asesinato?",
    "¿Qué es el homicidio?",
    "¿Qué pena tiene el asesinato?"  # Similar a la 3
]

# Generar embeddings
embeddings = model.encode(questions)

# Detectar duplicados
threshold = 0.85
duplicates = []

for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        similarity = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[j].reshape(1, -1)
        )[0][0]

        if similarity >= threshold:
            duplicates.append({
                "index_1": i,
                "index_2": j,
                "question_1": questions[i],
                "question_2": questions[j],
                "similarity": similarity
            })

# Mostrar duplicados
for dup in duplicates:
    print(f"\n🔴 DUPLICADO DETECTADO (similitud: {dup['similarity']:.3f})")
    print(f"  [1] {dup['question_1']}")
    print(f"  [2] {dup['question_2']}")

# Output:
# 🔴 DUPLICADO DETECTADO (similitud: 0.912)
#   [1] ¿Qué artículo regula el homicidio?
#   [2] ¿Cuál es el artículo sobre homicidio en el CP?
#
# 🔴 DUPLICADO DETECTADO (similitud: 0.878)
#   [1] ¿Cuántos años de prisión por asesinato?
#   [2] ¿Qué pena tiene el asesinato?
```

### Uso en Nuestro Proyecto (Agente D)

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def agent_d_persistence(state: GraphState) -> GraphState:
    """Agente D: Deduplicación y persistencia"""

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    questions = state["validated_questions"]

    # Generar embeddings (pregunta + respuesta correcta)
    texts = [q.get_text_for_embedding() for q in questions]
    embeddings = model.encode(texts, batch_size=32)

    # Detectar duplicados
    unique_questions = []
    duplicate_count = 0

    for i, question in enumerate(questions):
        is_duplicate = False

        # Comparar con preguntas únicas ya procesadas
        for j, unique_q in enumerate(unique_questions):
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[j].reshape(1, -1)
            )[0][0]

            if similarity >= 0.85:  # Threshold
                question.is_duplicate = True
                question.duplicate_of = unique_q.id
                duplicate_count += 1
                is_duplicate = True
                break

        if not is_duplicate:
            unique_questions.append(question)

    # Persistir en SQLite (solo únicas)
    # TODO: Implementar SQLite insert

    # Estadísticas
    state["dedup_stats"] = {
        "total": len(questions),
        "unique": len(unique_questions),
        "duplicates": duplicate_count
    }

    return state
```

---

## 6. Unstructured - Procesamiento de PDFs

### ¿Qué es?
Librería avanzada para extraer texto de documentos (PDFs, Word, HTML, etc.) con análisis de estructura.

### ¿Por qué lo usamos?
- Extracción de PDFs mejor que PyPDF
- Detecta estructura: títulos, párrafos, tablas
- Chunking semántico automático
- OCR integrado para PDFs escaneados

### Ejemplo Simple: Extraer de PDF

```python
from unstructured.partition.pdf import partition_pdf

# Extraer elementos del PDF
elements = partition_pdf(
    filename="input_docs/tema_1.pdf",
    strategy="fast"  # fast, hi_res, ocr_only
)

# Ver elementos
for element in elements[:5]:
    print(f"Tipo: {element.category}")
    print(f"Texto: {element.text[:100]}...")
    print(f"Metadata: {element.metadata}")
    print("---")

# Output:
# Tipo: Title
# Texto: TEMA 1: DERECHO PENAL
# Metadata: {'page_number': 1, 'coordinates': {...}}
# ---
# Tipo: NarrativeText
# Texto: El Código Penal español regula los delitos y las penas...
# Metadata: {'page_number': 1}
# ---
```

### Ejemplo: Chunking Semántico por Títulos

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# Extraer con análisis de estructura
elements = partition_pdf(
    filename="input_docs/tema_1.pdf",
    strategy="hi_res",  # High resolution
    languages=["spa"]   # Español
)

# Dividir por títulos/secciones
chunks = chunk_by_title(
    elements,
    max_characters=1000,         # Tamaño máximo
    combine_text_under_n_chars=200  # Combinar textos pequeños
)

# Ver chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n=== CHUNK {i} ===")
    print(f"Tipo: {chunk.category}")
    print(f"Texto: {chunk.text[:200]}...")
    print(f"Página: {chunk.metadata.get('page_number')}")

# Output:
# === CHUNK 0 ===
# Tipo: CompositeElement
# Texto: TEMA 1: DERECHO PENAL. El Código Penal español regula...
# Página: 1
#
# === CHUNK 1 ===
# Tipo: CompositeElement
# Texto: Artículo 138: Homicidio. El que matare a otro...
# Página: 2
```

### Ejemplo: Extraer Tablas

```python
from unstructured.partition.pdf import partition_pdf

# Extraer con detección de tablas
elements = partition_pdf(
    filename="input_docs/tema_1.pdf",
    strategy="hi_res",
    infer_table_structure=True  # Detectar tablas
)

# Filtrar solo tablas
tables = [el for el in elements if el.category == "Table"]

for i, table in enumerate(tables):
    print(f"\n=== TABLA {i} ===")
    print(table.text)
    print(f"Página: {table.metadata.get('page_number')}")

# Output:
# === TABLA 0 ===
# Delito | Artículo | Pena
# Homicidio | 138 | 10-15 años
# Asesinato | 139 | 15-25 años
# Página: 3
```

### Uso en Nuestro Proyecto (Agente A Avanzado)

```python
from pathlib import Path
from agents.agent_z_rewriter import RewriterAgent

def agent_z_coherent_chunking_advanced(state: GraphState) -> GraphState:
    """Agente Z con chunking coherente y análisis LLM opcional"""

    all_chunks = []

    for doc_path in state["input_docs"]:
        # Crear agente Z con análisis LLM
        agent_z = RewriterAgent(use_llm=True)  # Usa LLM para análisis de coherencia

        # Generar chunks coherentes
        chunks = agent_z.create_coherent_chunks(
            Path(doc_path),
            topic=state["metadata"].get("topic", 1)
        )

        # Filtrar solo chunks coherentes
        raw_chunks = chunk_by_title(
            elements,
            max_characters=1000,
            combine_text_under_n_chars=200
        )

        # Convertir a modelo Chunk
        for i, chunk in enumerate(raw_chunks):
            all_chunks.append(Chunk(
                chunk_id=f"{doc_path}__chunk_{i}",
                content=chunk.text,
                source_document=doc_path,
                page=chunk.metadata.get("page_number"),
                token_count=len(chunk.text) // 4,
                metadata={
                    "type": chunk.category,  # Title, Table, NarrativeText
                    "is_table": chunk.category == "Table"
                }
            ))

    state["chunks"] = all_chunks
    state["current_step"] = "chunking_complete"
    return state
```

---

## 7. Guardrails AI - Validación de Outputs

### ¿Qué es?
Framework para validar y corregir outputs de LLMs en tiempo real.

### ¿Por qué lo usamos?
- Validar que las preguntas generadas tengan el formato correcto
- Reask automático si el LLM genera algo inválido
- Constraints: longitud, valores permitidos, formato JSON

### Ejemplo Simple: Validar Longitud

```python
import guardrails as gd

# Definir schema con validaciones
rail = """
<rail version="0.1">
<output>
    <string name="summary"
            validators="valid-length: 10 50"
            on-fail-valid-length="reask"
            description="Resumen corto"/>
</output>

<prompt>
Genera un resumen del siguiente texto:
{{text}}
</prompt>
</rail>
"""

# Crear guard
guard = gd.Guard.from_rail_string(rail)

# Ejecutar con LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")

# Validar output
result = guard(
    llm.invoke,
    prompt_params={"text": "El Código Penal regula delitos y penas en España."},
    num_reasks=2  # Máximo 2 reintentos
)

if result.validation_passed:
    print("✅ Validado:", result.validated_output)
else:
    print("❌ Error:", result.error)

# Output:
# ✅ Validado: {'summary': 'Código Penal español: delitos y penas'}
```

### Ejemplo: Validar Opciones Permitidas

```python
import guardrails as gd
from pydantic import BaseModel

# Definir schema Pydantic
class QuestionSchema(BaseModel):
    question: str
    answer1: str
    answer2: str
    answer3: str
    answer4: str
    correct: int  # Solo 1, 2, 3, o 4

# Crear guard desde Pydantic
guard = gd.Guard.from_pydantic(QuestionSchema)

# Output del LLM (JSON)
llm_output = """
{
    "question": "¿Qué es el homicidio?",
    "answer1": "Matar a otra persona",
    "answer2": "Robar un banco",
    "answer3": "Conducir ebrio",
    "answer4": "Falsificar documentos",
    "correct": 1
}
"""

# Validar
result = guard.parse(llm_output)

if result.validation_passed:
    print("✅ Pregunta válida:")
    print(result.validated_output)
else:
    print("❌ Errores:", result.error)

# Output:
# ✅ Pregunta válida:
# {'question': '¿Qué es el homicidio?', 'answer1': 'Matar a otra persona', ...}
```

### Ejemplo: Reask Automático

```python
import guardrails as gd
from langchain_openai import ChatOpenAI

# Rail con validaciones estrictas
rail = """
<rail version="0.1">
<output>
    <object name="question">
        <string name="text"
                validators="valid-length: 20 200"
                on-fail-valid-length="reask"/>

        <integer name="solution"
                 validators="valid-choices: 1 2 3 4"
                 on-fail-valid-choices="reask"/>
    </object>
</output>

<prompt>
Genera una pregunta tipo test sobre: {{topic}}
</prompt>
</rail>
"""

guard = gd.Guard.from_rail_string(rail)
llm = ChatOpenAI(model="gpt-4")

# Si el LLM genera algo inválido, automáticamente hace reask
result = guard(
    llm.invoke,
    prompt_params={"topic": "Derecho Penal"},
    num_reasks=3  # Hasta 3 reintentos
)

print(result.validated_output)
```

### Uso en Nuestro Proyecto (Agente B)

```python
import guardrails as gd
from pydantic import BaseModel, Field

# Schema de validación
class QuestionOutput(BaseModel):
    question: str = Field(..., min_length=20, max_length=500)
    answer1: str = Field(..., min_length=5, max_length=200)
    answer2: str = Field(..., min_length=5, max_length=200)
    answer3: str = Field(..., min_length=5, max_length=200)
    answer4: str = Field(..., min_length=5, max_length=200)
    solution: int = Field(..., ge=1, le=4)
    tip: str = Field(None, max_length=300)

# Crear guard
guard = gd.Guard.from_pydantic(QuestionOutput)

def agent_b_generation(state: GraphState) -> GraphState:
    llm = ChatOpenAI(model="gpt-4")

    for chunk in state["chunks"]:
        # Generar con LLM
        prompt = f"Genera una pregunta tipo test basada en: {chunk.content}"
        response = llm.invoke(prompt)

        # Validar con Guardrails
        result = guard.parse(
            response.content,
            llm_api=llm.invoke,
            num_reasks=2
        )

        if result.validation_passed:
            # Convertir a Question model
            question = Question(
                academy=state["metadata"]["academy"],
                topic=state["metadata"]["topic"],
                **result.validated_output,
                source_chunk_id=chunk.chunk_id
            )
            # Guardar...
        else:
            # Loggear error
            state["errors"].append({
                "node": "generation",
                "chunk_id": chunk.chunk_id,
                "error": str(result.error)
            })

    return state
```

---

## 8. Pydantic - Validación de Datos

### ¿Qué es?
Librería para validación de datos usando Python type hints. Define modelos con validación automática.

### ¿Por qué lo usamos?
- Validar datos de entrada/salida
- Type safety en todo el código
- Conversión automática de tipos
- Validaciones custom

### Ejemplo Simple: Modelo Básico

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str
    age: int = Field(..., ge=0, le=150)  # Entre 0 y 150
    email: str

# Crear instancia
person = Person(name="Juan", age=30, email="juan@example.com")
print(person)
# Output: Person(name='Juan', age=30, email='juan@example.com')

# Convertir a dict
print(person.model_dump())
# Output: {'name': 'Juan', 'age': 30, 'email': 'juan@example.com'}

# Error de validación
try:
    invalid = Person(name="Ana", age=200, email="ana@example.com")  # age > 150
except Exception as e:
    print("❌ Error:", e)
# Output: ❌ Error: 1 validation error for Person
#         age: ensure this value is less than or equal to 150
```

### Ejemplo: Validadores Custom

```python
from pydantic import BaseModel, validator

class Question(BaseModel):
    question: str
    answer1: str
    answer2: str
    answer3: str
    answer4: str | None = None
    solution: int  # 1, 2, 3, o 4

    @validator('solution')
    def validate_solution(cls, v, values):
        """Validar que solution sea coherente con las opciones"""
        if v < 1 or v > 4:
            raise ValueError('solution debe ser 1, 2, 3 o 4')

        # Si no hay answer4, solution no puede ser 4
        if 'answer4' in values and values['answer4'] is None and v == 4:
            raise ValueError('solution no puede ser 4 si answer4 es None')

        return v

    @validator('question')
    def validate_question_length(cls, v):
        if len(v) < 10:
            raise ValueError('La pregunta debe tener al menos 10 caracteres')
        return v

# Válido
q1 = Question(
    question="¿Qué es el homicidio?",
    answer1="A",
    answer2="B",
    answer3="C",
    solution=1
)
print("✅ Válido:", q1)

# Inválido: solution=4 pero no hay answer4
try:
    q2 = Question(
        question="¿Qué es el homicidio?",
        answer1="A",
        answer2="B",
        answer3="C",
        solution=4  # ERROR
    )
except Exception as e:
    print("❌ Error:", e)
# Output: ❌ Error: solution no puede ser 4 si answer4 es None
```

### Ejemplo: Settings desde Environment Variables

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Settings cargadas desde .env"""

    openai_api_key: str
    openai_model: str = "gpt-4"
    chunk_size: int = 1000
    max_retries: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = False

# Cachear para no recargar
@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Usar
settings = get_settings()
print(f"API Key: {settings.openai_api_key[:10]}...")
print(f"Model: {settings.openai_model}")

# Output:
# API Key: sk-proj-ab...
# Model: gpt-4
```

### Uso en Nuestro Proyecto

Ya implementado en:
- `models/question.py`: Modelo Question con validaciones
- `models/chunk.py`: Modelo Chunk
- `models/quality_metrics.py`: Modelo QualityMetrics
- `config/settings.py`: Settings globales desde .env
- `graph/state.py`: GraphState TypedDict

---

## Resumen Rápido

| Librería | Cuándo la usamos | Agente |
|----------|------------------|--------|
| **LangGraph** | Orquestar el pipeline, retry logic | Todos |
| **LangChain** | Cargar PDFs, dividir chunks, chat con LLM | A, B |
| **Ragas** | Evaluar calidad de preguntas | C |
| **DSPy** | Optimizar prompts automáticamente | B |
| **Sentence Transformers** | Deduplicación semántica | D |
| **Unstructured** | Extracción avanzada de PDFs | A |
| **Guardrails** | Validar outputs del LLM | B |
| **Pydantic** | Validar datos en todo el código | Todos |

---

## Próximos Pasos

1. **Implementar Agente A** con LangChain o Unstructured
2. **Implementar Agente B** con DSPy + Guardrails
3. **Implementar Agente C** con Ragas
4. **Implementar Agente D** con Sentence Transformers
5. **Orquestar con LangGraph**

¿Quieres que empecemos implementando algún agente específico?
