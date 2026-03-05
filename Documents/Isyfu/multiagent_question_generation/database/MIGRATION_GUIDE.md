# Guía de Migración V1 → V2

## 🎯 Objetivo

Migrar de la estructura **V1** (1 tabla con answer1-4) a la estructura **V2** (tabla questions + tabla question_options separadas), compatible con el sistema real de Supabase.

---

## 📊 Comparación de Estructuras

### V1 (Antigua)
```
questions
├── id
├── question
├── answer1      ← Todas las opciones en la misma tabla
├── answer2
├── answer3
├── answer4
├── solution (1-4)
├── tip
├── topic
└── ... metadata
```

### V2 (Nueva - Compatible con Supabase)
```
questions                           question_options
├── id                              ├── id
├── question                        ├── question_id (FK)
├── tip                             ├── answer
├── article                         ├── is_correct
├── topic                           ├── option_order (1-4)
├── published                       ├── created_at
├── shuffled                        └── updated_at
├── multimedia (images, audio)
├── estadísticas (num_answered, etc.)
└── ... metadata ampliada
```

---

## 🚀 Pasos de Migración

### Opción A: Migración Automática con SQL (Recomendada)

1. **Backup de la base de datos actual:**
   ```bash
   cp database/questions.db database/questions_v1_backup.db
   ```

2. **Ejecutar script de migración:**
   ```bash
   sqlite3 database/questions.db < database/migrate_v1_to_v2.sql
   ```

3. **Verificar migración:**
   El script incluye verificaciones automáticas que mostrarán:
   - Total de preguntas migradas
   - Total de opciones migradas
   - Preguntas sin suficientes opciones (debería ser 0)
   - Preguntas sin respuesta correcta (debería ser 0)
   - Opciones huérfanas (debería ser 0)

4. **Probar con consultas:**
   ```sql
   -- Vista de compatibilidad (formato legacy)
   SELECT * FROM questions_with_options LIMIT 5;

   -- Vista completa con opciones como JSON
   SELECT * FROM questions_full LIMIT 5;

   -- Consulta directa con joins
   SELECT q.id, q.question, qo.option_order, qo.answer, qo.is_correct
   FROM questions q
   JOIN question_options qo ON qo.question_id = q.id
   WHERE q.id = 1
   ORDER BY qo.option_order;
   ```

### Opción B: Migración con Python

1. **Crear script de migración:**
   ```python
   # Ver: database/migrate_db.py
   python database/migrate_db.py
   ```

2. **El script Python:**
   - Lee todas las preguntas de V1
   - Las convierte al formato V2
   - Inserta en las nuevas tablas
   - Verifica integridad
   - Genera reporte

---

## 📝 Cambios en el Código Python

### 1. Importar nuevos modelos

**Antes:**
```python
from models.question import Question
```

**Después:**
```python
from models.question_v2 import Question
from models.question_option import QuestionOption
```

### 2. Crear preguntas con opciones

**Antes:**
```python
question = Question(
    question="¿Quién...?",
    answer1="Opción A",
    answer2="Opción B",
    answer3="Opción C",
    answer4="Opción D",
    solution=2,
    topic=1,
    academy=1
)
```

**Después:**
```python
# Opción 1: Crear con opciones directamente
question = Question(
    question="¿Quién...?",
    topic=1,
    academy_id=1,
    options=[
        QuestionOption(question_id=0, answer="Opción A", is_correct=False, option_order=1),
        QuestionOption(question_id=0, answer="Opción B", is_correct=True, option_order=2),
        QuestionOption(question_id=0, answer="Opción C", is_correct=False, option_order=3),
        QuestionOption(question_id=0, answer="Opción D", is_correct=False, option_order=4),
    ]
)

# Opción 2: Convertir desde formato legacy
legacy_data = {
    "question": "¿Quién...?",
    "answer1": "Opción A",
    "answer2": "Opción B",
    "answer3": "Opción C",
    "answer4": "Opción D",
    "solution": 2,
    "topic": 1,
    "academy": 1
}
question = Question.from_legacy_dict(legacy_data)
```

### 3. Acceder a opciones

**Antes:**
```python
correct_answer = question.get_correct_answer()  # Devuelve answer2
all_answers = question.get_all_answers()  # ['answer1', 'answer2', 'answer3', 'answer4']
```

**Después:**
```python
# Obtener respuesta correcta
correct_answer = question.get_correct_answer_text()  # "Opción B"
correct_order = question.get_correct_answer_order()  # 2

# Obtener todas las opciones
all_answers = question.get_all_answers()  # ['Opción A', 'Opción B', 'Opción C', 'Opción D']

# Iterar sobre opciones
for option in question.options:
    print(f"{option.option_order}. {option.answer} {'✓' if option.is_correct else ''}")
```

### 4. Guardar en base de datos

**Antes (SQLite directo):**
```python
conn = sqlite3.connect('database/questions.db')
cursor = conn.cursor()

cursor.execute("""
    INSERT INTO questions (question, answer1, answer2, answer3, answer4, solution, topic, academy)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (q.question, q.answer1, q.answer2, q.answer3, q.answer4, q.solution, q.topic, q.academy))

question_id = cursor.lastrowid
conn.commit()
```

**Después (SQLite con opciones):**
```python
conn = sqlite3.connect('database/questions.db')
cursor = conn.cursor()

# 1. Insertar pregunta
cursor.execute("""
    INSERT INTO questions (question, tip, topic, academy_id, published, ...)
    VALUES (?, ?, ?, ?, ?, ...)
""", (q.question, q.tip, q.topic, q.academy_id, q.published, ...))

question_id = cursor.lastrowid

# 2. Insertar opciones
for option in q.options:
    cursor.execute("""
        INSERT INTO question_options (question_id, answer, is_correct, option_order)
        VALUES (?, ?, ?, ?)
    """, (question_id, option.answer, option.is_correct, option.option_order))

conn.commit()
```

---

## 🔧 Actualizar Agente B

El Agente B debe generar preguntas en el nuevo formato. Los cambios necesarios:

### 1. Actualizar la herramienta `generate_question_with_advanced_prompt`

**Antes:** El prompt devuelve JSON con answer1-4, solution

**Después:** El prompt debe devolver JSON con un array de options

**Ejemplo de salida esperada:**
```json
{
  "question": "¿Cuál es el plazo para...",
  "tip": "La respuesta correcta es...",
  "article": "Artículo 45.1 de...",
  "options": [
    {"answer": "3 meses", "is_correct": false, "option_order": 1},
    {"answer": "6 meses", "is_correct": true, "option_order": 2},
    {"answer": "1 año", "is_correct": false, "option_order": 3},
    {"answer": "5 años", "is_correct": false, "option_order": 4}
  ],
  "distractor_techniques": ["juego_cómputo_tiempo", "alteración_números"]
}
```

### 2. Actualizar el parsing en el Agente B

```python
# En agent_b_generator.py, función generate_question_with_advanced_prompt

question_data = json.loads(content)

# Convertir options del JSON a QuestionOption objects
if 'options' in question_data:
    options = [
        QuestionOption(
            question_id=0,  # Se asignará al guardar
            answer=opt['answer'],
            is_correct=opt['is_correct'],
            option_order=opt['option_order']
        )
        for opt in question_data['options']
    ]
    question_data['options'] = options

# Si viene en formato legacy (answer1-4, solution), convertir
elif 'answer1' in question_data:
    question = Question.from_legacy_dict(question_data)
    return question

return Question(**question_data)
```

---

## ✅ Verificación Post-Migración

### 1. Integridad de datos
```sql
-- Todas las preguntas deben tener al menos 3 opciones
SELECT q.id, q.question, COUNT(qo.id) as num_options
FROM questions q
LEFT JOIN question_options qo ON qo.question_id = q.id
GROUP BY q.id
HAVING COUNT(qo.id) < 3;
-- Debería devolver 0 filas

-- Todas las preguntas deben tener exactamente 1 respuesta correcta
SELECT q.id, q.question, COUNT(CASE WHEN qo.is_correct THEN 1 END) as correct_count
FROM questions q
LEFT JOIN question_options qo ON qo.question_id = q.id
GROUP BY q.id
HAVING COUNT(CASE WHEN qo.is_correct THEN 1 END) != 1;
-- Debería devolver 0 filas
```

### 2. Compatibilidad con código legacy
```sql
-- La vista questions_with_options debe funcionar igual que la tabla V1
SELECT id, question, answer1, answer2, answer3, answer4, solution
FROM questions_with_options
WHERE id = 1;
```

### 3. Test en Python
```python
# test_migration.py
from models.question_v2 import Question
from models.question_option import QuestionOption
import sqlite3

# Conectar a DB migrada
conn = sqlite3.connect('database/questions.db')
cursor = conn.cursor()

# Leer una pregunta
cursor.execute("""
    SELECT q.*,
           json_group_array(
               json_object(
                   'id', qo.id,
                   'answer', qo.answer,
                   'is_correct', qo.is_correct,
                   'option_order', qo.option_order
               )
           ) as options_json
    FROM questions q
    LEFT JOIN question_options qo ON qo.question_id = q.id
    WHERE q.id = 1
    GROUP BY q.id
""")

row = cursor.fetchone()
print(f"Pregunta migrada: {row}")

# Verificar que se puede parsear
# (Implementar función de hydrate desde row a Question object)
```

---

## 🐛 Troubleshooting

### Problema: "No such table: questions"
**Solución:** Ejecutar `init_schema_v2.sql` antes de `migrate_v1_to_v2.sql`

### Problema: "UNIQUE constraint failed: question_options.question_id, option_order"
**Solución:** Hay opciones duplicadas con el mismo question_id y option_order. Revisar datos en V1.

### Problema: Algunas preguntas no tienen respuesta correcta
**Solución:** Verificar en V1 que `solution` esté entre 1-4 y que la opción correspondiente exista.

### Problema: Código legacy no funciona con nuevos modelos
**Solución:** Usar los métodos de compatibilidad:
- `Question.from_legacy_dict()` para cargar
- `question.to_legacy_dict()` para exportar

---

## 📚 Referencias

- **Schema V1:** `database/init_schema.sql`
- **Schema V2:** `database/init_schema_v2.sql`
- **Script de migración SQL:** `database/migrate_v1_to_v2.sql`
- **Modelo Question V2:** `models/question_v2.py`
- **Modelo QuestionOption:** `models/question_option.py`
- **Estructura Supabase:** `/Users/pablofernandezlucas/Documents/Isyfu/opn_test_template/policia_nacional/supabase/migrations/00003_tables_content.sql`

---

## 🎉 Beneficios de V2

1. ✅ **Compatible con Supabase:** Estructura idéntica al sistema de producción
2. ✅ **Flexible:** Permite 3, 4, 5 o N opciones de respuesta
3. ✅ **Escalable:** Más fácil agregar metadata por opción (ej: explicación por opción incorrecta)
4. ✅ **Normalizada:** Elimina redundancia (answer1-4)
5. ✅ **Mantenible:** Cambios en opciones no afectan la tabla principal
6. ✅ **Extensible:** Fácil agregar campos nuevos sin alterar estructura existente

---

**Fecha de creación:** 2026-01-19
**Versión:** 2.0
**Estado:** ✅ Listo para migración