-- =========================================================================
-- SCHEMA V2 - Compatible con estructura Supabase
-- =========================================================================
-- Estructura de base de datos adaptada del proyecto real en Supabase
-- Tablas: questions (con metadata), question_options (opciones separadas)
-- =========================================================================

-- -------------------------------------------------------------------------
-- 1. TABLA QUESTIONS
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Contenido de la pregunta
    question TEXT NOT NULL,
    tip TEXT,  -- Explicación/feedback detallado
    article TEXT,  -- Artículo o referencia legal

    -- Clasificación y organización
    topic INTEGER NOT NULL,  -- FK a topics (no creamos tabla topics en este proyecto)
    academy_id INTEGER NOT NULL DEFAULT 1,
    classification_category_id INTEGER,  -- Para clasificación cruzada
    classification_topic_id INTEGER,  -- Para clasificación cruzada

    -- Orden y visualización
    order_num INTEGER DEFAULT 0 NOT NULL,
    published BOOLEAN DEFAULT TRUE NOT NULL,
    shuffled BOOLEAN,  -- Si las opciones deben mezclarse

    -- URLs de multimedia
    question_image_url TEXT DEFAULT '',
    retro_image_url TEXT DEFAULT '',
    retro_audio_enable BOOLEAN DEFAULT FALSE NOT NULL,
    retro_audio_text TEXT DEFAULT '',
    retro_audio_url TEXT DEFAULT '',

    -- Estadísticas de uso (se actualizarán cuando se implemente el sistema de tests)
    num_answered INTEGER DEFAULT 0 NOT NULL,
    num_fails INTEGER DEFAULT 0 NOT NULL,
    num_empty INTEGER DEFAULT 0 NOT NULL,
    -- difficult_rate se puede calcular: num_answered / (num_answered + num_fails + num_empty)

    -- Metadata de creación
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_by TEXT,  -- UUID del usuario creador
    created_by_cms_user_id INTEGER,  -- ID del usuario CMS

    -- Challenges/Disputes
    challenge_by_tutor BOOLEAN DEFAULT FALSE NOT NULL,
    challenge_reason TEXT,

    -- ==================== METADATA DE GENERACIÓN LLM ====================
    -- Estos campos son específicos del sistema de generación automática
    llm_model TEXT,
    by_llm BOOLEAN DEFAULT 1,
    question_prompt TEXT,  -- Prompt usado para generar la pregunta
    generation_method TEXT,  -- 'advanced_prompt', 'simple', etc.

    -- ==================== QUALITY TRACKING ====================
    faithfulness_score REAL CHECK(faithfulness_score IS NULL OR (faithfulness_score >= 0 AND faithfulness_score <= 1)),
    relevancy_score REAL CHECK(relevancy_score IS NULL OR (relevancy_score >= 0 AND relevancy_score <= 1)),
    source_chunk_id TEXT,  -- ID del chunk del documento fuente
    source_document TEXT,  -- Nombre del documento fuente
    source_page INTEGER,  -- Página del documento

    -- ==================== DEDUPLICATION & RETRY ====================
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of INTEGER REFERENCES questions(id) ON DELETE SET NULL,
    retry_count INTEGER DEFAULT 0,
    needs_manual_review BOOLEAN DEFAULT FALSE,

    -- ==================== TÉCNICAS DE DISTRACCIÓN APLICADAS ====================
    distractor_techniques TEXT  -- JSON array con técnicas usadas: ["cambio_requisitos", "juego_tiempo"]
);

-- -------------------------------------------------------------------------
-- 2. TABLA QUESTION_OPTIONS
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS question_options (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL,
    answer TEXT NOT NULL,
    is_correct BOOLEAN DEFAULT FALSE NOT NULL,
    option_order INTEGER NOT NULL,  -- 1, 2, 3, 4 (orden original)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,

    -- Constraint para asegurar que cada pregunta tenga opciones únicas
    UNIQUE(question_id, option_order),

    -- Foreign key
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);

-- -------------------------------------------------------------------------
-- 3. TABLA BATCHES (metadata de generación)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_name TEXT UNIQUE NOT NULL,
    topic INTEGER NOT NULL,
    academy_id INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_questions INTEGER DEFAULT 0,
    unique_questions INTEGER DEFAULT 0,
    duplicates INTEGER DEFAULT 0,
    manual_review_count INTEGER DEFAULT 0,
    avg_faithfulness REAL,
    avg_relevancy REAL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),

    -- Metadata adicional
    source_documents TEXT,  -- JSON array de documentos procesados
    generation_config TEXT  -- JSON con configuración de generación
);

-- =========================================================================
-- ÍNDICES PARA OPTIMIZACIÓN
-- =========================================================================

-- Índices en questions
CREATE INDEX IF NOT EXISTS idx_questions_topic ON questions(topic);
CREATE INDEX IF NOT EXISTS idx_questions_academy ON questions(academy_id);
CREATE INDEX IF NOT EXISTS idx_questions_published ON questions(published);
CREATE INDEX IF NOT EXISTS idx_questions_duplicate ON questions(is_duplicate);
CREATE INDEX IF NOT EXISTS idx_questions_manual_review ON questions(needs_manual_review);
CREATE INDEX IF NOT EXISTS idx_questions_source_chunk ON questions(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_questions_created_at ON questions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_questions_classification_category ON questions(classification_category_id);
CREATE INDEX IF NOT EXISTS idx_questions_classification_topic ON questions(classification_topic_id);
CREATE INDEX IF NOT EXISTS idx_questions_cms_creator ON questions(created_by_cms_user_id);

-- Índices en question_options
CREATE INDEX IF NOT EXISTS idx_options_question_id ON question_options(question_id);
CREATE INDEX IF NOT EXISTS idx_options_is_correct ON question_options(is_correct);
CREATE INDEX IF NOT EXISTS idx_options_order ON question_options(question_id, option_order);

-- Índices en batches
CREATE INDEX IF NOT EXISTS idx_batch_status ON batches(status);
CREATE INDEX IF NOT EXISTS idx_batch_created ON batches(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_topic ON batches(topic);
CREATE INDEX IF NOT EXISTS idx_batch_academy ON batches(academy_id);

-- =========================================================================
-- TRIGGERS PARA UPDATED_AT
-- =========================================================================

-- Trigger para questions.updated_at
CREATE TRIGGER IF NOT EXISTS update_questions_timestamp
AFTER UPDATE ON questions
FOR EACH ROW
BEGIN
    UPDATE questions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger para question_options.updated_at
CREATE TRIGGER IF NOT EXISTS update_question_options_timestamp
AFTER UPDATE ON question_options
FOR EACH ROW
BEGIN
    UPDATE question_options SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- =========================================================================
-- VISTAS ÚTILES
-- =========================================================================

-- Vista para obtener preguntas con sus opciones agregadas (compatibilidad con código legacy)
CREATE VIEW IF NOT EXISTS questions_with_options AS
SELECT
    q.*,
    (SELECT answer FROM question_options WHERE question_id = q.id AND option_order = 1) AS answer1,
    (SELECT answer FROM question_options WHERE question_id = q.id AND option_order = 2) AS answer2,
    (SELECT answer FROM question_options WHERE question_id = q.id AND option_order = 3) AS answer3,
    (SELECT answer FROM question_options WHERE question_id = q.id AND option_order = 4) AS answer4,
    (SELECT option_order FROM question_options WHERE question_id = q.id AND is_correct = 1 LIMIT 1) AS solution
FROM questions q;

-- Vista para obtener preguntas completas con todas sus opciones como JSON
CREATE VIEW IF NOT EXISTS questions_full AS
SELECT
    q.*,
    (
        SELECT json_group_array(
            json_object(
                'id', qo.id,
                'answer', qo.answer,
                'is_correct', qo.is_correct,
                'option_order', qo.option_order
            )
        )
        FROM question_options qo
        WHERE qo.question_id = q.id
        ORDER BY qo.option_order
    ) AS options_json,
    (SELECT option_order FROM question_options WHERE question_id = q.id AND is_correct = 1 LIMIT 1) AS correct_answer_order
FROM questions q;

-- Vista para estadísticas de calidad por batch
CREATE VIEW IF NOT EXISTS batch_quality_stats AS
SELECT
    b.id,
    b.batch_name,
    b.topic,
    b.academy_id,
    b.total_questions,
    b.unique_questions,
    b.duplicates,
    b.manual_review_count,
    b.avg_faithfulness,
    b.avg_relevancy,
    b.status,
    b.created_at,
    -- Contar preguntas reales asociadas al batch
    COUNT(DISTINCT q.id) as actual_questions_count,
    AVG(q.faithfulness_score) as computed_avg_faithfulness,
    AVG(q.relevancy_score) as computed_avg_relevancy
FROM batches b
LEFT JOIN questions q ON q.source_document LIKE '%' || b.batch_name || '%'
GROUP BY b.id;

-- =========================================================================
-- COMENTARIOS SOBRE LA ESTRUCTURA
-- =========================================================================

-- NOTAS IMPORTANTES:
-- 1. Esta estructura es compatible con el sistema real de Supabase
-- 2. La tabla questions almacena la pregunta y metadata
-- 3. La tabla question_options almacena las opciones de respuesta (1 a N opciones)
-- 4. Se mantiene compatibilidad con el sistema de generación LLM mediante campos adicionales
-- 5. Las vistas permiten compatibilidad con código legacy que espera answer1-4
-- 6. Los triggers mantienen updated_at actualizado automáticamente

-- DIFERENCIAS CON SUPABASE:
-- 1. SQLite usa INTEGER AUTOINCREMENT en lugar de bigint GENERATED BY DEFAULT AS IDENTITY
-- 2. No tenemos soporte para vector embeddings (requiere pgvector)
-- 3. No tenemos tsvector para full text search (se puede implementar con FTS5)
-- 4. No tenemos stored generated columns (difficult_rate debe calcularse en queries)
-- 5. Los tipos de datos son adaptados: timestamp -> TIMESTAMP, text -> TEXT, bigint -> INTEGER

-- MIGRACIÓN DESDE SCHEMA V1:
-- Ver script: database/migrate_v1_to_v2.sql
