-- =========================================================================
-- MIGRACIÓN DE SCHEMA V1 A V2
-- =========================================================================
-- Este script migra datos desde el esquema antiguo (answer1-4 en una tabla)
-- al nuevo esquema (questions + question_options separadas)
-- =========================================================================

-- -------------------------------------------------------------------------
-- PASO 1: BACKUP DE LA TABLA ANTIGUA
-- -------------------------------------------------------------------------

-- Renombrar tabla antigua a _backup
ALTER TABLE questions RENAME TO questions_v1_backup;

-- -------------------------------------------------------------------------
-- PASO 2: CREAR NUEVAS TABLAS
-- -------------------------------------------------------------------------

-- Ejecutar init_schema_v2.sql o crear manualmente las tablas
-- (Este script asume que ya se han creado las tablas nuevas)

-- -------------------------------------------------------------------------
-- PASO 3: MIGRAR DATOS DE QUESTIONS
-- -------------------------------------------------------------------------

INSERT INTO questions (
    -- ID (se auto-generará si es AUTOINCREMENT, pero intentamos preservar el original)
    id,

    -- Contenido
    question,
    tip,
    article,  -- NULL en V1, pero lo incluimos

    -- Clasificación
    topic,
    academy_id,  -- academy en V1
    classification_category_id,
    classification_topic_id,

    -- Orden
    order_num,
    published,
    shuffled,

    -- Multimedia (todos NULL en V1)
    question_image_url,
    retro_image_url,
    retro_audio_enable,
    retro_audio_text,
    retro_audio_url,

    -- Estadísticas (inicializar en 0)
    num_answered,
    num_fails,
    num_empty,

    -- Metadata
    created_at,
    updated_at,  -- Igual que created_at en V1
    created_by,
    created_by_cms_user_id,

    -- Challenges
    challenge_by_tutor,
    challenge_reason,

    -- LLM
    llm_model,
    by_llm,
    question_prompt,
    generation_method,  -- NULL en V1

    -- Quality
    faithfulness_score,
    relevancy_score,
    source_chunk_id,
    source_document,
    source_page,  -- NULL en V1

    -- Deduplication
    is_duplicate,
    duplicate_of,
    retry_count,
    needs_manual_review,

    -- Distractor techniques
    distractor_techniques  -- NULL en V1
)
SELECT
    -- ID preservado
    id,

    -- Contenido
    question,
    tip,
    NULL as article,

    -- Clasificación
    topic,
    academy as academy_id,  -- Mapear academy -> academy_id
    NULL as classification_category_id,
    NULL as classification_topic_id,

    -- Orden
    COALESCE(order_num, 0) as order_num,
    TRUE as published,  -- Por defecto publicado
    NULL as shuffled,

    -- Multimedia
    '' as question_image_url,
    '' as retro_image_url,
    FALSE as retro_audio_enable,
    '' as retro_audio_text,
    '' as retro_audio_url,

    -- Estadísticas
    0 as num_answered,
    0 as num_fails,
    0 as num_empty,

    -- Metadata
    created_at,
    created_at as updated_at,  -- Usar created_at también como updated_at
    NULL as created_by,
    NULL as created_by_cms_user_id,

    -- Challenges
    FALSE as challenge_by_tutor,
    NULL as challenge_reason,

    -- LLM
    llm_model as llm_model,
    COALESCE(by_llm, TRUE) as by_llm,
    question_prompt,
    NULL as generation_method,

    -- Quality
    faithfulness_score,
    relevancy_score,
    source_chunk_id,
    source_document,
    NULL as source_page,

    -- Deduplication
    COALESCE(is_duplicate, FALSE) as is_duplicate,
    duplicate_of,
    COALESCE(retry_count, 0) as retry_count,
    COALESCE(needs_manual_review, FALSE) as needs_manual_review,

    -- Distractor techniques
    NULL as distractor_techniques

FROM questions_v1_backup;

-- -------------------------------------------------------------------------
-- PASO 4: MIGRAR OPCIONES DE RESPUESTA
-- -------------------------------------------------------------------------

-- Para cada pregunta en V1, crear 3 o 4 filas en question_options
-- dependiendo de si answer4 existe o no

-- Opción 1 (siempre existe)
INSERT INTO question_options (question_id, answer, is_correct, option_order, created_at, updated_at)
SELECT
    id as question_id,
    answer1 as answer,
    (solution = 1) as is_correct,
    1 as option_order,
    created_at,
    created_at as updated_at
FROM questions_v1_backup
WHERE answer1 IS NOT NULL;

-- Opción 2 (siempre existe)
INSERT INTO question_options (question_id, answer, is_correct, option_order, created_at, updated_at)
SELECT
    id as question_id,
    answer2 as answer,
    (solution = 2) as is_correct,
    2 as option_order,
    created_at,
    created_at as updated_at
FROM questions_v1_backup
WHERE answer2 IS NOT NULL;

-- Opción 3 (siempre existe)
INSERT INTO question_options (question_id, answer, is_correct, option_order, created_at, updated_at)
SELECT
    id as question_id,
    answer3 as answer,
    (solution = 3) as is_correct,
    3 as option_order,
    created_at,
    created_at as updated_at
FROM questions_v1_backup
WHERE answer3 IS NOT NULL;

-- Opción 4 (solo si existe)
INSERT INTO question_options (question_id, answer, is_correct, option_order, created_at, updated_at)
SELECT
    id as question_id,
    answer4 as answer,
    (solution = 4) as is_correct,
    4 as option_order,
    created_at,
    created_at as updated_at
FROM questions_v1_backup
WHERE answer4 IS NOT NULL;

-- -------------------------------------------------------------------------
-- PASO 5: VERIFICACIÓN
-- -------------------------------------------------------------------------

-- Contar preguntas migradas
SELECT 'VERIFICACIÓN: Total de preguntas migradas' as check_name, COUNT(*) as count FROM questions;

-- Contar opciones migradas
SELECT 'VERIFICACIÓN: Total de opciones migradas' as check_name, COUNT(*) as count FROM question_options;

-- Verificar que cada pregunta tiene al menos 3 opciones
SELECT
    'VERIFICACIÓN: Preguntas sin suficientes opciones' as check_name,
    COUNT(*) as count
FROM questions q
WHERE (SELECT COUNT(*) FROM question_options WHERE question_id = q.id) < 3;

-- Verificar que cada pregunta tiene exactamente 1 respuesta correcta
SELECT
    'VERIFICACIÓN: Preguntas sin respuesta correcta' as check_name,
    COUNT(*) as count
FROM questions q
WHERE (SELECT COUNT(*) FROM question_options WHERE question_id = q.id AND is_correct = TRUE) != 1;

-- Verificar integridad referencial
SELECT
    'VERIFICACIÓN: Opciones huérfanas (sin pregunta)' as check_name,
    COUNT(*) as count
FROM question_options qo
WHERE NOT EXISTS (SELECT 1 FROM questions WHERE id = qo.question_id);

-- -------------------------------------------------------------------------
-- PASO 6: OPCIONAL - ELIMINAR BACKUP
-- -------------------------------------------------------------------------

-- PRECAUCIÓN: Solo ejecutar después de verificar que la migración fue exitosa
-- DROP TABLE IF EXISTS questions_v1_backup;

-- =========================================================================
-- NOTAS FINALES
-- =========================================================================

-- DESPUÉS DE LA MIGRACIÓN:
-- 1. Verificar que los conteos son correctos
-- 2. Probar consultas con las vistas: questions_with_options, questions_full
-- 3. Actualizar código Python para usar los nuevos modelos
-- 4. Solo después de confirmar que todo funciona, eliminar questions_v1_backup

-- EJEMPLO DE CONSULTA CON VISTA DE COMPATIBILIDAD:
-- SELECT * FROM questions_with_options WHERE id = 1;

-- EJEMPLO DE CONSULTA CON OPCIONES:
-- SELECT q.*, json_group_array(qo.answer) as all_answers
-- FROM questions q
-- LEFT JOIN question_options qo ON qo.question_id = q.id
-- GROUP BY q.id;
