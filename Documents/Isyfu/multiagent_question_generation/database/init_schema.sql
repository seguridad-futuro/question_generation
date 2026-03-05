-- Schema para SQLite
-- Compatible con el modelo Question y con tracking de calidad/deduplicación

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    academy INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer1 TEXT NOT NULL,
    answer2 TEXT NOT NULL,
    answer3 TEXT NOT NULL,
    answer4 TEXT,
    solution INTEGER NOT NULL CHECK(solution >= 1 AND solution <= 4),
    tip TEXT,
    topic INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    question_prompt TEXT,
    llm_model TEXT,
    order_num INTEGER,
    by_llm BOOLEAN DEFAULT 1,

    -- Quality tracking
    faithfulness_score REAL CHECK(faithfulness_score IS NULL OR (faithfulness_score >= 0 AND faithfulness_score <= 1)),
    relevancy_score REAL CHECK(relevancy_score IS NULL OR (relevancy_score >= 0 AND relevancy_score <= 1)),
    source_chunk_id TEXT,
    source_document TEXT,

    -- Deduplication & retry
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of INTEGER REFERENCES questions(id) ON DELETE SET NULL,
    retry_count INTEGER DEFAULT 0,
    needs_manual_review BOOLEAN DEFAULT FALSE
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_topic ON questions(topic);
CREATE INDEX IF NOT EXISTS idx_academy ON questions(academy);
CREATE INDEX IF NOT EXISTS idx_duplicate ON questions(is_duplicate);
CREATE INDEX IF NOT EXISTS idx_manual_review ON questions(needs_manual_review);
CREATE INDEX IF NOT EXISTS idx_source_chunk ON questions(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON questions(created_at DESC);

-- Table para almacenar metadata de batches
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_name TEXT UNIQUE NOT NULL,
    topic INTEGER NOT NULL,
    academy INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_questions INTEGER DEFAULT 0,
    unique_questions INTEGER DEFAULT 0,
    duplicates INTEGER DEFAULT 0,
    manual_review_count INTEGER DEFAULT 0,
    avg_faithfulness REAL,
    avg_relevancy REAL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_batch_status ON batches(status);
CREATE INDEX IF NOT EXISTS idx_batch_created ON batches(created_at DESC);
