from __future__ import annotations

import sqlite3
from pathlib import Path


DB_FILENAME = "pipelines.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pipelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    data_source TEXT,
    technical_prompt_id INTEGER CHECK (technical_prompt_id IS NULL OR technical_prompt_id > 0),
    clinical_prompt_id INTEGER CHECK (clinical_prompt_id IS NULL OR clinical_prompt_id > 0),
    ensemble_enabled INTEGER NOT NULL DEFAULT 0 CHECK (ensemble_enabled IN (0, 1)),
    ensemble_size INTEGER NOT NULL DEFAULT 3 CHECK (ensemble_size >= 1),
    chain_of_verification INTEGER NOT NULL DEFAULT 0 CHECK (chain_of_verification IN (0, 1)),
    contextual_grounding INTEGER NOT NULL DEFAULT 0 CHECK (contextual_grounding IN (0, 1)),
    llm_as_judge INTEGER NOT NULL DEFAULT 0 CHECK (llm_as_judge IN (0, 1)),
    judge_model_id INTEGER CHECK (judge_model_id IS NULL OR judge_model_id > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (llm_as_judge = 0 OR judge_model_id IS NOT NULL),
    CHECK (ensemble_enabled = 0 OR ensemble_size >= 2)
);

CREATE TABLE IF NOT EXISTS pipeline_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    step_order INTEGER NOT NULL CHECK (step_order >= 1),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pipeline_models_pipeline_step
    ON pipeline_models (pipeline_id, step_order);

CREATE INDEX IF NOT EXISTS idx_pipeline_models_model
    ON pipeline_models (model_id);

CREATE TRIGGER IF NOT EXISTS trg_pipelines_touch_timestamp
AFTER UPDATE ON pipelines
FOR EACH ROW
WHEN NEW.updated_at <= OLD.updated_at
BEGIN
    UPDATE pipelines
    SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    WHERE id = NEW.id;
END;
"""


def ensure_database(db_path: Path, ddl: str) -> None:
    """Create the SQLite database and apply the schema if needed."""
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.executescript(ddl)
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        print(f"Initialized SQLite database at {db_path}")
        print(f"Tables: {', '.join(tables)}")


if __name__ == "__main__":
    ensure_database(Path(__file__).resolve().parent / DB_FILENAME, SCHEMA)
