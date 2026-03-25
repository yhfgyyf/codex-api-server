import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import AsyncGenerator
from pathlib import Path

from config import DB_PATH

logger = logging.getLogger("codex-api-server.db")

_db: sqlite3.Connection | None = None
_lock = asyncio.Lock()

DDL = """
CREATE TABLE IF NOT EXISTS api_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT    NOT NULL,
    endpoint    TEXT    NOT NULL,          -- 'chat/completions' | 'responses'
    model       TEXT    NOT NULL DEFAULT '',
    stream      INTEGER NOT NULL DEFAULT 0,
    request     TEXT    NOT NULL DEFAULT '{}',
    response    TEXT    NOT NULL DEFAULT '{}',
    status      TEXT    NOT NULL DEFAULT 'ok',  -- 'ok' | 'error'
    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
    completion_tokens   INTEGER NOT NULL DEFAULT 0,
    total_tokens        INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint   ON api_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_logs_model      ON api_logs(model);
CREATE INDEX IF NOT EXISTS idx_api_logs_created_at ON api_logs(created_at);
"""


def init_db():
    global _db
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    _db.execute("PRAGMA journal_mode=WAL")
    _db.execute("PRAGMA synchronous=NORMAL")
    _db.executescript(DDL)
    _db.commit()
    logger.info("Database initialized at %s", DB_PATH)


def close_db():
    global _db
    if _db:
        _db.close()
        _db = None


async def save_log(
    request_id: str,
    endpoint: str,
    model: str,
    stream: bool,
    request_body: dict,
    response_body: dict | str,
    status: str = "ok",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    duration_ms: int = 0,
):
    """Save an API call log to the database."""
    async with _lock:
        if not _db:
            return
        try:
            req_json = json.dumps(request_body, ensure_ascii=False)
            resp_json = json.dumps(response_body, ensure_ascii=False) if isinstance(response_body, dict) else response_body
            _db.execute(
                """INSERT INTO api_logs
                   (request_id, endpoint, model, stream, request, response,
                    status, prompt_tokens, completion_tokens, total_tokens,
                    duration_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id, endpoint, model, int(stream), req_json, resp_json,
                    status, prompt_tokens, completion_tokens, total_tokens,
                    duration_ms, time.time(),
                ),
            )
            _db.commit()
        except Exception as e:
            logger.error("Failed to save log: %s", e)


def query_logs(
    endpoint: str | None = None,
    model: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Query logs with optional filters."""
    if not _db:
        return []

    conditions = []
    params: list = []
    if endpoint:
        conditions.append("endpoint = ?")
        params.append(endpoint)
    if model:
        conditions.append("model = ?")
        params.append(model)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM api_logs {where} ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = _db.execute(sql, params).fetchall()
    columns = [desc[0] for desc in _db.execute(f"SELECT * FROM api_logs LIMIT 0").description]
    return [dict(zip(columns, row)) for row in rows]


def count_logs(endpoint: str | None = None, model: str | None = None) -> int:
    if not _db:
        return 0
    conditions = []
    params: list = []
    if endpoint:
        conditions.append("endpoint = ?")
        params.append(endpoint)
    if model:
        conditions.append("model = ?")
        params.append(model)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    row = _db.execute(f"SELECT COUNT(*) FROM api_logs {where}", params).fetchone()
    return row[0] if row else 0


def export_jsonl_iter(
    endpoint: str | None = None,
    model: str | None = None,
) -> list[str]:
    """Export logs as JSONL lines."""
    if not _db:
        return []

    conditions = []
    params: list = []
    if endpoint:
        conditions.append("endpoint = ?")
        params.append(endpoint)
    if model:
        conditions.append("model = ?")
        params.append(model)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM api_logs {where} ORDER BY id ASC"
    cursor = _db.execute(sql, params)
    columns = [desc[0] for desc in cursor.description]

    lines = []
    for row in cursor:
        record = dict(zip(columns, row))
        # Parse JSON fields back to objects for clean export
        for field in ("request", "response"):
            try:
                record[field] = json.loads(record[field])
            except (json.JSONDecodeError, TypeError):
                pass
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


def export_jsonl_file(
    filepath: str | Path,
    endpoint: str | None = None,
    model: str | None = None,
) -> int:
    """Export logs to a JSONL file. Returns number of records exported."""
    lines = export_jsonl_iter(endpoint, model)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return len(lines)
