import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

from config import DB_PATH, EXPORT_DIR

logger = logging.getLogger("codex-api-server.db")

_db: sqlite3.Connection | None = None
_lock = asyncio.Lock()
_columns: list[str] = []

DDL = """
CREATE TABLE IF NOT EXISTS api_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT    NOT NULL,
    endpoint    TEXT    NOT NULL,
    model       TEXT    NOT NULL DEFAULT '',
    stream      INTEGER NOT NULL DEFAULT 0,
    request     TEXT    NOT NULL DEFAULT '{}',
    response    TEXT    NOT NULL DEFAULT '{}',
    status      TEXT    NOT NULL DEFAULT 'ok',
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
    global _db, _columns
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    _db.execute("PRAGMA journal_mode=WAL")
    _db.execute("PRAGMA synchronous=NORMAL")
    _db.executescript(DDL)
    _db.commit()
    # Cache column names
    cursor = _db.execute("SELECT * FROM api_logs LIMIT 0")
    _columns = [desc[0] for desc in cursor.description]
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Database initialized at %s", DB_PATH)


def close_db():
    global _db
    if _db:
        _db.close()
        _db = None


def _build_where(endpoint: str | None, model: str | None) -> tuple[str, list]:
    conditions = []
    params: list = []
    if endpoint:
        conditions.append("endpoint = ?")
        params.append(endpoint)
    if model:
        conditions.append("model = ?")
        params.append(model)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return where, params


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


async def query_logs(
    endpoint: str | None = None,
    model: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    async with _lock:
        if not _db:
            return []
        where, params = _build_where(endpoint, model)
        sql = f"SELECT * FROM api_logs {where} ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = _db.execute(sql, params).fetchall()
        return [dict(zip(_columns, row)) for row in rows]


async def count_logs(endpoint: str | None = None, model: str | None = None) -> int:
    async with _lock:
        if not _db:
            return 0
        where, params = _build_where(endpoint, model)
        row = _db.execute(f"SELECT COUNT(*) FROM api_logs {where}", params).fetchone()
        return row[0] if row else 0


async def export_jsonl_iter(
    endpoint: str | None = None,
    model: str | None = None,
) -> list[str]:
    async with _lock:
        if not _db:
            return []
        where, params = _build_where(endpoint, model)
        sql = f"SELECT * FROM api_logs {where} ORDER BY id ASC"
        cursor = _db.execute(sql, params)
        lines = []
        for row in cursor:
            record = dict(zip(_columns, row))
            for field in ("request", "response"):
                try:
                    record[field] = json.loads(record[field])
                except (json.JSONDecodeError, TypeError):
                    pass
            lines.append(json.dumps(record, ensure_ascii=False))
        return lines


def _validate_export_path(filepath: str) -> Path:
    """Resolve and validate export path is within EXPORT_DIR."""
    resolved = Path(filepath).expanduser().resolve()
    allowed = EXPORT_DIR.resolve()
    if not str(resolved).startswith(str(allowed) + "/") and resolved != allowed:
        raise ValueError(f"Export path must be within {EXPORT_DIR}")
    return resolved


async def export_jsonl_file(
    filepath: str,
    endpoint: str | None = None,
    model: str | None = None,
) -> tuple[int, Path]:
    """Export logs to a JSONL file within EXPORT_DIR. Returns (count, resolved_path)."""
    resolved = _validate_export_path(filepath)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lines = await export_jsonl_iter(endpoint, model)
    with open(resolved, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return len(lines), resolved
