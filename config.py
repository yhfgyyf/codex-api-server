import os
from pathlib import Path

AUTH_JSON_PATH = Path(os.environ.get("CODEX_AUTH_PATH", "~/.codex/auth.json")).expanduser()

# Codex uses chatgpt.com/backend-api with /codex/responses endpoint
CODEX_BASE_URL = os.environ.get("CODEX_BASE_URL", "https://chatgpt.com/backend-api")
CODEX_RESPONSES_PATH = "/codex/responses"

TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = os.environ.get("CODEX_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
SERVER_PORT = int(os.environ.get("CODEX_PORT", "18888"))
# Default to loopback only — set CODEX_HOST=0.0.0.0 to expose externally
SERVER_HOST = os.environ.get("CODEX_HOST", "127.0.0.1")
TOKEN_REFRESH_BUFFER_SECONDS = 300
LOCAL_API_KEY = os.environ.get("CODEX_API_KEY")

DB_PATH = Path(os.environ.get("CODEX_DB_PATH", "~/.codex/api_logs.db")).expanduser()
EXPORT_DIR = Path(os.environ.get("CODEX_EXPORT_DIR", "~/.codex/exports")).expanduser()

# Max request body size (10 MB)
MAX_BODY_BYTES = int(os.environ.get("CODEX_MAX_BODY_BYTES", str(10 * 1024 * 1024)))

CODEX_MODELS = [
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2-codex",
    "gpt-5.1-codex",
]
