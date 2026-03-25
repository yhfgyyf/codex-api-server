import asyncio
import base64
import json
import logging
import tempfile
import time
from pathlib import Path

import httpx

from config import (
    AUTH_JSON_PATH,
    CLIENT_ID,
    TOKEN_REFRESH_BUFFER_SECONDS,
    TOKEN_URL,
    USER_AGENT,
)

logger = logging.getLogger("codex-api-server.auth")


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without signature verification."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")
    payload_b64 = parts[1]
    # Add padding
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding
    payload_bytes = base64.urlsafe_b64decode(payload_b64)
    return json.loads(payload_bytes)


class TokenManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._access_token: str = ""
        self._refresh_token: str = ""
        self._account_id: str = ""
        self._expires_at: float = 0
        self._auth_data: dict = {}
        self._load_auth()

    def _load_auth(self):
        """Load credentials from auth.json."""
        if not AUTH_JSON_PATH.exists():
            raise FileNotFoundError(f"Auth file not found: {AUTH_JSON_PATH}")

        with open(AUTH_JSON_PATH) as f:
            self._auth_data = json.load(f)

        tokens = self._auth_data.get("tokens", {})
        self._access_token = tokens.get("access_token", "")
        self._refresh_token = tokens.get("refresh_token", "")
        self._account_id = tokens.get("account_id", "")

        if not self._access_token:
            raise ValueError("No access_token in auth.json")

        # Decode JWT to get expiry
        try:
            payload = _decode_jwt_payload(self._access_token)
            self._expires_at = payload.get("exp", 0)
            # Extract account_id from JWT if not in tokens
            if not self._account_id:
                auth_claim = payload.get("https://api.openai.com/auth", {})
                self._account_id = auth_claim.get("chatgpt_account_id", "")
        except Exception as e:
            logger.warning(f"Failed to decode JWT: {e}")
            self._expires_at = time.time() + 3600  # assume 1h validity

    def _is_expired(self) -> bool:
        return time.time() >= (self._expires_at - TOKEN_REFRESH_BUFFER_SECONDS)

    def _save_auth(self):
        """Atomically write updated tokens back to auth.json."""
        self._auth_data["tokens"]["access_token"] = self._access_token
        self._auth_data["tokens"]["refresh_token"] = self._refresh_token
        self._auth_data["tokens"]["account_id"] = self._account_id
        self._auth_data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

        parent = AUTH_JSON_PATH.parent
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with open(fd, "w") as f:
                json.dump(self._auth_data, f, indent=2)
            Path(tmp_path).replace(AUTH_JSON_PATH)
            AUTH_JSON_PATH.chmod(0o600)
        except Exception:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
            raise

    async def _do_refresh(self):
        """Refresh the access token using the refresh token."""
        logger.info("Refreshing Codex OAuth token...")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": CLIENT_ID,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": USER_AGENT,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        self._access_token = data["access_token"]
        if "refresh_token" in data:
            self._refresh_token = data["refresh_token"]

        # Update expiry from new JWT
        try:
            payload = _decode_jwt_payload(self._access_token)
            self._expires_at = payload.get("exp", 0)
            auth_claim = payload.get("https://api.openai.com/auth", {})
            if auth_claim.get("chatgpt_account_id"):
                self._account_id = auth_claim["chatgpt_account_id"]
        except Exception:
            expires_in = data.get("expires_in", 3600)
            self._expires_at = time.time() + expires_in

        self._save_auth()
        logger.info("Token refreshed successfully, expires at %s", time.ctime(self._expires_at))

    async def refresh_if_needed(self):
        """Refresh token if expired, with lock to prevent concurrent refreshes."""
        if not self._is_expired():
            return
        async with self._lock:
            # Double-check after acquiring lock
            if not self._is_expired():
                return
            # Re-read auth.json in case another process refreshed it
            try:
                self._load_auth()
                if not self._is_expired():
                    return
            except Exception:
                pass
            await self._do_refresh()

    async def get_headers(self) -> dict[str, str]:
        """Get authentication headers for upstream requests."""
        await self.refresh_if_needed()
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "User-Agent": USER_AGENT,
        }
        if self._account_id:
            headers["ChatGPT-Account-Id"] = self._account_id
        return headers

    @property
    def is_valid(self) -> bool:
        return bool(self._access_token)

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def expires_at(self) -> float:
        return self._expires_at
