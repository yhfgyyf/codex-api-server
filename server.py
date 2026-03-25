import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from auth import TokenManager
from config import CODEX_MODELS, LOCAL_API_KEY, SERVER_HOST, SERVER_PORT
from models import get_model_info, get_model_list
from proxy import OpenAIProxy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("codex-api-server")

token_manager: TokenManager | None = None
proxy: OpenAIProxy | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global token_manager, proxy
    token_manager = TokenManager()
    proxy = OpenAIProxy(token_manager)
    logger.info(
        "Server started. Token valid: %s, expires: %s, account: %s",
        token_manager.is_valid,
        time.ctime(token_manager.expires_at),
        token_manager.account_id,
    )
    yield
    if proxy:
        await proxy.close()


app = FastAPI(title="Codex API Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_api_key(request: Request):
    if not LOCAL_API_KEY:
        return
    auth_header = request.headers.get("authorization", "")
    key = auth_header[7:] if auth_header.startswith("Bearer ") else auth_header
    if key != LOCAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "token_valid": token_manager.is_valid if token_manager else False,
        "token_expires": time.ctime(token_manager.expires_at) if token_manager else None,
        "account_id": token_manager.account_id if token_manager else None,
        "models": CODEX_MODELS,
    }


@app.get("/v1/models")
async def list_models(request: Request):
    _check_api_key(request)
    return get_model_list().model_dump()


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, request: Request):
    _check_api_key(request)
    info = get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return info.model_dump()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_api_key(request)
    body = await request.json()
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            proxy.chat_completions_stream(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await proxy.chat_completions(body)
    status = 200 if "error" not in result else (result.get("error", {}).get("status", 500))
    return JSONResponse(content=result, status_code=status if isinstance(status, int) else 500)


@app.post("/v1/responses")
async def responses(request: Request):
    _check_api_key(request)
    body = await request.json()
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            proxy.responses_proxy_stream(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await proxy.responses_proxy(body)
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        access_log=True,
    )
