#!/usr/bin/env python
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import html
import json
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any
from urllib.parse import urlencode
import uuid

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tips_ask import INDEX_PATH, MAX_CONTEXT_HITS, answer_question, augment_hits, build_codex_context, expand_neighbor_hits, load_index, rerank_context_hits, retrieve


ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_google_oauth_client_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    client_type = "web" if data.get("web") else "installed" if data.get("installed") else "raw"
    client = data.get("web") or data.get("installed") or data
    client_id = str(client.get("client_id") or "").strip()
    client_secret = str(client.get("client_secret") or "").strip()
    if client_id and "GOOGLE_CLIENT_ID" not in os.environ:
        os.environ["GOOGLE_CLIENT_ID"] = client_id
    if client_secret and "GOOGLE_CLIENT_SECRET" not in os.environ:
        os.environ["GOOGLE_CLIENT_SECRET"] = client_secret
    if "GOOGLE_CLIENT_TYPE" not in os.environ:
        os.environ["GOOGLE_CLIENT_TYPE"] = client_type


load_env_file(ROOT / "secrets" / "tips_oauth.env")
load_google_oauth_client_file(Path(os.environ.get("GOOGLE_CLIENT_SECRETS_FILE", ROOT / "secrets" / "google_oauth_client.json")))

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_CLIENT_TYPE = os.environ.get("GOOGLE_CLIENT_TYPE", "").strip()
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()
TIPS_PUBLIC_BASE_URL = os.environ.get("TIPS_PUBLIC_BASE_URL", "").strip().rstrip("/")
ALLOWED_EMAIL = "jsmcel@gmail.com"
SESSION_SECRET = os.environ.get("TIPS_SESSION_SECRET", "").strip() or secrets.token_urlsafe(48)
SESSION_COOKIE = "tips_session"
OAUTH_STATE_COOKIE = "tips_oauth_state"
SESSION_MAX_AGE = 12 * 60 * 60
OAUTH_STATE_MAX_AGE = 10 * 60
COOKIE_SECURE = os.environ.get("TIPS_COOKIE_SECURE", "").lower() in {"1", "true", "yes"}
AUTH_DISABLED = os.environ.get("TIPS_AUTH_DISABLED", "").lower() in {"1", "true", "yes", "on"}
OPEN_AUTH_PATHS = {"/login", "/auth/google", "/oauth2/callback", "/oauth2/redirect-uri", "/logout", "/favicon.ico"}
ACCESS_REQUESTS_PATH = ROOT / "output" / "access_requests.jsonl"

app = FastAPI(title="TIPS Local Bot", version="1.0")
ASK_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("TIPS_ASK_WORKERS", "2")))
ASK_JOBS: dict[str, dict[str, Any]] = {}
ASK_JOBS_LOCK = threading.Lock()
ASK_JOB_TTL = int(os.environ.get("TIPS_ASK_JOB_TTL", "3600"))


def oauth_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def external_request_base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip()
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc)).split(",", 1)[0].strip()
    return f"{proto}://{host}".rstrip("/")


def callback_uri(request: Request) -> str:
    if TIPS_PUBLIC_BASE_URL:
        return f"{TIPS_PUBLIC_BASE_URL}/oauth2/callback"
    if GOOGLE_REDIRECT_URI:
        return GOOGLE_REDIRECT_URI
    return f"{external_request_base_url(request)}/oauth2/callback"


def public_auth_url(request: Request) -> str | None:
    if not TIPS_PUBLIC_BASE_URL:
        return None
    if external_request_base_url(request).lower() == TIPS_PUBLIC_BASE_URL.lower():
        return None
    query = f"?{request.url.query}" if request.url.query else ""
    return f"{TIPS_PUBLIC_BASE_URL}{request.url.path}{query}"


def clean_next_path(value: str | None) -> str:
    if not value:
        return "/"
    value = value.strip()
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    if value.startswith(("/login", "/auth/google", "/oauth2/callback", "/access-request")):
        return "/"
    return value


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign_payload(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.setdefault("iat", int(time.time()))
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(SESSION_SECRET.encode("utf-8"), raw, hashlib.sha256).digest()
    return f"{b64url_encode(raw)}.{b64url_encode(sig)}"


def read_signed_payload(token: str | None, max_age: int) -> dict[str, Any] | None:
    if not token or "." not in token:
        return None
    raw_b64, sig_b64 = token.split(".", 1)
    try:
        raw = b64url_decode(raw_b64)
        sig = b64url_decode(sig_b64)
    except Exception:
        return None
    expected = hmac.new(SESSION_SECRET.encode("utf-8"), raw, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    iat = int(payload.get("iat") or 0)
    if not iat or time.time() - iat > max_age:
        return None
    return payload


def read_session(request: Request) -> dict[str, Any] | None:
    payload = read_signed_payload(request.cookies.get(SESSION_COOKIE), SESSION_MAX_AGE)
    if not payload:
        return None
    email = str(payload.get("email") or "").strip().lower()
    if email != ALLOWED_EMAIL:
        return None
    return payload


def set_signed_cookie(response: RedirectResponse | JSONResponse, name: str, payload: dict[str, Any], max_age: int) -> None:
    response.set_cookie(
        name,
        sign_payload(payload),
        max_age=max_age,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
    )


def clear_auth_cookies(response: RedirectResponse | JSONResponse | HTMLResponse) -> None:
    for name in (SESSION_COOKIE, OAUTH_STATE_COOKIE):
        response.delete_cookie(name, httponly=True, secure=COOKIE_SECURE, samesite="lax")


def wants_json(request: Request) -> bool:
    path = request.url.path
    accept = request.headers.get("accept", "")
    return path in {"/ask", "/manifest", "/health", "/me"} or "application/json" in accept


def login_html(request: Request, message: str = "", access_message: str = "") -> str:
    next_path = clean_next_path(request.query_params.get("next"))
    config_note = ""
    login_href = f"/auth/google?{urlencode({'next': next_path})}"
    access_href = "/auth/google?mode=request_access"
    login_disabled = ""
    if not oauth_configured():
        login_href = "#"
        access_href = "#"
        login_disabled = " disabled"
        config_note = (
            "<p class=\"warning\">Missing <code>GOOGLE_CLIENT_ID</code> and/or "
            "<code>GOOGLE_CLIENT_SECRET</code>. The app stays closed until OAuth is configured.</p>"
        )
    message_html = f"<p class=\"error\">{html.escape(message)}</p>" if message else ""
    access_html = f"<p class=\"ok\">{html.escape(access_message)}</p>" if access_message else ""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>TIPS GPT - Access</title>
    <style>
      :root {{ color-scheme: light; --ink:#191917; --muted:#6f6a60; --line:#ddd8cc; --green:#0f8f72; --paper:#fffdf8; --page:#f4f2ec; }}
      * {{ box-sizing: border-box; }}
      body {{ margin:0; min-height:100dvh; display:grid; place-items:center; background:var(--page); color:var(--ink); font-family:ui-sans-serif,"Segoe UI",Aptos,Calibri,sans-serif; padding:22px; }}
      main {{ width:min(460px,100%); background:var(--paper); border:1px solid var(--line); border-radius:8px; box-shadow:0 18px 50px rgba(36,35,31,.14); padding:26px; }}
      .mark {{ width:48px; height:48px; display:grid; place-items:center; border-radius:8px; background:var(--green); color:white; font-weight:900; margin-bottom:18px; }}
      h1 {{ margin:0 0 8px; font-size:24px; letter-spacing:0; }}
      p {{ margin:0 0 14px; line-height:1.45; color:var(--muted); }}
      .google, button {{ width:100%; min-height:46px; border-radius:8px; border:2px solid #bdb5a6; background:white; color:var(--ink); font-weight:900; text-decoration:none; display:grid; place-items:center; cursor:pointer; }}
      .google {{ margin:16px 0 10px; border-color:var(--green); background:#e4f4ee; }}
      .access {{ margin-top:10px; border-color:#245b89; background:#edf4fb; }}
      .google.disabled {{ pointer-events:none; opacity:.55; }}
      .error,.warning {{ color:#9d2e25; font-weight:800; }}
      .ok {{ color:#0a6d58; font-weight:800; }}
      code {{ color:var(--ink); }}
    </style>
  </head>
  <body>
    <main>
      <div class="mark">T</div>
      <h1>TIPS GPT</h1>
      <p>Restricted access. Only the authorized Google account can sign in: <strong>{html.escape(ALLOWED_EMAIL)}</strong>.</p>
      {message_html}
      {config_note}
      <a class="google{login_disabled}" href="{html.escape(login_href)}">Sign in with Google</a>
      <a class="google access{login_disabled}" href="{html.escape(access_href)}">Request access with Google</a>
      <p>The access request button also goes through Google OAuth. The email is recorded only if Google confirms it belongs to the authenticated account.</p>
      {access_html}
    </main>
  </body>
</html>"""


def unauthorized_response(request: Request) -> JSONResponse | RedirectResponse:
    if wants_json(request):
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return RedirectResponse(f"/login?{urlencode({'next': request.url.path})}", status_code=303)


def write_access_request(request: Request, user: dict[str, Any]) -> None:
    ACCESS_REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACCESS_REQUESTS_PATH.open("a", encoding="utf-8").write(
        json.dumps(
            {
                "ts": int(time.time()),
                "google_email": str(user.get("email") or "").strip().lower(),
                "google_sub": user.get("sub") or "",
                "email_verified": user.get("email_verified"),
                "name": user.get("name") or "",
                "remote": request.client.host if request.client else "",
                "user_agent": request.headers.get("user-agent", ""),
            },
            ensure_ascii=False,
        )
        + "\n"
    )


@app.middleware("http")
async def require_google_session(request: Request, call_next):
    path = request.url.path
    if AUTH_DISABLED:
        request.state.user = {"email": "lan@local", "name": "LAN local"}
        return await call_next(request)
    if path in OPEN_AUTH_PATHS:
        return await call_next(request)
    session = read_session(request)
    if not session:
        return unauthorized_response(request)
    request.state.user = session
    return await call_next(request)


class ChatTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 24
    mode: str = "answer"
    use_codex: bool = True
    model: str = "codex_high"
    language: str = "auto"
    history: list[ChatTurn] = Field(default_factory=list)


def history_for_model(history: list[ChatTurn]) -> list[dict[str, str]]:
    return [
        {"role": turn.role, "content": turn.content}
        for turn in history[-10:]
        if turn.role in {"user", "assistant"} and turn.content.strip()
    ]


def contextual_query(question: str, history: list[ChatTurn]) -> str:
    parts = [question.strip()]
    recent = [
        f"{turn.role}: {turn.content.strip()}"
        for turn in history[-10:]
        if turn.role in {"user", "assistant"} and turn.content.strip()
    ]
    if recent:
        parts.append("Recent chat for resolving follow-up references:")
        parts.extend(recent)
    return "\n".join(parts)


@app.get("/login")
def login(request: Request):
    if AUTH_DISABLED:
        return RedirectResponse("/", status_code=303)
    if read_session(request):
        return RedirectResponse(clean_next_path(request.query_params.get("next")), status_code=303)
    return HTMLResponse(login_html(request))


@app.get("/auth/google")
def auth_google(request: Request):
    if not oauth_configured():
        return HTMLResponse(login_html(request, "OAuth de Google no esta configurado."), status_code=503)
    public_url = public_auth_url(request)
    if public_url:
        return RedirectResponse(public_url, status_code=303)
    mode = "request_access" if request.query_params.get("mode") == "request_access" else "login"
    redirect_uri = callback_uri(request)
    state = sign_payload(
        {
            "nonce": secrets.token_urlsafe(18),
            "next": clean_next_path(request.query_params.get("next")),
            "mode": mode,
            "redirect_uri": redirect_uri,
        }
    )
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    response = RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=303)
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        state,
        max_age=OAUTH_STATE_MAX_AGE,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
    )
    return response


@app.get("/oauth2/callback")
def oauth_callback(request: Request):
    if not oauth_configured():
        return HTMLResponse(login_html(request, "OAuth de Google no esta configurado."), status_code=503)
    error = request.query_params.get("error")
    if error:
        return HTMLResponse(login_html(request, f"Google ha devuelto un error: {error}"), status_code=401)
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    cookie_state = request.cookies.get(OAUTH_STATE_COOKIE)
    if not code or not state or not cookie_state or not hmac.compare_digest(state, cookie_state):
        return HTMLResponse(login_html(request, "Sesion OAuth invalida. Vuelve a iniciar sesion."), status_code=401)
    state_payload = read_signed_payload(state, OAUTH_STATE_MAX_AGE)
    if not state_payload:
        return HTMLResponse(login_html(request, "Sesion OAuth caducada. Vuelve a iniciar sesion."), status_code=401)
    redirect_uri = str(state_payload.get("redirect_uri") or callback_uri(request))

    try:
        token_response = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
        token_response.raise_for_status()
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise RuntimeError("Google did not return an access_token")
        user_response = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20,
        )
        user_response.raise_for_status()
        user = user_response.json()
    except Exception as exc:
        return HTMLResponse(login_html(request, f"Could not validate the Google account: {exc}"), status_code=401)

    email = str(user.get("email") or "").strip().lower()
    verified = user.get("email_verified")
    is_verified = verified is True or str(verified).lower() == "true"
    if not email or not is_verified:
        response = HTMLResponse(
            login_html(request, "Google did not confirm a verified email for this account."),
            status_code=403,
        )
        clear_auth_cookies(response)
        return response

    if state_payload.get("mode") == "request_access":
        write_access_request(request, user)
        response = HTMLResponse(
            login_html(
                request,
                access_message=f"Access request recorded for {email}. Google validated that this email belongs to the authenticated account.",
            )
        )
        clear_auth_cookies(response)
        return response

    if email != ALLOWED_EMAIL:
        response = HTMLResponse(
            login_html(
                request,
                f"The account {email} is validated by Google, but is not authorized. Only {ALLOWED_EMAIL} can sign in.",
            ),
            status_code=403,
        )
        clear_auth_cookies(response)
        return response

    response = RedirectResponse(clean_next_path(str(state_payload.get("next") or "/")), status_code=303)
    clear_auth_cookies(response)
    set_signed_cookie(
        response,
        SESSION_COOKIE,
        {
            "email": email,
            "name": user.get("name") or email,
            "picture": user.get("picture") or "",
        },
        SESSION_MAX_AGE,
    )
    return response


@app.get("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse("/login", status_code=303)
    clear_auth_cookies(response)
    return response


@app.get("/oauth2/redirect-uri")
def oauth_redirect_uri(request: Request) -> dict[str, Any]:
    return {
        "redirect_uri": callback_uri(request),
        "public_base_url": TIPS_PUBLIC_BASE_URL or None,
        "request_base_url": external_request_base_url(request),
        "google_client_type": GOOGLE_CLIENT_TYPE or None,
        "allowed_email": ALLOWED_EMAIL,
        "auth_disabled": AUTH_DISABLED,
    }


@app.get("/me")
def me(request: Request) -> dict[str, Any]:
    return {
        "email": request.state.user.get("email"),
        "name": request.state.user.get("name"),
        "allowed_email": ALLOWED_EMAIL,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": INDEX_PATH.exists(), "index": str(INDEX_PATH)}


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    path = ROOT / "data" / "processed" / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run python tips_ingest.py first")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def error_answer(req: AskRequest, exc: Exception) -> dict[str, Any]:
    return {
        "question": req.question,
        "answer": (
            "Could not complete this query, but the web app is still running. "
            "Try rephrasing it with a concrete TIPS term, or switch to Context mode to inspect the retrieved evidence."
        ),
        "citations": [],
        "confidence": "low",
        "generated_by": "error_guard",
        "error": str(exc),
    }


def answer_request(req: AskRequest) -> dict[str, Any]:
    try:
        search_query = contextual_query(req.question, req.history)
        if req.mode == "context":
            index = load_index()
            retrieval_k = max(req.top_k, 32)
            hits = retrieve(index, search_query, top_k=retrieval_k)
            hits = augment_hits(index, search_query, hits)
            hits = expand_neighbor_hits(index, hits, max_neighbors=1, max_total=min(retrieval_k + 16, MAX_CONTEXT_HITS))
            hits = rerank_context_hits(search_query, hits, max_total=min(retrieval_k + 16, MAX_CONTEXT_HITS))
            return build_codex_context(search_query, hits, max_hits=req.top_k)
        return answer_question(
            req.question,
            top_k=req.top_k,
            language=req.language,
            generate=req.use_codex and req.model != "local_rag",
            model_preset=req.model,
            retrieval_query=search_query,
            chat_history=history_for_model(req.history),
        )
    except Exception as exc:
        return error_answer(req, exc)


def cleanup_ask_jobs() -> None:
    cutoff = time.time() - ASK_JOB_TTL
    with ASK_JOBS_LOCK:
        for job_id, job in list(ASK_JOBS.items()):
            if job.get("status") in {"done", "error", "cancelled"} and float(job.get("updated_at") or 0) < cutoff:
                ASK_JOBS.pop(job_id, None)


def set_ask_job(job_id: str, **updates: Any) -> None:
    with ASK_JOBS_LOCK:
        job = ASK_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def ask_job_snapshot(job_id: str) -> dict[str, Any]:
    with ASK_JOBS_LOCK:
        job = ASK_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Ask job not found")
        return {key: value for key, value in job.items() if key != "future"}


def run_ask_job(job_id: str, req: AskRequest) -> None:
    set_ask_job(job_id, status="running", stage="retrieving")
    try:
        result = answer_request(req)
        with ASK_JOBS_LOCK:
            job = ASK_JOBS.get(job_id)
            if not job:
                return
            if job.get("status") == "cancelled":
                return
            job.update(status="done", stage="done", result=result, updated_at=time.time())
    except Exception as exc:
        set_ask_job(job_id, status="error", stage="error", error=str(exc), result=error_answer(req, exc))


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    return answer_request(req)


@app.post("/ask/jobs")
def start_ask_job(req: AskRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    cleanup_ask_jobs()
    job_id = uuid.uuid4().hex
    now = time.time()
    with ASK_JOBS_LOCK:
        ASK_JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "question": req.question,
            "created_at": now,
            "updated_at": now,
        }
    future = ASK_EXECUTOR.submit(run_ask_job, job_id, req)
    with ASK_JOBS_LOCK:
        if job_id in ASK_JOBS:
            ASK_JOBS[job_id]["future"] = future
    return {"job_id": job_id, "status": "queued"}


@app.get("/ask/jobs/{job_id}")
def get_ask_job(job_id: str) -> dict[str, Any]:
    return ask_job_snapshot(job_id)


@app.delete("/ask/jobs/{job_id}")
def cancel_ask_job(job_id: str) -> dict[str, Any]:
    with ASK_JOBS_LOCK:
        job = ASK_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Ask job not found")
        future = job.get("future")
        if future:
            future.cancel()
        job.update(status="cancelled", stage="cancelled", updated_at=time.time())
    return ask_job_snapshot(job_id)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(PUBLIC / "index.html")


app.mount("/static", StaticFiles(directory=PUBLIC), name="static")


if __name__ == "__main__":
    host = os.environ.get("TIPS_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("TIPS_WEB_PORT", "8787"))
    uvicorn.run("tips_web:app", host=host, port=port, reload=False)
