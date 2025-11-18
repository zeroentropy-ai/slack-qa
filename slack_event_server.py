import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("slack_event_server")

app = FastAPI(title="Slack Event Receiver", version="0.1.0")

latest_action_token: dict[str, Any | None] = {
    "token": None,
    "event": None,
    "last_updated": None,
}


def verify_slack_signature(timestamp: str | None, signature: str | None, body: bytes) -> bool:
    """Validate Slack request signatures to ensure authenticity."""
    signing_secret = os.getenv("SLACK_SIGNING_SECRET")
    if not signing_secret:
        logger.warning("SLACK_SIGNING_SECRET is not set; skipping signature verification.")
        return True

    if not timestamp or not signature:
        logger.error("Missing Slack signature headers.")
        return False

    try:
        ts = int(timestamp)
    except ValueError:
        logger.error("Invalid timestamp header from Slack.")
        return False

    if abs(time.time() - ts) > 60 * 5:
        logger.error("Stale Slack request rejected (timestamp outside 5 minute window).")
        return False

    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    digest = hmac.new(
        signing_secret.encode("utf-8"),
        basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    expected_signature = f"v0={digest}"

    if not hmac.compare_digest(expected_signature, signature):
        logger.error("Slack signature verification failed.")
        return False

    return True


@app.get("/healthz")
async def healthcheck() -> dict[str, Any]:
    """Basic health endpoint for uptime checks."""
    return {
        "ok": True,
        "has_action_token": latest_action_token["token"] is not None,
        "last_updated": latest_action_token["last_updated"],
    }


@app.get("/slack/action-token")
async def get_action_token() -> dict[str, Any]:
    """Retrieve the most recently captured action_token payload."""
    if not latest_action_token["token"]:
        raise HTTPException(status_code=404, detail="No action_token captured yet.")
    return latest_action_token


@app.post("/slack/events")
async def slack_events(request: Request) -> JSONResponse:
    """Primary Slack Events API receiver endpoint."""
    raw_body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    signature = request.headers.get("X-Slack-Signature")

    if not verify_slack_signature(timestamp, signature, raw_body):
        raise HTTPException(status_code=403, detail="Invalid Slack signature.")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Unable to decode Slack payload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    event_type = payload.get("type")
    if event_type == "url_verification":
        challenge = payload.get("challenge")
        if not challenge:
            raise HTTPException(status_code=400, detail="Missing challenge value.")
        logger.info("Responding to Slack URL verification challenge.")
        return JSONResponse({"challenge": challenge})

    if event_type != "event_callback":
        logger.warning("Unhandled Slack event type: %s", event_type)
        return JSONResponse({"ok": True})

    event = payload.get("event", {})
    action_token = event.get("assistant_thread", {}).get("action_token")
    if action_token:
        latest_action_token["token"] = action_token
        latest_action_token["event"] = event
        latest_action_token["last_updated"] = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Captured action_token from event type=%s channel=%s",
            event.get("type"),
            event.get("channel"),
        )
    else:
        logger.info(
            "Received event type=%s channel=%s without action_token.",
            event.get("type"),
            event.get("channel"),
        )

    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "slack_event_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "80")),
        reload=False,
    )
