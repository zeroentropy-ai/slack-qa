#!/usr/bin/env python3
"""
Utility CLI for Slack's assistant.search.context API.

It expects a Bot User OAuth token (xoxb-...) and an action_token that Slack
delivers via Events API payloads once you subscribe to message events.

Usage examples:

    # Using env vars
    export SLACK_BOT_TOKEN="xoxb-..."
    python assistant_search_context.py --query "zerank2 launch"

    # Explicit action token (e.g., copied from /slack/action-token endpoint)
    python assistant_search_context.py \\
        --query "roadmap updates" \\
        --action-token "atk_123..." \\
        --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import requests

SLACK_ASSISTANT_ENDPOINT = "https://slack.com/api/assistant.search.context"


class SlackAssistantSearchError(RuntimeError):
    """Raised when assistant.search.context cannot be executed successfully."""


def fetch_action_token(
    action_token: Optional[str],
    token_source_url: Optional[str],
    timeout: float = 10.0,
) -> str:
    """
    Resolve an action_token by either using the provided value or fetching it
    from the FastAPI helper endpoint (defaults to http://127.0.0.1:80).
    """
    if action_token:
        return action_token

    source_url = token_source_url or os.getenv(
        "ACTION_TOKEN_URL",
        "https://e28e24d7550d.ngrok-free.app/slack/action-token",
    )

    try:
        response = requests.get(source_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SlackAssistantSearchError(
            f"Failed to fetch action_token from {source_url}: {exc}"
        ) from exc

    data = response.json()
    token = data.get("token")
    if not token:
        raise SlackAssistantSearchError(
            f"No action_token present in response from {source_url}: {json.dumps(data)}"
        )
    return token


def call_assistant_search(
    bot_token: str,
    action_token: str,
    query: str,
    limit: int,
    cursor: Optional[str],
) -> Dict[str, Any]:
    """Invoke Slack's assistant.search.context API."""
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    payload: Dict[str, Any] = {
        "action_token": action_token,
        "query": query,
    }

    if limit:
        payload["limit"] = limit
    if cursor:
        payload["cursor"] = cursor

    try:
        response = requests.post(
            SLACK_ASSISTANT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SlackAssistantSearchError(
            f"HTTP request to {SLACK_ASSISTANT_ENDPOINT} failed: {exc}"
        ) from exc

    data = response.json()
    if not data.get("ok", False):
        raise SlackAssistantSearchError(
            f"Slack API error: {data.get('error', 'unknown_error')} | payload={data}"
        )
    return data


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Slack assistant.search.context with an action_token."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Free-text query to run against Slack content.",
    )
    parser.add_argument(
        "--bot-token",
        default=os.getenv("SLACK_BOT_TOKEN"),
        help="Slack Bot User OAuth token (defaults to $SLACK_BOT_TOKEN).",
    )
    parser.add_argument(
        "--action-token",
        default=os.getenv("SLACK_ACTION_TOKEN"),
        help="Explicit action_token (otherwise fetched from helper endpoint).",
    )
    parser.add_argument(
        "--action-token-url",
        default=os.getenv("ACTION_TOKEN_URL"),
        help="Override helper endpoint used to fetch action_token.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit search results (Slack defaults to 20).",
    )
    parser.add_argument(
        "--cursor",
        help="Optional cursor for pagination (set from response_metadata.next_cursor).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON without formatting.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not args.bot_token:
        print(
            "Missing Bot User OAuth token. Provide --bot-token or set $SLACK_BOT_TOKEN.",
            file=sys.stderr,
        )
        return 2

    try:
        token = fetch_action_token(args.action_token, args.action_token_url)
        response = call_assistant_search(
            bot_token=args.bot_token,
            action_token=token,
            query=args.query,
            limit=args.limit,
            cursor=args.cursor,
        )
    except SlackAssistantSearchError as exc:
        print(f"[assistant.search.context] {exc}", file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(response, indent=2))
    else:
        messages = response.get("results", {}).get("messages", [])
        next_cursor = response.get("response_metadata", {}).get("next_cursor")
        print(f"Found {len(messages)} messages")
        for idx, message in enumerate(messages, start=1):
            channel = message.get("channel_id")
            ts = message.get("message_ts")
            content = message.get("content", "").replace("\n", " ")
            permalink = message.get("permalink", "n/a")
            print(f"[{idx}] {channel} :: {ts}")
            print(f"    {content}")
            print(f"    {permalink}")
        if next_cursor:
            print(f"\nnext_cursor: {next_cursor}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
