# Slack-QA: An Annotated Query-Document Dataset on Public Slack Workspaces

This repository contains tools to archive Slack workspaces, clean and chunk messages, generate synthetic search queries against those chunks using LLMs, and convert validated query-chunk pairs to a format compatible with [ZeroEntropy's evals repo](https://github.com/zeroentropy-ai/evals).

We end up with Slack-QA, a dataset of LLM-generated human-like queries with their corresponding answer in chunked documents from scraped public Slack workspaces.

Core features
- Archive Slack workspaces given an auth token for that workspace (both user-browser and workspace-installed app tokens work)
- Clean and load archived Slack data
- Multiple chunking strategies (individual messages, threads, sliding windows)
- Generate synthetic queries using LLMs (OpenAI / Anthropic via `ai.py` wrapper)
- Validate and export query-chunk pairs to BEIR-compatible JSONL

Repository layout (important files)
- `slack_archiver_browser.py` — archiver that uses browser session (`xoxc` + cookies)
- `slack_archiver.py` — archiver for app tokens (`xoxp`) and related utilities
- `slack_archiver_rich.py` — richer archiving utilities and helpers
- `slack_json_cleaner.py` — cleaning utilities to normalize Slack JSON
- `generate_query.py` — main synthetic query generator and pipeline (LLM-based)
- `generate_query_sequential_version.py` — alternate sequential implementation
- `ai.py` — internal AI wrapper used by the query generator (embeddings, calls)
- `slack_search.py` — small wrapper around Slack search API (browser or app mode)
- `assistant_search_context.py` — CLI helper to call Slack’s `assistant.search.context` API once you obtain an `action_token`
- `validated_query_chunk_pairs_to_proper_json_format.py` — converts validated pairs to BEIR format
- `requirements.txt` — Python dependencies used by the project
- `test_boolean_slack_api.py`, `test_search_api_wrapper.py` — small tests / examples

Quick start
1. Create a Python 3.11+ venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Archive a workspace (browser mode example)

 - You need an `xoxc-...` token and the browser cookie string for the Slack workspace. This can be garnered in the network tab of any browser while connected to the slack community in question.

```bash
# edit the top of slack_archiver_browser.py or call the helper function
python slack_archiver_browser.py
```

3. Clean and inspect archived data

 - After archiving, files are written under `slack_archive/<WORKSPACE_NAME>/`.
 - Use `slack_json_cleaner.py` to normalize the JSON into the `channels-clean/` subfolders.

4. Generate synthetic queries (LLM credentials required)

 - `generate_query.py` uses the `ai.py` wrapper. Provide your preferred provider and API key via env vars or pass them to the pipeline.
 - Example (conceptual):

```bash
export ANTHROPIC_API_KEY="..."
python generate_query.py
```

Notes about AI and auth
- `ai.py` centralizes calls to OpenAI/Anthropic and embedding models. Review it before running generation.
- `generate_query.py` supports `provider` values like `openai` and `anthropic`. You must supply the corresponding API key.
- Browser-mode Slack scripts require a valid `xoxc` token and full cookie string. App-mode uses `xoxp` and requires `search:read` and other scopes as appropriate.

Assistant search context (Slack AI)
-----------------------------------
Slack’s [`assistant.search.context`](https://docs.slack.dev/reference/methods/assistant.search.context/) API requires an `action_token` that is only issued when your app subscribes to message events. The included FastAPI server (`slack_event_server.py`) captures the most recent token at `http://127.0.0.1:80/slack/action-token`, and the `assistant_search_context.py` helper makes invoking the API straightforward:

1. Run the event receiver: `PORT=80 python3 slack_event_server.py`
2. Expose it via ngrok (or your reverse proxy): `ngrok http 80`
3. Configure Slack Event Subscriptions to point to `https://<your-tunnel>.ngrok-free.app/slack/events` and subscribe to events such as `message.im`, `message.channels`, and `app_mention`.
4. After Slack confirms the Request URL, send a DM or mention to generate an `action_token`.
5. Invoke the assistant search helper:
   ```bash
   export SLACK_BOT_TOKEN="xoxb-..."
   python assistant_search_context.py --query "mobile UX revamp roadmap"
   ```
   You can also pass `--action-token <token>` explicitly or let the script fetch it from the FastAPI helper (`--action-token-url`).

evals format export
- Use `validated_query_chunk_pairs_to_proper_json_format.py` to convert `validated_query_chunk_pairs.jsonl` into `documents.jsonl`, `queries.jsonl`, and `qrels.jsonl` for retrieval evaluation.


Next steps
- Add a small wrapper script to orchestrate the archive → clean → chunk → generate → validate pipeline.
- Expand dataset with more slack workspaces

Contact
- Open a PR here or contact the [ZeroEntropy Team On Slack](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://join.slack.com/t/zeroentropy-community/shared_invite/zt-2venuk3yt-Of~~NtGIQn_hsD_QJe4fkQ&ved=2ahUKEwj_6e2ImImQAxW3BDQIHeriMA0QFnoECB4QAQ&usg=AOvVaw0-xGP2UpzCaLQoDk6mW_Rr).
