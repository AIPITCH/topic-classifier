# Evaluation API

Standalone Flask API for Ollama-backed Markdown evaluation against the MISP `content-classification` taxonomy.

## Install

```bash
pip install -r requirements.txt
```

## Configure

Copy the tracked default config, then edit local `config.yaml`:

```bash
cp config.yaml.default config.yaml
```

`config.yaml` is ignored by git so local tokens and deployment settings do not get committed.

Default config template:

```yaml
log: info
logrotate:
  enabled: true
  retention_days: 30
auth:
  enabled: false
  tokens: []
flask:
  listen: 127.0.0.1
  listen_family: ipv4
  client_host: 127.0.0.1
  port: 5151
  max_body_bytes: 2097152
taxonomy:
  url: https://raw.githubusercontent.com/MISP/misp-taxonomies/main/content-classification/machinetag.json
  cache_path: .cache/content-classification.json
  cache_ttl_days: 30
  timeout: 30
queue:
  enabled: true
  allow_sync: true
  cache_path: .cache/classification_jobs
  cache_ttl_hours: 24
  scheduler_interval_seconds: 30
  stale_job_ttl_hours: 24
  workers: 1
health:
  scheduler_interval_seconds: 30
  timeout_seconds: 5
ollama:
  host: localhost
  port: 11434
  engine: gemma4:31b
  allowlist:
    - gemma4:31b
  timeout: 300
  temperature: 0.1
```

Security limits:

- request body limit: `flask.max_body_bytes`
- auth is disabled by default with `auth.enabled: false`
- taxonomy cache refresh: `taxonomy.cache_ttl_days` days
- queued result cache: `queue.cache_ttl_hours` hours
- request timeout override max: `900` seconds
- model override must be in `ollama.allowlist` when allowlist is set

Listen config:

- `flask.listen`: IP address to bind. Use `127.0.0.1`, `0.0.0.0`, `::1`, `::`, or `*`.
- `flask.listen_family`: used only when `flask.listen: "*"`; accepted values are `ipv4`, `ipv6`, or `dual`.
- `flask.listen: "*"` with `listen_family: ipv4` binds `0.0.0.0`.
- `flask.listen: "*"` with `listen_family: ipv6` or `dual` binds `::`.
- `flask.client_host`: host used by the Python client helpers when building the API URL.

Local-only IPv4:

```yaml
flask:
  listen: 127.0.0.1
  listen_family: ipv4
  client_host: 127.0.0.1
  port: 5151
```

Listen on all IPv4 interfaces:

```yaml
flask:
  listen: "*"
  listen_family: ipv4
  client_host: 127.0.0.1
  port: 5151
```

Listen on all IPv6 interfaces:

```yaml
flask:
  listen: "*"
  listen_family: ipv6
  client_host: ::1
  port: 5151
```

Use `listen_family: dual` only when the OS accepts IPv4-mapped traffic on an IPv6 wildcard socket.

## Authentication

Bearer token authentication is optional and disabled by default.

Generate a token:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Enable auth in local `config.yaml`:

```yaml
auth:
  enabled: true
  tokens:
    - paste-generated-token-here
```

Use the token:

```bash
curl -H 'Authorization: Bearer paste-generated-token-here' \
  'http://127.0.0.1:5151/getmodels'
```

If `auth.enabled` is `true` and `tokens` is empty, all protected API requests are denied. Set `auth.enabled: false` or remove the `auth` block to run without auth.
Do not commit real tokens to git; keep local config or deployment secrets private.

## Run

```bash
python3 classification_server.py
```

Override config path:

```bash
CCE_CONFIG=/path/to/config.yaml python3 classification_server.py
```

If local `config.yaml` is missing, the server falls back to `config.yaml.default`.

Default base URL:

```text
http://127.0.0.1:5151
```

## Evaluate

Raw Markdown:

```bash
curl -X POST 'http://127.0.0.1:5151/evaluate?justify=true' \
  -H 'Authorization: Bearer paste-generated-token-here' \
  -H 'Content-Type: text/markdown' \
  --data-binary @sample.md
```

JSON:

```bash
curl -X POST 'http://127.0.0.1:5151/evaluate' \
  -H 'Authorization: Bearer paste-generated-token-here' \
  -H 'Content-Type: application/json' \
  -d '{
    "data": "# Channel dump\n\nSelling leaked databases and credential dumps.",
    "model": "gemma4:31b",
    "justify": true,
    "async": false
  }'
```

Queued JSON:

```bash
curl -X POST 'http://127.0.0.1:5151/evaluate' \
  -H 'Authorization: Bearer paste-generated-token-here' \
  -H 'Content-Type: application/json' \
  -d '{
    "data": "# Channel dump\n\nSelling leaked databases and credential dumps.",
    "justify": true,
    "async": true
  }'
```

Queued response:

```json
{
  "id": "63b7e59d-a58b-4d20-a047-cb5a0e9d0f8f",
  "status": "queued",
  "created_at": 1783660020.123,
  "updated_at": 1783660020.123
}
```

Check queued status:

```bash
curl -H 'Authorization: Bearer paste-generated-token-here' \
  'http://127.0.0.1:5151/evaluate/63b7e59d-a58b-4d20-a047-cb5a0e9d0f8f/status'
```

Fetch queued result:

```bash
curl -H 'Authorization: Bearer paste-generated-token-here' \
  'http://127.0.0.1:5151/evaluate/63b7e59d-a58b-4d20-a047-cb5a0e9d0f8f/result'
```

Response shape:

```json
{
  "labels": [
    {
      "uid": "c49fdfa3-47c5-596f-a157-d14154315dc8",
      "value": "Credential Theft",
      "predicate": "Cybercrime",
      "description": "...",
      "justification": "Evidence from the input."
    }
  ],
  "justify": true,
  "processing_time_seconds": 12.345,
  "truncated": false,
  "input_tokens": 42,
  "input_truncated": false,
  "input_token_limit": 10000
}
```

`justify=true` makes two Ollama calls:

1. evaluate Markdown into taxonomy UIDs
2. challenge selected UIDs and keep only justified labels

Taxonomy cache:

- source: `taxonomy.url`
- cache file: `taxonomy.cache_path`
- refresh at startup when cache is missing or older than `taxonomy.cache_ttl_days`
- stale cache is kept if refresh fails

Queue cache:

- enabled by `queue.enabled`
- synchronous requests can be disabled with `queue.allow_sync: false`
- queued results are stored in `queue.cache_path`
- queued results expire after `queue.cache_ttl_hours`
- queue maintenance runs with one APScheduler `BackgroundScheduler`
- `queue.scheduler_interval_seconds` controls scheduler frequency, default 30
- scheduler physically deletes expired `done`, `failed`, and `expired` job files
- scheduler requeues persisted `queued` jobs after server restart
- scheduler marks stale `queued` and `running` jobs failed after `queue.stale_job_ttl_hours`

Local cache behavior:

- taxonomy cache and queued result cache are local disk files under `.cache/` by default
- taxonomy cache stores the downloaded MISP taxonomy JSON and is refreshed when older than `taxonomy.cache_ttl_days`
- queued result cache stores one JSON file per job UUID under `queue.cache_path`
- queued job files include request metadata, current status, error text when failed, and result data when done
- finished and failed queued jobs become unavailable after `queue.cache_ttl_hours`
- queue scheduler deletes expired job files from disk every `queue.scheduler_interval_seconds`
- deleting `.cache/content-classification.json` forces taxonomy download on next startup
- deleting `.cache/classification_jobs/` removes queued job history and results
- cache files are runtime data and should not be committed

Logs:

- API logs are appended to `log/cc-YYYY-MM-DD.log`
- the server opens a new dated file automatically each local day
- log rotation can be disabled with `logrotate.enabled: false`; disabled mode appends to `log/cc.log`
- old dated log files are pruned automatically after `logrotate.retention_days` days, default 30
- each new `/evaluate` request logs client IP, async flag, justify flag, and model

Health:

- `GET /health` does not require authentication
- `/health` never calls Ollama directly; it returns cached scheduler probe state
- health probe runs once immediately at app startup, then on schedule
- APScheduler probes Ollama every `health.scheduler_interval_seconds`, default 30
- the probe uses `health.timeout_seconds`, default 5
- response is `200` when last probe succeeded
- response is `500` when last probe failed or no successful probe has run yet

## Models

```bash
curl -H 'Authorization: Bearer paste-generated-token-here' \
  'http://127.0.0.1:5151/getmodels'
```

## Health

```bash
curl 'http://127.0.0.1:5151/health'
```

Healthy response:

```json
{
  "status": "ok",
  "ai_engine": "ok",
  "last_check": 1783660020
}
```

Unhealthy response:

```json
{
  "status": "error",
  "ai_engine": "error",
  "error": "ollama timeout",
  "last_check": 1783660020
}
```

## Demo

```bash
python3 demo/test_classify.py
python3 demo/test_classify.py --model gemma4:31b
python3 demo/test_classify.py --model gemma4:31b --warmup
python3 demo/test_classify.py --model gemma4:31b --justify
python3 demo/test_classify.py --list-model
python3 demo/test_classify.py --token paste-generated-token-here
python3 demo/test_classify_queue.py --justify
python3 demo/test_classify_queue.py --token paste-generated-token-here
```

The demo can warm the selected model, loads `demo/test_data/test_sample_channel.json`, converts it to Markdown, then calls `/evaluate`.
The sync demo uses a 120 second request timeout by default.
Both demos also read `CCE_API_TOKEN` when auth is enabled.
The queue demo submits `async=true`, prints the queued job UID, polls `/evaluate/<id>/status` every 15 seconds by default, then fetches `/evaluate/<id>/result`.
The queue demo uses a 120 second request timeout by default.

## Files

- `classification_server.py`: Flask API server
- `client/classifier.py`: Python client helper
- `queries.py`: query loader
- `query/*.txt`: prompt templates
- `.cache/content-classification.json`: runtime taxonomy cache, ignored by git
- `log/cc-YYYY-MM-DD.log`: runtime dated API logs when `logrotate.enabled` is true, ignored by git
- `config.yaml.default`: tracked default config template
- `config.yaml`: local runtime config, ignored by git
- `documentation/README.md`: full route documentation
- `demo/test_classify_queue.py`: independent queued API example
