# Evaluation API

Local Flask API for Ollama-based Markdown evaluation against the MISP
`content-classification` taxonomy.

## Start

Install deps:

```bash
pip install -r requirements.txt
```

Start server:

```bash
python3 classification_server.py
```

Default config: `config.yaml`

```yaml
log: info
logrotate:
  enabled: true
  retention_days: 30
auth:
  enabled: false
  tokens: []
flask:
  host: 127.0.0.1
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
  workers: 1
ollama:
  host: localhost
  port: 11434
  engine: gemma4:31b
  allowlist:
    - gemma4:31b
  timeout: 300
  temperature: 0.1
```

Override config path:

```bash
CCE_CONFIG=/path/to/config.yaml python3 classification_server.py
```

Prompt templates live in `query/*.txt` and are mapped by `query/queries.json`.
The taxonomy is downloaded from `taxonomy.url` into `taxonomy.cache_path` at
startup when missing or older than `taxonomy.cache_ttl_days`. Existing stale
cache is kept if refresh fails.
Queued results are stored in `queue.cache_path` and expire after
`queue.cache_ttl_hours`.

## Authentication

Bearer token authentication uses `Flask-HTTPAuth` and is disabled by default.

Generate a token:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Enable auth in `config.yaml`:

```yaml
auth:
  enabled: true
  tokens:
    - paste-generated-token-here
```

Use the token with API requests:

```bash
curl -H 'Authorization: Bearer paste-generated-token-here' \
  'http://127.0.0.1:5151/getmodels'
```

Protected routes:

- `GET /getmodels`
- `POST /evaluate`
- `GET /evaluate/<job_id>/status`
- `GET /evaluate/<job_id>/result`
- `POST /warmup_model`

If `auth.enabled` is `true` and `tokens` is empty, all protected API requests are denied. Set `auth.enabled: false` or remove the `auth` block to run without auth.
Do not commit real tokens to git; keep local config or deployment secrets private.

## Local Cache

The API uses local disk cache files under `.cache/` by default.

Taxonomy cache:

- file path: `taxonomy.cache_path`
- default: `.cache/content-classification.json`
- contents: downloaded MISP `content-classification` taxonomy JSON
- refresh: at startup when missing or older than `taxonomy.cache_ttl_days`
- failure mode: stale cache is kept when refresh fails
- manual reset: delete `.cache/content-classification.json`

Queued result cache:

- directory: `queue.cache_path`
- default: `.cache/classification_jobs`
- contents: one JSON file per queued job UUID
- job file data: request metadata, `queued`/`running`/`done`/`failed`/`expired` status, timestamps, error text, and result data
- TTL: finished and failed jobs expire after `queue.cache_ttl_hours`
- manual reset: delete `.cache/classification_jobs/`

Cache files are runtime data and should not be committed.

## Logs

API logs are appended to `log/cc-YYYY-MM-DD.log` when `logrotate.enabled` is
true. The server automatically opens a new dated log file each local day and
prunes dated log files older than `logrotate.retention_days`; the default is 30
days. When `logrotate.enabled` is false, logs are appended to `log/cc.log`.

New `POST /evaluate` requests log:

- client IP, using the first `X-Forwarded-For` value when present
- `async` flag
- `justify` flag
- selected model

Status and result polling routes do not emit this new-query log line.

## Routes

Base URL default: `http://127.0.0.1:5151`

### `POST /evaluate`

Evaluate Markdown against the cached `content-classification` taxonomy.

Server behavior:

- prepends taxonomy prompt and taxonomy entries
- refreshes taxonomy cache at startup when missing or older than `taxonomy.cache_ttl_days`
- truncates only user Markdown to 10,000 tokens
- asks Ollama for taxonomy UUIDs
- corrects a UUID internally only when exactly one character is wrong and only one taxonomy UUID matches
- maps UUIDs back to public taxonomy labels
- when `justify=true`, makes a second Ollama call and keeps only labels justified by that second call

Accepted body formats:

- raw Markdown body, e.g. `Content-Type: text/markdown`
- JSON body with one of `data`, `markdown`, `md`, `text`

Query parameters:

- `model`: optional Ollama model override
- `timeout`: optional positive integer seconds, max `900`
- `include_raw`: optional `1`, `true`, or `yes`
- `justify`: optional `1`, `true`, or `yes`
- `async`: optional `1`, `true`, or `yes` to queue work and return a job ID

JSON body parameters:

- `data`: Markdown text. Required unless using raw body
- `markdown`, `md`, `text`: aliases for `data`
- `model`: optional Ollama model override
- `timeout`: optional positive integer seconds, max `900`
- `include_raw`: optional boolean
- `justify`: optional boolean
- `async`: optional boolean

Raw Markdown example:

```bash
curl -X POST 'http://127.0.0.1:5151/evaluate?justify=true' \
  -H 'Authorization: Bearer paste-generated-token-here' \
  -H 'Content-Type: text/markdown' \
  --data-binary @sample.md
```

JSON example:

```bash
curl -X POST 'http://127.0.0.1:5151/evaluate' \
  -H 'Authorization: Bearer paste-generated-token-here' \
  -H 'Content-Type: application/json' \
  -d '{
    "data": "# Channel dump\n\nSelling leaked databases and credential dumps.",
    "model": "gemma4:31b",
    "justify": true,
    "include_raw": false,
    "async": false
  }'
```

Queued JSON example:

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

Response:

```json
{
  "labels": [
    {
      "uid": "c49fdfa3-47c5-596f-a157-d14154315dc8",
      "value": "Credential Theft",
      "predicate": "Cybercrime",
      "description": "...",
      "justification": "The text explicitly mentions leaked databases and credential dumps."
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

Errors:

- `400 {"error": "data must be non-empty markdown text"}`
- `400 {"error": "model must be string"}`
- `400 {"error": "timeout must be integer"}`
- `400 {"error": "timeout must be positive"}`
- `400 {"error": "timeout must be <= 900"}`
- `400 {"error": "model is not allowed: <model>"}`
- `400 {"error": "async queue is disabled"}`
- `400 {"error": "synchronous evaluation is disabled"}`
- `401 {"error": "authentication required"}`
- `413 {"error": "request body too large"}`
- `502 {"error": "ollama evaluation failed"}`
- `502 {"error": "ollama evaluation returned invalid json"}` when `justify=true` and the second Ollama call does not return parseable JSON

### `GET /evaluate/<job_id>/status`

Return queued job status.

Response:

```json
{
  "id": "63b7e59d-a58b-4d20-a047-cb5a0e9d0f8f",
  "status": "running",
  "created_at": 1783660020.123,
  "updated_at": 1783660022.456
}
```

Statuses:

- `queued`
- `running`
- `done`
- `failed`
- `expired`

Errors:

- `404 {"error": "job not found"}`
- `401 {"error": "authentication required"}`

### `GET /evaluate/<job_id>/result`

Return queued result when complete. Add `include_raw=true` to include raw model output.

Responses:

- `200`: same shape as synchronous `/evaluate`
- `202`: job still `queued` or `running`
- `401 {"error": "authentication required"}`
- `404 {"error": "job not found"}`
- `404 {"error": "job expired"}`
- `500 {"error": "<job error>"}`

### `GET /getmodels`

Fetch available Ollama completion models live, refresh server cache, return model names and context lengths.

Query parameters:

- `capability`: optional capability filter

Example:

```bash
curl -H 'Authorization: Bearer paste-generated-token-here' \
  'http://127.0.0.1:5151/getmodels'
```

Response:

```json
[
  {
    "name": "gemma4:31b",
    "context_length": 131072
  }
]
```

Errors:

- `401 {"error": "authentication required"}`
- `502 {"error": "ollama model listing failed"}` if Ollama request fails

### `POST /warmup_model`

Send tiny prompt to Ollama to load model before timed requests.

JSON body parameters:

- `model`: optional model override. Defaults to configured `ollama.engine`
- `timeout`: optional positive integer seconds, max `900`

Query parameters:

- `model`: optional model override
- `timeout`: optional positive integer seconds, max `900`

Example:

```bash
curl -X POST 'http://127.0.0.1:5151/warmup_model' \
  -H 'Authorization: Bearer paste-generated-token-here' \
  -H 'Content-Type: application/json' \
  -d '{"model": "gemma4:31b"}'
```

Response:

```json
{"status": "ok"}
```

Errors:

- `400 {"error": "invalid json body"}`
- `400 {"error": "model must be string"}`
- `400 {"error": "model is not allowed: <model>"}`
- `400 {"error": "timeout must be <= 900"}`
- `401 {"error": "authentication required"}`
- `502 {"error": "ollama warmup failed"}`

## Python Client

Minimal runnable example:

```bash
python3 demo/test_classify.py
python3 demo/test_classify.py --model gemma4:31b
python3 demo/test_classify.py --model gemma4:31b --warmup
python3 demo/test_classify.py --justify
python3 demo/test_classify.py --list-model
python3 demo/test_classify.py --token paste-generated-token-here
python3 demo/test_classify_queue.py --justify
python3 demo/test_classify_queue.py --token paste-generated-token-here
```

The example can warm the selected model, loads `demo/test_data/test_sample_channel.json`, converts it to Markdown, then calls `/evaluate`.
`--list-model` lists available models and exits without calling `/evaluate`.
The sync example uses a 120 second request timeout by default.
Both examples also read `CCE_API_TOKEN` when auth is enabled.
The queue example submits `async=true`, prints the queued job UID, polls `/evaluate/<id>/status` every 15 seconds by default, then fetches `/evaluate/<id>/result`.
The queue example uses a 120 second request timeout by default.

Basic client usage:

```python
from client.classifier import ClassificationClient

client = ClassificationClient.from_config()
client.api_token = "paste-generated-token-here"

models = client.get_models()
print(models)

result = client.evaluate_markdown(
    "# Channel dump\n\nSelling leaked databases and credential dumps.",
    justify=True,
)
print(result["labels"])
```

Client methods:

- `ClassificationClient.from_config(path="config.yaml")`
- `ClassificationClient(api_token="paste-generated-token-here")`
- `ClassificationClient.get_models(capability="completion")`
- `ClassificationClient.warmup_model(model=None, timeout=None)`
- `ClassificationClient.evaluate_markdown(markdown, model=None, timeout=None, include_raw=False, justify=False)`

## Notes

- `/evaluate` is the only classification/evaluation endpoint.
- `/getmodels` calls Ollama and returns available models.
- Default model: `gemma4:31b`.
- Default timeout: 300 seconds.
- Server timeout override max: 900 seconds.
- Default max request body: 2097152 bytes.
