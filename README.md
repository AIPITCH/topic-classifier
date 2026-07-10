# Evaluation API

Standalone Flask API for Ollama-backed Markdown evaluation against the MISP `content-classification` taxonomy.

## Install

```bash
pip install -r requirements.txt
```

## Configure

Edit `config.yaml`:

```yaml
log: info
logrotate:
  enabled: true
  retention_days: 30
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

Security limits:

- request body limit: `flask.max_body_bytes`
- taxonomy cache refresh: `taxonomy.cache_ttl_days` days
- queued result cache: `queue.cache_ttl_hours` hours
- request timeout override max: `900` seconds
- model override must be in `ollama.allowlist` when allowlist is set

## Run

```bash
python3 classification_server.py
```

Override config path:

```bash
CCE_CONFIG=/path/to/config.yaml python3 classification_server.py
```

Default base URL:

```text
http://127.0.0.1:5151
```

## Evaluate

Raw Markdown:

```bash
curl -X POST 'http://127.0.0.1:5151/evaluate?justify=true' \
  -H 'Content-Type: text/markdown' \
  --data-binary @sample.md
```

JSON:

```bash
curl -X POST 'http://127.0.0.1:5151/evaluate' \
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
curl 'http://127.0.0.1:5151/evaluate/63b7e59d-a58b-4d20-a047-cb5a0e9d0f8f/status'
```

Fetch queued result:

```bash
curl 'http://127.0.0.1:5151/evaluate/63b7e59d-a58b-4d20-a047-cb5a0e9d0f8f/result'
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

Local cache behavior:

- taxonomy cache and queued result cache are local disk files under `.cache/` by default
- taxonomy cache stores the downloaded MISP taxonomy JSON and is refreshed when older than `taxonomy.cache_ttl_days`
- queued result cache stores one JSON file per job UUID under `queue.cache_path`
- queued job files include request metadata, current status, error text when failed, and result data when done
- finished and failed queued jobs become unavailable after `queue.cache_ttl_hours`
- deleting `.cache/content-classification.json` forces taxonomy download on next startup
- deleting `.cache/classification_jobs/` removes queued job history and results
- cache files are runtime data and should not be committed

Logs:

- API logs are appended to `log/cc-YYYY-MM-DD.log`
- the server opens a new dated file automatically each local day
- log rotation can be disabled with `logrotate.enabled: false`; disabled mode appends to `log/cc.log`
- old dated log files are pruned automatically after `logrotate.retention_days` days, default 30
- each new `/evaluate` request logs client IP, async flag, justify flag, and model

## Models

```bash
curl 'http://127.0.0.1:5151/getmodels'
```

## Demo

```bash
python3 demo/test_classify.py
python3 demo/test_classify.py --model gemma4:31b
python3 demo/test_classify.py --model gemma4:31b --warmup
python3 demo/test_classify.py --model gemma4:31b --justify
python3 demo/test_classify.py --list-model
python3 demo/test_classify_queue.py --justify
```

The demo can warm the selected model, loads `demo/test_data/test_sample_channel.json`, converts it to Markdown, then calls `/evaluate`.
The sync demo uses a 120 second request timeout by default.
The queue demo submits `async=true`, prints the queued job UID, polls `/evaluate/<id>/status` every 15 seconds by default, then fetches `/evaluate/<id>/result`.
The queue demo uses a 120 second request timeout by default.

## Files

- `classification_server.py`: Flask API server
- `client/classifier.py`: Python client helper
- `queries.py`: query loader
- `query/*.txt`: prompt templates
- `.cache/content-classification.json`: runtime taxonomy cache, ignored by git
- `log/cc-YYYY-MM-DD.log`: runtime dated API logs when `logrotate.enabled` is true, ignored by git
- `documentation/README.md`: full route documentation
- `demo/test_classify_queue.py`: independent queued API example
