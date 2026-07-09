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
flask:
  host: 127.0.0.1
  port: 5151
  max_body_bytes: 2097152
taxonomy:
  url: https://raw.githubusercontent.com/MISP/misp-taxonomies/main/content-classification/machinetag.json
  cache_path: .cache/content-classification.json
  cache_ttl_days: 30
  timeout: 30
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
    "justify": true
  }'
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
```

The demo can warm the selected model, loads `demo/test_data/test_sample_channel.json`, converts it to Markdown, then calls `/evaluate`.

## Files

- `classification_server.py`: Flask API server
- `client/classifier.py`: Python client helper
- `queries.py`: query loader
- `query/*.txt`: prompt templates
- `.cache/content-classification.json`: runtime taxonomy cache, ignored by git
- `documentation/README.md`: full route documentation
