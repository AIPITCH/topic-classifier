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

Override config path:

```bash
CCE_CONFIG=/path/to/config.yaml python3 classification_server.py
```

Prompt templates live in `query/*.txt` and are mapped by `query/queries.json`.
The taxonomy is downloaded from `taxonomy.url` into `taxonomy.cache_path` at
startup when missing or older than `taxonomy.cache_ttl_days`. Existing stale
cache is kept if refresh fails.

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

JSON body parameters:

- `data`: Markdown text. Required unless using raw body
- `markdown`, `md`, `text`: aliases for `data`
- `model`: optional Ollama model override
- `timeout`: optional positive integer seconds, max `900`
- `include_raw`: optional boolean
- `justify`: optional boolean

Raw Markdown example:

```bash
curl -X POST 'http://127.0.0.1:5151/evaluate?justify=true' \
  -H 'Content-Type: text/markdown' \
  --data-binary @sample.md
```

JSON example:

```bash
curl -X POST 'http://127.0.0.1:5151/evaluate' \
  -H 'Content-Type: application/json' \
  -d '{
    "data": "# Channel dump\n\nSelling leaked databases and credential dumps.",
    "model": "gemma4:31b",
    "justify": true,
    "include_raw": false
  }'
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
- `413 {"error": "request body too large"}`
- `502 {"error": "ollama evaluation failed"}`
- `502 {"error": "ollama evaluation returned invalid json"}` when `justify=true` and the second Ollama call does not return parseable JSON

### `GET /getmodels`

Fetch available Ollama completion models live, refresh server cache, return model names and context lengths.

Query parameters:

- `capability`: optional capability filter

Example:

```bash
curl 'http://127.0.0.1:5151/getmodels'
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
- `502 {"error": "ollama warmup failed"}`

## Python Client

Minimal runnable example:

```bash
python3 demo/test_classify.py
python3 demo/test_classify.py --model gemma4:31b
python3 demo/test_classify.py --model gemma4:31b --warmup
python3 demo/test_classify.py --justify
python3 demo/test_classify.py --list-model
```

The example can warm the selected model, loads `demo/test_data/test_sample_channel.json`, converts it to Markdown, then calls `/evaluate`.
`--list-model` lists available models and exits without calling `/evaluate`.

Basic client usage:

```python
from client.classifier import ClassificationClient

client = ClassificationClient.from_config()

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
