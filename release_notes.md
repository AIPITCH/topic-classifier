# Release Notes

## Unreleased

- Add queued classification requests with disk-backed result cache and status/result routes.
- Add independent queued API demo script.
- Add configurable dated daily API log rotation and client IP logging for new evaluation requests.
- Add optional Bearer token authentication.
- Track `config.yaml.default` and ignore local `config.yaml`.
- Add configurable Flask listen address with wildcard IPv4/IPv6 handling.
- Add APScheduler queue maintenance for cleanup, restart requeue, and stale job handling.
- Add cached `/health` endpoint backed by scheduled Ollama probes.
- Shard queued job cache files by first UUID character.
- Improve API file logs with UTC timestamps, IP field, job IDs, user IDs, and queued result retrieval entries.
- Add configurable client IP header for request logs, defaulting to `X-Forwarded-For`.
- Include job IDs in evaluation, justification, and summarization processing logs.
