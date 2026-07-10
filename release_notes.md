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
