# Topic Classifier

Topic Classifier is designed to classify conversations and messaging content,
such as Telegram chats and other communication channels. It analyzes text,
identifies relevant categories, and returns matching labels from the MISP
`content-classification` taxonomy.

The project uses Ollama for local analysis and exposes a Flask API. It is
intended for chat messages, discussions, reports, and content feeds collected
from Telegram or other sources, using a shared taxonomy.

## Features

- conversation and messaging-content classification against the MISP taxonomy;
- optional justification of selected labels;
- optional summary of analyzed content;
- synchronous or queued execution;
- HTTP API and Python client;
- local taxonomy and result caching;
- optional Bearer authentication;
- execution with an allowlisted Ollama model;
- optional JSONL collection of LLM-labelled documents for BERTopic training;
- quality-gated BERTopic inference with automatic Ollama fallback.

## Installation and usage

Installation, configuration, authentication, API usage, response formats,
asynchronous jobs, caching, logging, health checks, Python client, and examples
are documented here:

[Technical documentation](documentation/README.md#start) ·
[API routes](documentation/README.md#routes) ·
[Python client](documentation/README.md#python-client)

## License

See [LICENSE](LICENSE).

## AIPitch project

This classifier contributes to [AIPitch](https://aipitch.eu/), the *AI-Powered
Innovative Toolkit for Cybersecurity Hubs*. AIPitch is a European initiative
developing practical AI-powered tools for operational cyberdefence teams,
including national and enterprise Security Operations Centres (SOCs).

The project connects research, data, and operational tools to help teams detect
threats earlier, turn fragmented evidence into actionable intelligence, and
strengthen cyberdefence services. Its implementation period is January 2025 to
December 2027, with a consortium led by NASK PIB and including CIRCL,
Shadowserver Foundation, NCBJ, and ABI LAB.

Project website: [https://aipitch.eu/](https://aipitch.eu/)
