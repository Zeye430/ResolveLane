# ResolveLane

ResolveLane is an AI-powered support operations platform for digital marketplaces.

## Development

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,local-embedding]'
uvicorn resolvelane.api:app --reload
```

Settings are read from environment variables. For local development, copy `.env.example` to `.env` (ignored by git), fill in the values you use, and load it into the shell before starting the server or running scripts:

```bash
cp .env.example .env
set -a && source .env && set +a
```

The `local-embedding` extra installs the offline issue recognition tier (see below). Its ONNX runtime currently supports Python up to 3.14; on newer Python, install `.[dev]` and the tier is skipped.

The health endpoint is available at `http://127.0.0.1:8000/health` and the API documentation at `http://127.0.0.1:8000/docs`.

Run tests with `pytest`.

The health endpoint reports that the application is running; it does not check external services or which issue recognition tiers are available.

## Delivery examples

The delivery domain includes fictional orders, delivery events, support policies, and customer-report scenarios for development and tests. Customer reports are not verified facts, and expected review paths are test descriptions, not implemented functionality. The examples do not contain real platform records, execute refunds, or represent the policies of an existing marketplace.

## Support chat

`POST /api/v1/chat` accepts a customer message and an optional order ID (an order ID written in the message also works), and replies with what the order record shows:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Authorization: Bearer local-dev-token-customer-101-xyz" -H "Content-Type: application/json" \
  -d '{"message": "My order arrived late.", "order_id": "demo-order-1001"}'
```

Each conversation is a case. The first message returns a `case_id`; sending it with later messages continues the case. A case remembers its order, so follow-ups need not repeat the order ID, and an issue raised before the order ID is known is answered once it arrives. A case is bound only to an order the customer may access and cannot switch to another order (`409`). Another customer's case returns `404`, the same as an unknown case. `GET /api/v1/cases/{case_id}` returns the case with its messages.

Late deliveries, delivered-item reports and cancellations without a recorded reason move the case to `needs_review`. No support agent workflow exists yet to pick it up, and no refund is issued.

### Storage and retries

Cases, their messages and citations are stored in SQLite at `data/resolvelane.sqlite3` (ignored by git), so they survive a restart. Set `RESOLVELANE_DATABASE` to use another file, or to `:memory:` to keep nothing; tests use `:memory:`. The schema is versioned and upgraded on startup; a database written by a newer version is refused rather than changed.

A client that loses a response, for example to a timeout, can send the message again safely by setting an `Idempotency-Key` header, a unique value per message such as a UUID:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Authorization: Bearer local-dev-token-customer-101-xyz" -H "Content-Type: application/json" \
  -H "Idempotency-Key: 6f1c2a9e-3b7d-4e0a-9c55-2d8e1f4b7a60" \
  -d '{"message": "My order arrived late.", "order_id": "demo-order-1001"}'
```

The response is stored with the case in one transaction. A retry with the same key and body within 24 hours returns the original response with an `Idempotent-Replayed: true` header and changes nothing, even after a restart. The same key with a different body returns `422`. Keys are scoped to the customer. A message that fails leaves no receipt and can be retried as new. Without the header, every request is handled as a new message.

The server runs as a single process: a lock in that process orders the changes to each case. Running several worker processes would need a server database with row locking.

### Issue recognition

Each message is classified into the supported issues by a ladder of tiers. The first tier that answers wins; a tier that is unsure abstains, and a tier whose provider fails is skipped:

1. **Language model** (`llm`): uses the last few messages of the case as context and can find several issues in one message. Any Anthropic-compatible endpoint works.
2. **Remote embedding** (`remote_embedding`): compares the message with example messages for each issue using an embedding model from a different provider, so it keeps working when the language model provider is down. Any OpenAI-compatible `/embeddings` endpoint works.
3. **Local embedding** (`local_embedding`): the same comparison with a small model on this machine, so it works without a network connection. The model is loaded at startup and cached in a fixed directory.
4. **Keywords** (`keywords`): fixed phrases.

When no tier answers, the issue is unknown and the reply lists what support can help with instead of guessing. The model and embedding tiers only choose labels from a fixed list; they do not decide which order a customer may see or what happens to a case. Order IDs are extracted from the message by a rule. Tests never call a provider.

```bash
export ANTHROPIC_API_KEY=...                          # enables the language model
export ANTHROPIC_MODEL=claude-opus-5-5                # optional; this is the default
export ANTHROPIC_BASE_URL=...                         # optional, for a compatible provider
export RESOLVELANE_LLM_TIMEOUT_S=10                   # optional

export RESOLVELANE_EMBEDDING_API_KEY=...              # enables the remote embedding tier
export RESOLVELANE_EMBEDDING_BASE_URL=https://api.voyageai.com/v1    # optional; this is the default
export RESOLVELANE_EMBEDDING_MODEL=voyage-4-lite      # optional; this is the default

export RESOLVELANE_LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5       # optional; "off" disables the tier
export RESOLVELANE_MODEL_CACHE_DIR=~/.cache/resolvelane/models        # optional; this is the default
```

The language model tier is skipped for 30 seconds after three consecutive provider failures, and the remote embedding tier for 60 seconds after one; after the pause a single request tries the provider again. Unusable model answers (invalid or low-confidence JSON, refusals) fall through to the next tier without counting as provider failures.

An embedding tier answers only when a message is similar enough to the examples of one issue, using thresholds measured per model with `python scripts/calibrate_embeddings.py local|remote`. A model without measured thresholds is not used unless `RESOLVELANE_EMBEDDING_THRESHOLD` and `RESOLVELANE_EMBEDDING_MARGIN` (or the `RESOLVELANE_LOCAL_EMBEDDING_` equivalents) are set. Thresholds have been measured on 44 held-out messages for `voyage-4-lite` (35 correct, none wrong, 9 passed to the next tier) and `BAAI/bge-small-en-v1.5` (39 correct, 2 wrong, 3 passed on). Those figures come from the same messages the thresholds were fitted to, so they are optimistic. Voyage accounts without a payment method are limited to 3 requests a minute, too few for this tier during an outage.

### Replies

When a language model is configured, a reply agent writes each reply; otherwise replies are built by rules. Rules still make every decision: which order the customer may see, how late a delivery was, which policy is in effect, and whether the case goes to a support agent. The agent receives those results as case facts, together with the rule-based reply, and answers the customer's actual question from them.

The agent can call three tools through a gateway: `search_help_center` for general questions, `get_support_policy` for an issue whose policy is not already in the case facts, and `request_human_review` when the customer asks for a person or raises something the facts and policies do not settle. Tool inputs never name a customer or an order; the gateway gives each tool only the authenticated customer's case. A review request is a proposal that the conversation service applies to the case after re-checking it.

A reply is not used, and the rule-based reply is sent instead, when the model is unavailable, stops early, makes more than three calls or four tool calls, writes more than 1,200 characters, or appears to promise a refund, credit, replacement or compensation. That last check is a phrase filter backing up the instructions, not a guarantee. When the case needs review, the reply always says so. Recognition and replies share the model's circuit breaker, so once the provider is found to be down, neither waits for it. If another message changes the case while a reply is being written, the reply is decided again by rules from the current case.

### Help center

General questions, such as how long a refund takes or whether an address can still be changed, are answered from invented help center articles (`resolvelane.simulator.help_center`). Each article version has an effective period; search covers only the newest version in effect when the question is asked, so outdated guidance cannot be returned or cited.

Search combines keyword ranking (BM25) with similarity from the local embedding model, merged by reciprocal rank fusion. A passage is returned only if it shares a keyword with the question or is clearly similar to it, so a question the help center does not cover returns nothing, and the agent says it does not have that information. Without the local model, search uses keywords only. The agent writes its own search queries and may search more than once.

Replies name the help center sections they rely on; the chat response and the case record them with the article version (`citations`). A reply is replaced by the rule-based reply if it cites a passage that was not returned during that reply, or contains a number (a time, an amount, a number of days) that appears neither in the case facts nor in a cited passage. These checks catch invented citations and figures; they do not prove that every sentence is supported.


## Order lookup

`resolvelane.services.order_lookup.lookup_order` provides read-only access to an order only when the requester's customer ID matches the order owner. Lookups of another customer's order are classified internally as denied but presented to customers as not found, so order IDs cannot be probed.

`GET /api/v1/orders/{order_id}` exposes the lookup to customers. The caller is identified only by a bearer token; identity fields in the query string, headers or body are ignored. Denied and unknown orders both return `404 {"detail": "Order not found"}`. Support agents receive `403` because operator access will be scoped to assigned cases, which do not exist yet. Responses omit internal customer and merchant IDs.

Tokens are configured with `RESOLVELANE_API_TOKENS`, a JSON object mapping each token (at least 24 characters) to a fictional principal:

```bash
export RESOLVELANE_API_TOKENS='{"local-dev-token-customer-101-xyz": {"subject_id": "demo-customer-101", "role": "customer"}}'
curl -H "Authorization: Bearer local-dev-token-customer-101-xyz" http://127.0.0.1:8000/api/v1/orders/demo-order-1001
```

Without tokens, protected endpoints return `503` in development, and the application refuses to start when `RESOLVELANE_ENV=production`. Static tokens are a development mechanism for simulated accounts, not production authentication. Order lookups and case status changes are logged at INFO level under the `resolvelane` logger, including whether a lookup was denied or not found. Logs are not a durable audit trail.
