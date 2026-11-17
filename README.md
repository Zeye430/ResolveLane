# ResolveLane

ResolveLane is an AI-powered support operations platform for digital marketplaces.

## Development

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,local-embedding,mcp]'
uvicorn resolvelane.api:app --reload
```

Settings are read from environment variables. For local development, copy `.env.example` to `.env` (ignored by git), fill in the values you use, and load it into the shell before starting the server or running scripts:

```bash
cp .env.example .env
set -a && source .env && set +a
```

The `local-embedding` extra installs the offline issue recognition tier, and the `mcp` extra the connections to external services (see below). Its ONNX runtime currently supports Python up to 3.14; on newer Python, install `.[dev]` and the tier is skipped.

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

Late deliveries, delivered-item reports and cancellations without a recorded reason move the case to `needs_review`, as does the assistant when a customer asks for a person; support agents pick it up from their queue (below). When the customer says which items were missing, wrong or damaged, the assistant drafts a refund for a support agent to decide (see Refunds); it never issues or promises one.

### Support desk

Support agents (principals with the `support_agent` role) work through cases in review:

- `GET /api/v1/support/cases?status=needs_review` lists cases longest waiting first, a page at a time (`limit`, and `cursor` from the previous page's `next_cursor`).
- `GET /api/v1/support/cases/{case_id}` shows everything about a case, including what customers do not see: the customer ID, the assistant's review notes, the conversation summary, who wrote each message, and every status change with who made it and why.
- `POST /api/v1/support/cases/{case_id}/messages` sends the customer a message as a support agent. Customers see it with the role `support_agent` but not the agent's ID. The assistant sees it in the conversation and does not contradict it.
- `POST /api/v1/support/cases/{case_id}/resolution` resolves the case with an internal note.

Both actions accept an `Idempotency-Key` header, like customer messages. A resolved case takes no further agent replies (`409`). When the customer writes again, the case goes back to the queue, joining at the end, because a support agent made the last decision; the assistant tells the customer it has been reopened. Customer endpoints refuse support agents and the desk refuses customers (`403`). Cases are not yet assigned to a particular agent, so two agents can open the same case.

### Refunds

A refund is proposed, then approved or rejected by a support agent. Whoever proposes it, the assistant (with its `propose_refund` tool) or a support agent, names only order items, quantities and whether the delivery fee is included. The amount is computed from the unit prices on the order record, never taken from the model or the request. Across all of an order's approved refunds, in any case, each item can be refunded up to the quantity ordered and the delivery fee once. Only delivered orders are refunded, and tax is not refunded in this simulation.

- `POST /api/v1/support/cases/{case_id}/refunds` proposes a refund on a case that has none waiting for a decision.
- `POST /api/v1/support/refunds/{refund_id}/versions` revises a proposal, naming the version it replaces (`revises`).
- `POST /api/v1/support/refunds/{refund_id}/approval` and `.../rejection` decide on a proposal, naming the `version` reviewed, with an internal note (required for a rejection).

Each revision is a new version; earlier versions are kept, with who wrote each and why. A decision is accepted only for the current version, so an agent cannot approve an amount they have not seen: if the assistant revised the proposal after a new customer message, approving the old version returns `409` and names the current one. Approval checks the order's limits again, because another refund of the order may have been approved meanwhile, then sends the customer a message with the amount and items. A decision is final. The assistant drafts at most one open proposal per case and revises it as the customer adds details, but never revises a proposal last written by a support agent. A case with an undecided refund cannot be resolved (`409`).

Customers see approved refunds in their case (`refunds`), not proposals or rejections; the support agent explains a rejection in a reply. The assistant may state an approved refund's amount and items, and the reply check allows saying a refund was approved only once one has been. Refund versions and decisions are protected in the database itself: triggers refuse any update or deletion of a version and any change to a decided refund. All refund actions accept an `Idempotency-Key`.

No money moves yet: an approved refund is a recorded decision, not a payment.


### Storage and retries

Cases, their messages, citations and refunds are stored in SQLite at `data/resolvelane.sqlite3` (ignored by git), so they survive a restart. Set `RESOLVELANE_DATABASE` to use another file, or to `:memory:` to keep nothing; tests use `:memory:`. The schema is versioned and upgraded on startup; a database written by a newer version is refused rather than changed.

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

### Skills

Support operations can adjust how the agent handles a kind of case without changing code, through handling guides in `skills/<name>/SKILL.md` (format in [`skills/README.md`](skills/README.md)). A guide is added to the agent's instructions when the case has raised one of its issues, or the customer's message contains one of its keywords as whole words. Keywords cover situations that are not an issue type: the `allergy-or-illness` guide applies to any message mentioning an allergic reaction or feeling ill.

Guides sit below the agent's fixed rules and cannot override them; they contain guidance, not times, amounts or policy terms, which come from policies and the help center. Every guide is validated at startup, and an invalid one stops the application from starting. The log records which guide versions shaped each reply (`skills=missing-item@2`). Set `RESOLVELANE_SKILLS_DIR` to load guides from another directory. Rule-based replies do not use guides.

### External services (MCP)

The reply agent can use tools of external services through the Model Context Protocol. `mcp_servers.example.json` connects a simulated courier tracking service (`resolvelane.simulator.courier_mcp`, invented data), so the agent can say where an order in progress is:

```bash
export RESOLVELANE_MCP_CONFIG=mcp_servers.example.json
```

What a server says about itself is not trusted:

- Only tools listed in the configuration are offered, and only if the server marks them read-only; the simulator's `reassign_courier` is never offered, even when listed.
- The agent sees the description written in the configuration. The server's tool description and the descriptions in its input schema are dropped, so a server cannot add instructions to the agent's tool list.
- Arguments that identify data are bound in the configuration (`"bind": {"order_id": "order_id"}`): they are filled in from the case's order and are not part of the schema the agent sees, so the agent cannot ask about another customer's order. Only the order or merchant ID can be bound; customer IDs are never sent.
- A stdio server receives only `PATH`, `HOME` and the variables listed in `env_vars`, never the application's keys. A `streamable_http` server takes its authentication headers from environment variables named in `headers_env`.
- Results larger than `max_result_chars` are refused; results are labelled as the service's reports and treated as data. Numbers in them count as sourced in the reply's grounding check.

Connections stay open and are reconnected in the background when they fail. Each server has a circuit breaker (two failures pause it for 30 seconds), and while a server is down its tools are not offered; the agent tells the customer it cannot check rather than guessing. When a server announces that its tools changed (protocol version 2026-07-28 subscriptions), the list is read and checked again; for older servers it is read again on reconnection.

The help center can also be searched by other MCP clients: `python -m resolvelane.mcp_server` serves a read-only `search_help_center` tool over stdio. It exposes only the public help center, never cases or orders.

### Help center

General questions, such as how long a refund takes or whether an address can still be changed, are answered from invented help center articles (`resolvelane.simulator.help_center`). Each article version has an effective period; search covers only the newest version in effect when the question is asked, so outdated guidance cannot be returned or cited.

Search combines keyword ranking (BM25) with similarity from the local embedding model, merged by reciprocal rank fusion. A passage is returned only if it shares a keyword with the question or is clearly similar to it, so a question the help center does not cover returns nothing, and the agent says it does not have that information. Without the local model, search uses keywords only. The agent writes its own search queries and may search more than once.

Replies name the help center sections they rely on; the chat response and the case record them with the article version (`citations`). A reply is replaced by the rule-based reply if it cites a passage that was not returned during that reply, or contains a number (a time, an amount, a number of days) that appears neither in the case facts nor in a cited passage. These checks catch invented citations and figures; they do not prove that every sentence is supported.


### Conversation memory

The reply agent sees three kinds of context, each limited to the customer's own data:

- **Recent messages** of the case, word for word, up to twelve.
- **A summary of older messages.** Once ten messages of a case are not covered by a summary, the language model folds all but the newest four into a new summary on a background thread, so no reply waits for it. Summaries are stored as numbered versions next to the messages, which are never removed or changed, and are not shown to the customer. The summary reports what was said, with customer statements written as claims; it never replaces case facts. If summaries fall behind, for example while the model is down, the agent sees only the newest twelve uncovered messages and the gap is logged.
- **Earlier cases**: the customer's other cases from the last 90 days (up to five), read from the case records: order, issues, status and opening date. The agent uses them when the customer refers to an earlier report, and is instructed never to comment on how often a customer reports problems.

No long-term customer profile (preferences extracted by a model) is kept yet. Without a language model, no summaries are written; replies are then built by rules, which use only the case facts.

## Order lookup

`resolvelane.services.order_lookup.lookup_order` provides read-only access to an order only when the requester's customer ID matches the order owner. Lookups of another customer's order are classified internally as denied but presented to customers as not found, so order IDs cannot be probed.

`GET /api/v1/orders/{order_id}` exposes the lookup to customers. The caller is identified only by a bearer token; identity fields in the query string, headers or body are ignored. Denied and unknown orders both return `404 {"detail": "Order not found"}`. Support agents receive `403` because operator access will be scoped to assigned cases, which do not exist yet. Responses omit internal customer and merchant IDs.

Tokens are configured with `RESOLVELANE_API_TOKENS`, a JSON object mapping each token (at least 24 characters) to a fictional principal:

```bash
export RESOLVELANE_API_TOKENS='{"local-dev-token-customer-101-xyz": {"subject_id": "demo-customer-101", "role": "customer"}}'
curl -H "Authorization: Bearer local-dev-token-customer-101-xyz" http://127.0.0.1:8000/api/v1/orders/demo-order-1001
```

Without tokens, protected endpoints return `503` in development, and the application refuses to start when `RESOLVELANE_ENV=production`. Static tokens are a development mechanism for simulated accounts, not production authentication. Order lookups and case status changes are logged at INFO level under the `resolvelane` logger, including whether a lookup was denied or not found. Logs are not a durable audit trail.
