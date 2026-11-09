# Architecture

## Design goal

The model must never define authorization scope. A request is authenticated and reduced to an explicit patient/source scope before any retrieval or model component sees record text.

## Request path

1. Authenticate the user and unlock the local encrypted workspace.
2. Resolve role and patient access.
3. Resolve intent and dialogue state.
4. Route to structured fact/event lookup or patient-scoped retrieval.
5. Validate source provenance against the current source version/hash.
6. Apply safety boundaries.
7. Use the local model only when open synthesis is actually needed.

## Retrieval

The default path is patient-scoped SQLite FTS5. The codebase also contains evaluation-only paths for dense embeddings, Reciprocal Rank Fusion, cross-encoder reranking, and decision/rescue gating.

All experimental paths accept an already-authorized source set. If a caller mixes another patient's document into the candidate input, the hybrid workflow fails closed rather than filtering it silently.

## Evidence and dialogue state

Conversation history and evidence are deliberately separate.

Conversation state is stored per user/patient/session and helps interpret follow-ups. Record-grounded answers reload reviewed sources; previous assistant prose cannot become a source of truth. When the patient revision changes, stale dialogue state is invalidated.

Derived facts retain source identity, source version, SHA-256 content hash, exact quote, and character offsets. This allows a fact or medication event to be rejected if its source no longer matches the stored snapshot.

## Model boundary

The model adapter targets an OpenAI-compatible local endpoint. Remote inference is blocked by default and can only be enabled for one explicitly configured HTTPS origin.

The runtime uses bounded history and output budgets. Several structured record routes avoid model inference entirely.

## Storage

Application state is stored in an encrypted local SQLite snapshot. Account passwords are hashed with Argon2; the workspace snapshot is protected with authenticated encryption. Runtime vaults and databases are excluded from the public repository.

## Why this matters

The main engineering work in an applied LLM system is often outside the model itself: scope control, evidence integrity, failure handling, reproducible evaluation, and deciding when *not* to use a model.
