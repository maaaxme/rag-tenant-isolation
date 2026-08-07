# RAG Tenant Isolation Test

A reproducible test for one specific question: **in a multi-tenant RAG system, can retrieval hand a model context from a tenant it has no business seeing -- even when your database access control is airtight?**

Short answer from building this myself: yes, easily, and it looks fine until you check for it specifically.

## The problem

Row-level security in Postgres (or any relational access control) protects rows returned by SQL queries. A RAG pipeline's retrieval step is a different code path: it embeds a query, runs a similarity search against a vector index, and hands the top-k matching chunks to the model as context. **RLS never sees that query.** There is no `WHERE tenant_id = ...` for the vector index to enforce, because similarity search doesn't go through SQL row access at all -- it goes through the embedding index directly.

The practical consequence: you can have a perfectly correct RLS policy on your Postgres tables and still leak tenant A's document chunks into a session that's scoped to tenant B, because nothing in the retrieval step ever checked which tenant a chunk belongs to. This is not a Postgres bug and not a vector-database bug -- it's a gap between two components that each do their own job correctly and neither one covers the seam between them.

This is easy to miss because most questions still get answered "correctly" -- the top-matching chunks are usually still the right ones. The leak shows up on the questions that don't obviously call for cross-tenant material, and it shows up as a plausible-sounding wrong answer, not a crash.

## What's in this repo

- `before/rag_core_buggy.py` -- the version with no tenant boundary at all, kept for reference, plus a real captured run showing the actual mixing failure (see below).
- `rag_core.py` -- the fixed version: every chunk is tagged with a tenant ID at index time, and `retrieve()` requires a tenant context and removes every chunk from a different tenant **before** the similarity search runs, not after.
- `test_suite.py` -- 12 questions across 3 fictional tenants plus firm-wide context, 5 of them adversarial "trap" questions, run against the fixed version.
- `corpus/` -- the fictional test documents (5 short case files, entirely invented, see `corpus/README.md`).
- `results/` -- the actual captured output of the test run (`test_results.md` / `.json` in German, matching the corpus language; `test_results_en.md` / `.json` after running `test_suite.py` yourself).

## The bug, for real

This started as a real mistake in my own build, not a constructed example. Before the fix, a question scoped to Tenant 2 ("summarize the Familie Keskin file") produced this (German original, from `before/buggy_run_output.json`):

> "Zusätzlich droht eine Versäumnisurteil und die Präklusion des Vortrags zur betriebsbedingten Kündigung nach § 296 ZPO, wenn die Klageerwiderungsfrist am 14. August 2026 nicht eingehalten wird."

Translated: *"There is also a risk of a default judgment and preclusion of the argument on the operational dismissal under Sec. 296 ZPO, if the response deadline of 14 August 2026 is missed."*

That entire paragraph belongs to Tenant 1's employment dispute (a different case, different client, different deadline). It has nothing to do with Tenant 2's tenancy dispute. The model didn't hallucinate this -- it was handed a chunk from the wrong tenant in its context window and used it exactly as instructed. The retrieval step ran a flat top-k similarity search over the entire corpus with no tenant boundary, and that specific chunk happened to score well enough to make the top 4.

## The fix

Tag every chunk with a tenant ID at index time (here: a case reference number extracted from the source document). Require every retrieval call to carry an explicit tenant context, coming from the application/session, never from the free-text query. Filter out every chunk that doesn't belong to that tenant (or to firm-wide documents) **before** ranking by similarity -- a structural exclusion, not a relevance threshold that only works "most of the time." Even when a query itself names another tenant or smuggles in one of its facts (the trap questions below), no chunk from that tenant is ever a candidate, because it was removed before the search ran.

## The test methodology

12 questions against 3 fictional tenants (an employment dispute, a tenancy dispute, a contract dispute) plus a firm-wide context (documents that apply across all tenants, e.g. an internal confidentiality policy). 5 of the 12 are trap questions that deliberately embed a wrong fact from another tenant into the question itself -- e.g. asking "is the deadline in file B the same 14 August as in the other file?" when 14 August is actually file A's deadline. This checks whether the system repeats or corrects a false premise, which is a stronger test than asking a neutral question.

Every case is checked two ways, independently:
1. **Structural check** -- did retrieval return even a single chunk tagged with a different tenant? This runs regardless of what the model says in its answer, so a leak can't be masked by the model happening to ignore the bad context.
2. **Content check** -- does the answer contain a forbidden foreign-tenant fact, or is it missing a fact it should have?

Result on the fixed version: **12/12 passed, zero structural leaks in any case, including all 5 trap questions.** Full output: `results/test_results.md`.

## Reproduce it yourself

Requirements: Python 3, a local [Ollama](https://ollama.com) instance with `nomic-embed-text` and `qwen2.5:7b-instruct` pulled.

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b-instruct
python3 test_suite.py
```

Exit code is 0 only on 12/12. Swap `CORPUS_DIR`, the case-reference regex, and the two model names in `rag_core.py` for your own setup -- the tenant-filter logic itself doesn't depend on any of them.

## What this does NOT prove

- It proves the **method** works against a small, fictional, single-machine corpus (3 tenants, 5 documents). It does not prove that any specific production system -- including mine -- is secure. Different vector databases, different chunking, different metadata schemes can all reintroduce the same class of bug in a different shape.
- 12 test cases is a small sample. This is a methodology demonstration, not a statistical security claim.
- No adversarial testing beyond the 5 trap questions here -- no prompt injection, no jailbreak attempts, no testing of what happens if a chunk's own text tries to override the system instruction.
- No concurrency/race-condition testing (many simultaneous sessions, index updates while queries run).
- No test of whether sensitive content leaks through the embedding model itself or through caching layers -- this only tests the retrieval-and-filter step.
- This was built and tested by one person on their own implementation. It is not an independent third-party audit.

If you're evaluating a vendor's or your own multi-tenant RAG system: this repo is a starting methodology, not a compliance checklist.

## License

MIT, see `LICENSE`. Copyright Määäx.

## Context

Built while working on a self-hosted, privacy-first AI hosting offering for small businesses and law/tax/medical practices in Germany, where "your documents never leave your own infrastructure" is the actual product, not a footnote -- [maaax.me](https://maaax.me).
