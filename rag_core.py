#!/usr/bin/env python3
"""RAG core with a hard tenant filter applied BEFORE similarity ranking.

This is the fixed version. See before/rag_core_buggy.py for the version that
leaked context across tenants, and README.md for the full writeup.

How it works:
Every chunk is tagged with a tenant_id at index time, extracted from a case
reference number found in the source document (regex on the German "Az. ..."
convention used by the fictional test corpus -- swap this for whatever
tenant identifier convention your own documents use). Documents without a
case reference are treated as firm-wide / tenant-independent.

retrieve() always requires a tenant context (in a real application this
comes from the UI/session -- the case file a user currently has open, not
from the free-text query) and removes every chunk that belongs to a
different tenant BEFORE the similarity search runs. That is a structural
exclusion, not a relevance threshold applied after the fact -- it holds even
when the query itself references a fact from another tenant (see the trap
questions in test_suite.py).
"""
import json
import os
import re
import sys
import urllib.request

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b-instruct"
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

CASE_REF_PATTERN = re.compile(r"Az\.\s*(\S+)")

TENANT_LABELS = {
    "214/26-K": "Tenant 1 -- Baecker Sonnenschein GmbH (employment dispute)",
    "188/26-M": "Tenant 2 -- Familie Keskin (tenancy dispute)",
    "201/26-V": "Tenant 3 -- Elbtal Handwerksbetrieb GmbH (contract dispute)",
}
FIRM_WIDE_LABEL = "firm-wide (no single tenant)"


def ollama_post(path, payload, timeout=180):
    req = urllib.request.Request(
        OLLAMA + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed(text):
    r = ollama_post("/api/embed", {"model": EMBED_MODEL, "input": text})
    return r["embeddings"][0]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def chunk_text(text, max_chars=700):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > max_chars and buf:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = (buf + "\n\n" + p) if buf else p
    if buf:
        chunks.append(buf.strip())
    return chunks


def extract_tenant_id(full_text):
    m = CASE_REF_PATTERN.search(full_text)
    if not m:
        return None
    return m.group(1).rstrip(",.")


def load_corpus():
    items = []
    for fname in sorted(os.listdir(CORPUS_DIR)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(CORPUS_DIR, fname), encoding="utf-8") as f:
            text = f.read()
        tenant_id = extract_tenant_id(text)
        for i, ch in enumerate(chunk_text(text)):
            items.append({
                "source": fname,
                "chunk_id": i,
                "text": ch,
                "tenant_id": tenant_id,
            })
    return items


def build_index(items):
    for it in items:
        it["embedding"] = embed(it["text"])
    return items


def retrieve(index, query, context_tenant, k=4):
    """context_tenant: the case reference of the tenant currently being worked
    on (comes from the UI/session selection, not from the query itself) --
    or None for a firm-wide context with no tenant scope. Chunks belonging to
    ANY OTHER tenant are removed before the similarity search even runs."""
    candidates = [it for it in index if it["tenant_id"] in (context_tenant, None)]
    qv = embed(query)
    scored = [(cosine(qv, it["embedding"]), it) for it in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def verify_no_leak(hits, context_tenant):
    """Structural check: no hit may come from a foreign tenant."""
    leaks = [h for _, h in hits if h["tenant_id"] not in (context_tenant, None)]
    return leaks


def answer(query, hits, context_tenant):
    context = "\n\n".join(
        f"[Source: {h['source']} #{h['chunk_id']} | Tenant {h['tenant_id'] or 'firm-wide'} | Score {score:.2f}]\n{h['text']}"
        for score, h in hits
    )
    tenant_desc = TENANT_LABELS.get(context_tenant, context_tenant) if context_tenant else FIRM_WIDE_LABEL
    system = (
        f"You are an internal assistant. You work EXCLUSIVELY on the file "
        f"'{tenant_desc}'. Answer only from the given context. Never mix in "
        "information from other tenants, even if the question implies it or "
        "names another tenant -- if a needed fact is not in the given "
        "context, say so explicitly ('not present in this file') and do not "
        "invent it. List the source files you used at the end."
    )
    user = f"Context (only from tenant '{tenant_desc}' and firm-wide documents):\n{context}\n\nQuestion: {query}"
    r = ollama_post(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=180,
    )
    return r["message"]["content"]


if __name__ == "__main__":
    print(f"Loading corpus from {CORPUS_DIR} ...", file=sys.stderr)
    items = load_corpus()
    by_tenant = {}
    for it in items:
        by_tenant.setdefault(it["tenant_id"], 0)
        by_tenant[it["tenant_id"]] += 1
    print(f"{len(items)} chunks, tenant distribution: {by_tenant}", file=sys.stderr)
    print("Embedding chunks ...", file=sys.stderr)
    index = build_index(items)
    print("Index ready. Use test_suite.py to run the test.", file=sys.stderr)
