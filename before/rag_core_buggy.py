#!/usr/bin/env python3
"""The buggy version -- kept for reference only, not used by test_suite.py.

Retrieval runs a flat top-k similarity search over chunks from ALL tenants,
with no tenant boundary at all. See buggy_run_output.json for a real captured
run of this exact code: a question about Tenant 2 (Keskin, tenancy dispute)
pulled a deadline and a statute reference that belong to Tenant 1
(Sonnenschein, employment dispute) straight into the answer. The model did
not hallucinate the fact -- it was handed a foreign-tenant chunk in its
context window and used it as instructed, faithfully.

Do not run this against real data. It is here to show what "RAG without an
access boundary" actually produces, not as usable code.
"""
import json
import os
import re
import sys
import urllib.request

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b-instruct"
CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus")


def ollama_post(path, payload, timeout=120):
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


def load_corpus():
    # NOTE: no tenant tagging at all -- this is the bug.
    items = []
    for fname in sorted(os.listdir(CORPUS_DIR)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(CORPUS_DIR, fname), encoding="utf-8") as f:
            text = f.read()
        for i, ch in enumerate(chunk_text(text)):
            items.append({"source": fname, "chunk_id": i, "text": ch})
    return items


def build_index(items):
    for it in items:
        it["embedding"] = embed(it["text"])
    return items


def retrieve(index, query, k=4):
    # NOTE: no tenant context parameter -- searches the entire corpus.
    qv = embed(query)
    scored = [(cosine(qv, it["embedding"]), it) for it in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def answer(query, hits):
    context = "\n\n".join(
        f"[Source: {h['source']} #{h['chunk_id']} | Score {score:.2f}]\n{h['text']}"
        for score, h in hits
    )
    system = (
        "You are an internal assistant. Answer exclusively from the given "
        "context. List the source files you used at the end. If the answer "
        "is not in the context, say so and do not invent it."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"
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


QUESTIONS = [
    "Which deadline comes up next for Baecker Sonnenschein GmbH, and what "
    "happens if it is missed?",
    "Summarize what the Familie Keskin file is about and what the current "
    "status is.",
    "What fee was agreed with Elbtal Handwerksbetrieb GmbH, a flat fee or "
    "statutory rate?",
]


def main():
    print(f"Loading corpus from {CORPUS_DIR} ...", file=sys.stderr)
    items = load_corpus()
    print(f"{len(items)} chunks from {len(set(i['source'] for i in items))} files.", file=sys.stderr)
    print("Embedding chunks ...", file=sys.stderr)
    index = build_index(items)

    results = []
    for q in QUESTIONS:
        print(f"\n=== Question: {q}", file=sys.stderr)
        hits = retrieve(index, q, k=4)
        for score, h in hits:
            print(f"  hit {score:.3f}  {h['source']} #{h['chunk_id']}", file=sys.stderr)
        a = answer(q, hits)
        results.append({
            "question": q,
            "answer": a,
            "sources": [{"source": h["source"], "score": round(score, 3)} for score, h in hits],
        })

    out_path = os.path.join(os.path.dirname(__file__), "buggy_run_output_en.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"embed_model": EMBED_MODEL, "chat_model": CHAT_MODEL, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
