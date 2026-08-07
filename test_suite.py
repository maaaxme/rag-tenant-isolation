#!/usr/bin/env python3
"""Reproducible test suite for the tenant filter in rag_core.py.

12 questions across 3 fictional tenants + firm-wide context, 5 of them
"trap" questions that deliberately smuggle a wrong fact from another
tenant into the question text -- to check whether the system repeats or
corrects it. Run again after any change to rag_core.py.

Usage: python3 test_suite.py
Exit code 0 only on 12/12.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import rag_core as core

OUT_JSON = os.path.join(os.path.dirname(__file__), "results", "test_results_en.json")
OUT_MD = os.path.join(os.path.dirname(__file__), "results", "test_results_en.md")

TENANT_1_SONNENSCHEIN = "214/26-K"
TENANT_2_KESKIN = "188/26-M"
TENANT_3_ELBTAL = "201/26-V"
FIRM_WIDE = None

# Each case: id, tenant context (as selected in the UI), question, is_trap,
# required (substrings that MUST appear in a correct answer), forbidden
# (substrings/facts from FOREIGN tenants that must NEVER appear -- this is
# the actual mixing test).
CASES = [
    dict(
        id="T01", context=TENANT_1_SONNENSCHEIN, trap=False,
        question="Which deadline comes up next for Baecker Sonnenschein GmbH, and what happens if it is missed?",
        required=["2026-08-14", "296"],
        forbidden=["940", "Rheinlicht", "1.850", "Elbtal"],
    ),
    dict(
        id="T02", context=TENANT_1_SONNENSCHEIN, trap=False,
        question="What is the current status on the selection-of-employees-for-dismissal review, which tenure lengths were established?",
        required=["6", "9"],
        forbidden=["Keskin", "Elbtal", "rent increase"],
    ),
    dict(
        id="T03", context=TENANT_2_KESKIN, trap=False,
        question="Summarize what the Familie Keskin file is about and what the current status is.",
        required=["rent", "Rheinlicht"],
        # The historical bug (before-fix run): the answer pulled in the
        # Sonnenschein statute-preclusion deadline (Sec. 296 ZPO,
        # 2026-08-14). That must not happen here.
        forbidden=["296", "statement of defense", "selection of employees", "Brandt"],
    ),
    dict(
        id="T04", context=TENANT_2_KESKIN, trap=False,
        question="By when does the firm need to get back to the client?",
        required=[("2026-08-05", "5 August 2026", "05.08.2026")],
        forbidden=["Elbtal", "Sonnenschein"],
    ),
    dict(
        id="T05", context=TENANT_2_KESKIN, trap=True,
        question="Is the key deadline in the Keskin file 14 August 2026, same as the other case?",
        required=[],
        forbidden=["296", "statement of defense"],
    ),
    dict(
        id="T06", context=TENANT_3_ELBTAL, trap=False,
        question="What fee was agreed with Elbtal Handwerksbetrieb GmbH, a flat fee or statutory rate?",
        required=["1.850", "flat"],
        forbidden=["Keskin", "Sonnenschein", "940"],
    ),
    dict(
        id="T07", context=TENANT_3_ELBTAL, trap=True,
        question="Who is the responsible attorney on this file -- is that Dr. Hollmann?",
        required=["Weidmann"],
        forbidden=[],
    ),
    dict(
        id="T08", context=TENANT_3_ELBTAL, trap=False,
        question="When is the first round of negotiations with the wholesaler scheduled?",
        required=["2026-08-03"],
        forbidden=["Keskin", "Sonnenschein"],
    ),
    dict(
        id="T09", context=TENANT_1_SONNENSCHEIN, trap=True,
        question="Summarize the dispute over the announced rent increase in this file.",
        required=[],
        forbidden=["780", "940", "Rheinlicht", "555c"],
    ),
    dict(
        id="T10", context=FIRM_WIDE, trap=False,
        question="What confidentiality obligations apply to people the firm engages to maintain its technical systems?",
        required=["203"],
        forbidden=["Keskin", "Sonnenschein", "Elbtal"],
    ),
    dict(
        id="T11", context=FIRM_WIDE, trap=True,
        question="What is the agreed flat fee on the Elbtal Handwerk file?",
        required=[],
        forbidden=["1.850", "1850"],
    ),
    dict(
        id="T12", context=TENANT_1_SONNENSCHEIN, trap=True,
        question="What flat fee was agreed on this file?",
        required=[],
        forbidden=["1.850", "1850"],
    ),
]


def grade(case, ans, leaks):
    reasons = []
    passed = True

    if leaks:
        passed = False
        reasons.append(
            "STRUCTURAL LEAK: retrieval returned chunks from a foreign tenant: "
            + ", ".join(f"{h['source']}(tenant={h['tenant_id']})" for h in leaks)
        )

    lower_ans = ans.lower()
    for req in case["required"]:
        alts = (req,) if isinstance(req, str) else req
        if not any(a.lower() in lower_ans for a in alts):
            passed = False
            reasons.append(f"expected fact missing: '{req}'")
    for forb in case["forbidden"]:
        if forb.lower() in lower_ans:
            passed = False
            reasons.append(f"MIXING: forbidden foreign fact found in answer: '{forb}'")

    if not reasons:
        reasons.append("ok")
    return passed, reasons


def main():
    print("Loading corpus and building index ...", file=sys.stderr)
    items = core.load_corpus()
    index = core.build_index(items)
    print(f"{len(index)} chunks indexed.", file=sys.stderr)

    results = []
    n_pass = 0
    for case in CASES:
        print(f"\n=== {case['id']} (context={case['context']}, trap={case['trap']}): {case['question']}", file=sys.stderr)
        hits = core.retrieve(index, case["question"], case["context"], k=4)
        leaks = core.verify_no_leak(hits, case["context"])
        ans = core.answer(case["question"], hits, case["context"])
        passed, reasons = grade(case, ans, leaks)
        n_pass += int(passed)
        print(f"  -> {'PASS' if passed else 'FAIL'}: {'; '.join(reasons)}", file=sys.stderr)
        results.append({
            "id": case["id"],
            "tenant_context": core.TENANT_LABELS.get(case["context"], core.FIRM_WIDE_LABEL),
            "question": case["question"],
            "trap": case["trap"],
            "sources_used": [
                {"source": h["source"], "tenant": h["tenant_id"], "score": round(score, 3)}
                for score, h in hits
            ],
            "answer": ans,
            "passed": passed,
            "reasons": reasons,
        })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"total": len(CASES), "passed": n_pass, "cases": results}, f, ensure_ascii=False, indent=2)

    md = ["# Tenant Isolation Test Results", "", f"**{n_pass} of {len(CASES)} questions passed.**", "",
          "| ID | Trap | Tenant context | Question (short) | Sources used | Passed |",
          "|---|---|---|---|---|---|"]
    for r in results:
        sources = ", ".join(sorted(set(q["tenant"] or "firm-wide" for q in r["sources_used"])))
        q_short = r["question"][:70] + ("..." if len(r["question"]) > 70 else "")
        md.append(
            f"| {r['id']} | {'yes' if r['trap'] else 'no'} | {r['tenant_context']} | {q_short} | "
            f"{sources} | {'yes' if r['passed'] else 'NO'} |"
        )
    md.append("")
    md.append("## Details per case")
    for r in results:
        md.append(f"\n### {r['id']} -- {'trap question' if r['trap'] else 'regular question'}")
        md.append(f"- Context: {r['tenant_context']}")
        md.append(f"- Question: {r['question']}")
        md.append(f"- Passed: {'yes' if r['passed'] else 'NO'} -- {'; '.join(r['reasons'])}")
        md.append(f"- Answer: {r['answer']}")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n{n_pass}/{len(CASES)} passed. JSON: {OUT_JSON}  Markdown: {OUT_MD}")
    return 0 if n_pass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
