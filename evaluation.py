from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import EVAL_DIR, Settings

# Prefer the pipeline file used in the latest run; fall back to the conventional name.
from rag_pipeline import INSUFFICIENT, RAGPipeline, content_words, strip_citations


EVAL_SET = "eval_set_example.json"
RELEVANCE_FLOOR = 0.30
FAITHFULNESS_FLOOR = 0.60


# ============================================================
# 1. HELPERS AND METRICS
# ============================================================

def path_name(path: str) -> str:
    return str(path).replace("\\", "/").split("/")[-1].lower()


def chunk_is_relevant(chunk: dict, item: dict) -> bool:
    wanted_sources = {path_name(s) for s in item.get("relevant_sources", [])}
    if wanted_sources and path_name(chunk["source"]) not in wanted_sources:
        return False
    keywords = item.get("relevant_keywords") or []
    if not keywords:
        return bool(wanted_sources) and path_name(chunk["source"]) in wanted_sources
    text = chunk["text"].lower()
    return any(str(k).lower() in text for k in keywords)


def retrieval_metrics(chunks: list[dict], item: dict) -> dict:
    flags = [chunk_is_relevant(c, item) for c in chunks]
    first = next((i for i, flag in enumerate(flags, start=1) if flag), None)
    wanted = {path_name(s) for s in item.get("relevant_sources", [])}
    return {
        "hit": bool(first),
        "reciprocal_rank": 1.0 / first if first else 0.0,
        "precision": sum(flags) / len(flags) if flags else 0.0,
        "source_ok": bool(chunks) and (
            not wanted or path_name(chunks[0]["source"]) in wanted
        ),
        "n_retrieved": len(chunks),
    }


def answer_relevance(pipe: RAGPipeline, question: str, answer: str) -> float:
    text = strip_citations(answer)
    if not text:
        return 0.0
    q_vec, answer_vec = pipe.embed([question, text])
    return float(q_vec @ answer_vec)


def keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    lower = answer.lower()
    return sum(str(k).lower() in lower for k in keywords) / len(keywords)


def percentile(values: list[float], p: float) -> float:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def mean(values: list[float | int | bool | None]) -> float | None:
    usable = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(usable) if usable else None


def serializable_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "source": c.get("source"),
            "page": c.get("page"),
            "distance": round(float(c.get("distance", 0.0)), 4),
        }
        for c in chunks
    ]


def validate_eval_set(items: list[dict]) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError("Evaluation set must be a non-empty JSON list.")
    seen = set()
    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation item {i} must be a JSON object.")
        for key in ("id", "question", "answerable"):
            if key not in item:
                raise ValueError(f"Evaluation item {i} is missing '{key}'.")
        if item["id"] in seen:
            raise ValueError(f"Duplicate evaluation id: {item['id']}")
        seen.add(item["id"])
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Evaluation item {item['id']} has an empty question.")
        if not isinstance(item["answerable"], bool):
            raise ValueError(f"Evaluation item {item['id']} field 'answerable' must be true/false.")
        if item["answerable"] and not item.get("relevant_sources"):
            raise ValueError(
                f"Answerable item {item['id']} needs relevant_sources for retrieval scoring."
            )
        item.setdefault("relevant_keywords", [])
        item.setdefault("answer_keywords", [])


# ============================================================
# 2. RUN THE EVALUATION SET
# ============================================================

def evaluate_retrieval(pipe: RAGPipeline, items: list[dict], cfg: Settings) -> list[dict]:
    rows = []
    for item in items:
        retrieval = pipe.retrieve(item["question"], cfg)
        row = {
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "retrieved": serializable_chunks(retrieval.chunks),
            "retrieval_s": retrieval.seconds,
            "skipped": retrieval.skipped,
            "semantic_query": retrieval.semantic_query,
        }
        if item["answerable"]:
            row.update(retrieval_metrics(retrieval.chunks, item))
        else:
            row["correct_abstention"] = len(retrieval.chunks) == 0
            row["n_retrieved"] = len(retrieval.chunks)
        rows.append(row)
    return rows


def summarize_retrieval(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    return {
        "hit_rate": mean([r["hit"] for r in answerable]),
        "mrr": mean([r["reciprocal_rank"] for r in answerable]),
        "precision": mean([r["precision"] for r in answerable]),
        "source_accuracy": mean([r["source_ok"] for r in answerable]),
        "avg_chunks": mean([r["n_retrieved"] for r in answerable]),
        "no_result_rate": mean([r["n_retrieved"] == 0 for r in answerable]),
        "correct_abstention": mean([r["correct_abstention"] for r in unanswerable]),
        "retrieval_p50_ms": percentile([r["retrieval_s"] * 1000 for r in rows], 0.50),
        "retrieval_p95_ms": percentile([r["retrieval_s"] * 1000 for r in rows], 0.95),
    }


def evaluate_generation(pipe: RAGPipeline, items: list[dict]) -> list[dict]:
    rows = []
    for n, item in enumerate(items, start=1):
        print(f"  [{n}/{len(items)}] {item['question'][:70]}")
        result = pipe.answer(item["question"])
        row = {
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "answer": result.answer,
            "refused": result.refused,
            "retrieved": serializable_chunks(result.chunks),
            "timings": result.timings,
            "faithfulness": result.grounding.get("faithfulness"),
            "citation_coverage": result.grounding.get("citation_coverage"),
            "citation_precision": result.grounding.get("citation_precision"),
            "unsupported": [
                r["sentence"] for r in result.grounding.get("rows", [])
                if r.get("support") is not None
                and r["support"] < pipe.cfg.support_threshold
            ],
        }
        if item["answerable"]:
            row.update(retrieval_metrics(result.chunks, item))
            row["relevance"] = answer_relevance(pipe, item["question"], result.answer) if not result.refused else None
            row["keyword_coverage"] = (
                keyword_coverage(result.answer, item.get("answer_keywords", []))
                if not result.refused else None
            )
        else:
            row["correct_abstention"] = bool(result.refused)
            row["n_retrieved"] = len(result.chunks)
        rows.append(row)
    return rows


def summarize_generation(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["answerable"] and not r["refused"]]
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    live = [r for r in rows if r.get("timings")]

    def col(key: str, source: list[dict] | None = None) -> list[float]:
        source = live if source is None else source
        return [
            float(r["timings"][key]) for r in source
            if key in r.get("timings", {}) and r["timings"][key] is not None
        ]

    return {
        "answer_relevance": mean([r.get("relevance") for r in answered]),
        "keyword_coverage": mean([r.get("keyword_coverage") for r in answered]),
        "faithfulness": mean([r.get("faithfulness") for r in answered]),
        "citation_coverage": mean([r.get("citation_coverage") for r in answered]),
        "citation_precision": mean([r.get("citation_precision") for r in answered]),
        "wrongful_refusal_rate": mean([r.get("refused", False) for r in answerable]),
        "correct_abstention": mean([r.get("correct_abstention") for r in unanswerable]),
        "first_token_p50_s": percentile(col("first_token_s"), 0.50),
        "first_token_p95_s": percentile(col("first_token_s"), 0.95),
        "total_p50_s": percentile(col("total_s"), 0.50),
        "total_p95_s": percentile(col("total_s"), 0.95),
        "avg_prompt_tokens": mean(col("prompt_tokens")),
        "avg_output_tokens": mean(col("output_tokens")),
        "tokens_per_second": mean([
            r["timings"]["output_tokens"] / r["timings"]["generation_s"]
            for r in live
            if r["timings"].get("generation_s")
            and r["timings"].get("output_tokens") is not None
        ]),
    }


# ============================================================
# 3. FAILURE ANALYSIS
# ============================================================

FIXES = {
    "retrieval_miss": "Relevant chunk not in top-k. Inspect labels; increase candidates, expand the query, or test another embedding model.",
    "no_retrieval": "No chunks passed retrieval. Inspect distances and thresholds before changing them.",
    "wrong_refusal": "Relevant evidence was retrieved but the model refused. Check passage quality and the prompt.",
    "answered_out_of_scope": "An unanswerable question received an answer. Review the evidence threshold and abstention rule.",
    "unfaithful": "Answer sentences have weak semantic support. Inspect them manually; embedding similarity is only a heuristic.",
    "off_topic": "Question-answer embedding similarity is low. Inspect retrieved passages and the generated answer.",
    "uncited": "Answer has no valid [S#] citations. Improve citation instructions and validate citations against passages.",
    "bad_citation": "Cited passages have weak semantic support for the claims. Review citation-to-sentence links.",
}


def classify_failures(rows: list[dict]) -> list[dict]:
    failures = []
    for r in rows:
        kinds = []
        if r["answerable"]:
            if r.get("n_retrieved", 0) == 0:
                kinds.append("no_retrieval")
            elif not r.get("hit", False):
                kinds.append("retrieval_miss")
            if r.get("refused") and r.get("n_retrieved", 0) > 0 and r.get("hit"):
                kinds.append("wrong_refusal")
            if "relevance" in r and r.get("relevance") is not None:
                if r["relevance"] < RELEVANCE_FLOOR:
                    kinds.append("off_topic")
                faith = r.get("faithfulness")
                if faith is not None and faith < FAITHFULNESS_FLOOR:
                    kinds.append("unfaithful")
                coverage = r.get("citation_coverage")
                if coverage == 0:
                    kinds.append("uncited")
                precision = r.get("citation_precision")
                if precision is not None and precision < 0.5:
                    kinds.append("bad_citation")
        elif "refused" in r and not r["refused"]:
            kinds.append("answered_out_of_scope")
        elif "correct_abstention" in r and not r["correct_abstention"]:
            kinds.append("answered_out_of_scope")
        for kind in kinds:
            failures.append({
                "id": r["id"],
                "question": r["question"],
                "type": kind,
                "fix": FIXES[kind],
                "retrieved": r.get("retrieved", []),
                "answer": r.get("answer", ""),
            })
    return failures


# ============================================================
# 4. RETRIEVAL CONFIG SWEEP
# ============================================================

def sweep(pipe: RAGPipeline, items: list[dict]) -> list[dict]:
    base = pipe.cfg
    grid = itertools.product(
        [3, 4, 5],
        [0.45, 0.55, 0.65],
        [0.0, 0.15, 0.30],
        [0.10, 0.15, 1.0],
    )
    results = []
    for index, (top_k, max_d, lex, margin) in enumerate(grid, start=1):
        cfg = replace(
            base, top_k=top_k, max_distance=max_d,
            lexical_weight=lex, relative_margin=margin,
        )
        summary = summarize_retrieval(evaluate_retrieval(pipe, items, cfg))
        score = (
            0.35 * (summary["hit_rate"] or 0)
            + 0.25 * (summary["mrr"] or 0)
            + 0.20 * (summary["precision"] or 0)
            + 0.20 * (summary["correct_abstention"] or 0)
        )
        results.append({
            "top_k": top_k, "max_distance": max_d,
            "lexical_weight": lex, "relative_margin": margin,
            "score": score, **summary,
        })
        if index % 10 == 0:
            print(f"  Sweep configurations evaluated: {index}/81")
    return sorted(results, key=lambda r: r["score"], reverse=True)


# ============================================================
# 5. REPORT
# ============================================================

def fmt(value, pct: bool = False, digits: int = 3, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if pct:
        return f"{value:.0%}"
    return f"{value:.{digits}f}{unit}"


def write_report(path: Path, stamp: str, cfg: Settings, retr: dict,
                 gen: dict | None, failures: list[dict],
                 sweep_rows: list[dict] | None, n_items: int) -> None:
    lines = [
        "# RAG evaluation report", "",
        f"Run: {stamp} | questions: {n_items} | model: {cfg.llm_model} | "
        f"top_k={cfg.top_k}, max_distance={cfg.max_distance}", "",
        "## 1. Retrieval", "", "| Metric | Value |", "|---|---|",
        f"| Hit@k (any relevant chunk) | {fmt(retr['hit_rate'], pct=True)} |",
        f"| MRR | {fmt(retr['mrr'])} |",
        f"| Precision@k | {fmt(retr['precision'], pct=True)} |",
        f"| Top-1 from a relevant source | {fmt(retr['source_accuracy'], pct=True)} |",
        f"| Answerable questions with zero chunks | {fmt(retr['no_result_rate'], pct=True)} |",
        f"| Correct retrieval abstention (unanswerable) | {fmt(retr['correct_abstention'], pct=True)} |",
        f"| Retrieval latency p50 / p95 | {fmt(retr['retrieval_p50_ms'], digits=0)} / {fmt(retr['retrieval_p95_ms'], digits=0)} ms |",
        "",
    ]
    if gen is not None:
        lines += [
            "## 2. Answer quality", "", "| Metric | Value |", "|---|---|",
            f"| Answer relevance (question-answer embedding similarity) | {fmt(gen['answer_relevance'])} |",
            f"| Expected-keyword coverage | {fmt(gen['keyword_coverage'], pct=True)} |",
            f"| Wrongful refusals on answerable questions | {fmt(gen['wrongful_refusal_rate'], pct=True)} |",
            f"| Correct end-to-end abstention | {fmt(gen['correct_abstention'], pct=True)} |", "",
            "## 3. Grounding and citations", "", "| Metric | Value |", "|---|---|",
            f"| Faithfulness heuristic | {fmt(gen['faithfulness'], pct=True)} |",
            f"| Citation coverage | {fmt(gen['citation_coverage'], pct=True)} |",
            f"| Citation precision heuristic | {fmt(gen['citation_precision'], pct=True)} |", "",
            "## 4. Latency", "", "| Metric | Value |", "|---|---|",
            f"| Time to first token p50 / p95 | {fmt(gen['first_token_p50_s'], digits=1)} / {fmt(gen['first_token_p95_s'], digits=1)} s |",
            f"| Total time p50 / p95 | {fmt(gen['total_p50_s'], digits=1)} / {fmt(gen['total_p95_s'], digits=1)} s |",
            f"| Average prompt / output tokens | {fmt(gen['avg_prompt_tokens'], digits=0)} / {fmt(gen['avg_output_tokens'], digits=0)} |",
            f"| Generation speed | {fmt(gen['tokens_per_second'], digits=1)} tokens/s |", "",
        ]
    else:
        lines += ["_Generation evaluation skipped; run without --retrieval-only for answer and latency metrics._", ""]
    lines += ["## 5. Failure cases", ""]
    if not failures:
        lines.append("No failures detected by the configured heuristic checks.")
    for failure in failures:
        lines += [
            f"- **{failure['id']}** `{failure['type']}`: {failure['question']}",
            f"  - Retrieved: {failure['retrieved']}",
            f"  - Suggested investigation: {failure['fix']}",
        ]
    lines.append("")
    if sweep_rows:
        lines += [
            f"## 6. Retrieval tuning sweep (top 8 of {len(sweep_rows)} configurations)", "",
            "| top_k | max_dist | lexical | margin | Hit | MRR | Prec | Abstain | score |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sweep_rows[:8]:
            lines.append(
                f"| {r['top_k']} | {r['max_distance']} | {r['lexical_weight']} | "
                f"{r['relative_margin']} | {fmt(r['hit_rate'], pct=True)} | "
                f"{fmt(r['mrr'], digits=2)} | {fmt(r['precision'], pct=True)} | "
                f"{fmt(r['correct_abstention'], pct=True)} | {r['score']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Interpretation notes", "",
        "- Scores depend on the quality of the manually labelled evaluation set.",
        "- Keyword coverage and embedding similarity are proxies, not proof of correctness.",
        "- Review low-scoring cases and citations manually before changing production settings.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 6. MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Research Intelligence Assistant")
    parser.add_argument("--eval-set", default=EVAL_SET, help="JSON evaluation-set path")
    parser.add_argument("--retrieval-only", action="store_true", help="skip LLM generation")
    parser.add_argument("--sweep", action="store_true", help="test 81 retrieval configurations")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    args = parser.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        raise SystemExit(
            f"Evaluation set not found: {eval_path.resolve()}\n"
            "Create eval_set.json using the example format supplied with this script."
        )
    try:
        items = json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {eval_path}: {exc}") from exc
    validate_eval_set(items)
    if args.limit > 0:
        items = items[:args.limit]
    if not items:
        raise SystemExit("No evaluation questions selected.")

    pipe = RAGPipeline()
    if not pipe.has_index():
        raise SystemExit("No index found. Run your ingestion/index setup first; do not rebuild unless needed.")

    available = {path_name(s) for s in pipe.indexed_sources()}
    for item in items:
        for source in item.get("relevant_sources", []):
            if path_name(source) not in available:
                print(f"Warning: {item['id']} expects {source}, which is not indexed.")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Retrieval evaluation on {len(items)} questions...")
    retrieval_rows = evaluate_retrieval(pipe, items, pipe.cfg)
    retrieval_summary = summarize_retrieval(retrieval_rows)

    rows = retrieval_rows
    generation_summary = None
    if not args.retrieval_only:
        print("Generation evaluation (this calls the local LLM and can take time)...")
        pipe.warm_up()
        rows = evaluate_generation(pipe, items)
        generation_summary = summarize_generation(rows)

    failures = classify_failures(rows)
    sweep_rows = sweep(pipe, items) if args.sweep else None

    report_path = EVAL_DIR / f"report_{stamp}.md"
    json_path = EVAL_DIR / f"results_{stamp}.json"
    write_report(report_path, stamp, pipe.cfg, retrieval_summary,
                 generation_summary, failures, sweep_rows, len(items))
    json_path.write_text(
        json.dumps({
            "retrieval": retrieval_summary,
            "generation": generation_summary,
            "rows": rows,
            "failures": failures,
            "sweep": sweep_rows,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n" + report_path.read_text(encoding="utf-8"))
    print(f"\nSaved report: {report_path}")
    print(f"Saved detailed results: {json_path}")


if __name__ == "__main__":
    main()
