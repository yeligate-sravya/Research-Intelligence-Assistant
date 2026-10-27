from __future__ import annotations
 
import argparse
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator
 
import numpy as np
 
from config import Settings
 
INSUFFICIENT = (
    "The retrieved passages do not contain enough information "
    "to answer this reliably."
)
 
# ============================================================
# 1. TEXT HELPERS (pure functions, no models needed)
# ============================================================
 
REFERENCE_MARKERS = re.compile(
    r"\barxiv\b|\bdoi\b|\bcrossref\b|\bpreprint\b|"
    r"\bproceedings\b|\bvol\.\b|\bpp\.\b|\bet al\.",
    re.IGNORECASE,
)
CITATION_GROUP = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]")
NUMERIC_CITATION = re.compile(r"(?<!\w)\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")
# Common hallucinated single-letter/short citation markers such as [M].
UNKNOWN_CITATION = re.compile(r"\[\s*[A-Z]{1,3}\s*\]")
TRAILING_CITATIONS = re.compile(r"([.!?])[ \t]*((?:\[S\d+(?:, S\d+)*\][ \t]*)+)")
ABSTAIN = re.compile(
    r"\b(?:not|no|cannot)\b.{0,40}\b(?:stated|mentioned|specified|provided|"
    r"contain|explicit|explicitly)\b",
    re.IGNORECASE,
)
 
FILE_REF = re.compile(r"[\w.-]+\.pdf", re.IGNORECASE)
PAPER_REF = re.compile(r"\bresearch[\s_-]*paper[\s_-]*\d+\b", re.IGNORECASE)
PAGE_REF = re.compile(r"\bpage\s+\d+\b", re.IGNORECASE)
POLITE = re.compile(r"\b(?:please|tell me|explain)\b", re.IGNORECASE)
# "what does <the paper> say about ..." carries no topical signal.
SAY_ABOUT = re.compile(
    r"\b(?:what|which)\s+(?:does|do|did)\s+(?:(?:it|this|that|the)\s+)?"
    r"(?:(?:paper|document|pdf|article|study|review|survey)\s+)?"
    r"(?:say|says|said|mention|mentions|discuss|discusses|state|states)\b"
    r"(?:\s+(?:about|on|regarding))?",
    re.IGNORECASE,
)
DANGLING = re.compile(r"\b(?:in|from|of|on|at)\s*$", re.IGNORECASE)
 
STOPWORDS = {
    "the", "and", "for", "are", "was", "how", "what", "does", "did", "its",
    "can", "has", "not", "but", "you", "any", "all", "who", "why",
    "that", "this", "with", "from", "have", "were", "which", "their",
    "there", "these", "those", "been", "also", "such", "using", "used",
    "into", "more", "than", "they", "will", "when", "while", "about",
    "other", "paper", "passage", "passages", "provided", "mentions",
    "states", "describes", "answer", "according", "source", "sources",
    "because", "through", "very", "some", "many", "each", "most",
}
 
METHODOLOGY_PHRASES = (
    "publication window", "peer-reviewed, english-language",
    "peer-reviewed english-language", "reported findings",
    "review methodology", "selection bias",
    "generalizability of the results", "completeness of the review",
)
REVIEW_LIMIT_PHRASES = (
    "limitations of this review", "limitations of the review",
    "limitations of this study", "limitations of the present study",
    "limitations of our review", "limitations of the systematic review",
)
ASKS_ABOUT_REVIEW = (
    "limitations of the review", "limitations of this review",
    "limitations of the paper", "limitations of this paper",
    "limitations of the study",
)
 
 
def make_semantic_query(question: str) -> str:
    """Text to embed: the question minus filename/page/filler instructions.
 
    Filters belong in metadata, not in the embedding. The LLM still sees
    the original question unchanged.
    """
    query = question.strip()
    for pattern in (FILE_REF, PAPER_REF, PAGE_REF, POLITE, SAY_ABOUT):
        query = pattern.sub(" ", query)
    query = re.sub(r"[^\w\s'-]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    query = DANGLING.sub("", query).strip()
    return query or question.strip()
 
 
def looks_like_reference(text: str) -> bool:
    markers = len(REFERENCE_MARKERS.findall(text))
    initials = len(re.findall(r"\b[A-Z]\.", text))
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    return markers >= 2 or (initials >= 6 and years >= 1)
 
 
def looks_like_review_methodology(text: str) -> bool:
    """Review-methodology caveats are not technical limits of RAG."""
    lower = text.lower()
    if any(p in lower for p in REVIEW_LIMIT_PHRASES):
        return True
    return any(p in lower for p in METHODOLOGY_PHRASES) and any(
        t in lower for t in ("review", "studies", "publication")
    )
 
 
def asks_technical_rag_limits(question: str) -> bool:
    q = question.lower()
    return (
        "rag" in q
        and "limitation" in q
        and not any(p in q for p in ASKS_ABOUT_REVIEW)
    )
 
 
def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS}
 
 
def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned = (p.strip(" -*•\t") for p in parts)
    return [p for p in cleaned if p]
 
 
def cited_labels(text: str) -> list[int]:
    numbers: set[int] = set()
    for group in CITATION_GROUP.findall(text):
        numbers.update(int(n) for n in re.findall(r"S(\d+)", group))
    return sorted(numbers)
 
 
def strip_citations(text: str) -> str:
    text = CITATION_GROUP.sub("", text)
    text = NUMERIC_CITATION.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
 
 
def clean_citations(answer: str, n_sources: int) -> str:
    """Keep valid [S#] labels, drop invented ones, remove the papers' own
    numeric citations ([14], [14-18]) and put labels before the full stop."""
    valid = {f"S{i}" for i in range(1, n_sources + 1)}
 
    def fix_group(match: re.Match) -> str:
        labels = [l for l in re.findall(r"S\d+", match.group(0)) if l in valid]
        labels = list(dict.fromkeys(labels))
        return "[" + ", ".join(labels) + "]" if labels else ""
 
    answer = CITATION_GROUP.sub(fix_group, answer)
    answer = NUMERIC_CITATION.sub("", answer)
    answer = UNKNOWN_CITATION.sub("", answer)
    answer = TRAILING_CITATIONS.sub(
        lambda m: " " + m.group(2).strip() + m.group(1) + " ", answer
    )
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\s+([,.;:!?])", r"\1", answer)
    return answer.strip()
 
 
# ============================================================
# 2. RESULT TYPES
# ============================================================
 
@dataclass
class Retrieval:
    question: str
    semantic_query: str
    source: str | None
    page: int | None
    chunks: list[dict]
    skipped: dict
    seconds: float
 
 
@dataclass
class Answer:
    question: str
    answer: str
    chunks: list[dict]
    sources_text: str
    warnings: list[str]
    refused: bool
    timings: dict
    grounding: dict
    retrieval: Retrieval | None = None
 
    @property
    def cited(self) -> list[int]:
        return [n for n in cited_labels(self.answer) if 1 <= n <= len(self.chunks)]
 
    def to_dict(self, include_text: bool = False) -> dict:
        cited = set(self.cited)
        sources = []
        for i, c in enumerate(self.chunks, start=1):
            item = {
                "label": f"S{i}",
                "source": c["source"],
                "page": c["page"],
                "similarity": round(1.0 - c["distance"], 3),
                "cited": i in cited,
            }
            if include_text:
                item["text"] = c["text"]
            sources.append(item)
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "sources": sources,
            "warnings": self.warnings,
            "grounding": {k: v for k, v in self.grounding.items() if k != "rows"},
            "timings": self.timings,
        }
 
 
# ============================================================
# 3. THE PIPELINE
# ============================================================
 
class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or Settings()
        self._embedder = None
        self._chroma = None
        self._llm = None
        self._sources: list[str] | None = None
        self._qvec_cache: dict[str, np.ndarray] = {}
        self._raw_cache: dict[tuple, list[dict]] = {}
 
    # ---------- lazy resources ----------
 
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
 
            self._embedder = SentenceTransformer(self.cfg.embedding_model)
        return self._embedder
 
    @property
    def chroma(self):
        if self._chroma is None:
            import chromadb
 
            self.cfg.db_path.mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=str(self.cfg.db_path))
        return self._chroma
 
    @property
    def llm(self):
        if self._llm is None:
            import ollama
 
            self._llm = ollama.Client(host=self.cfg.ollama_host)
        return self._llm
 
    def embed(self, texts) -> np.ndarray:
        vectors = self.embedder.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors)
 
    def embed_query(self, text: str) -> np.ndarray:
        """Query embeddings are cached; repeats and eval sweeps are free."""
        vec = self._qvec_cache.get(text)
        if vec is None:
            if len(self._qvec_cache) >= 512:
                self._qvec_cache.clear()
            vec = self.embed([text])[0]
            self._qvec_cache[text] = vec
        return vec
 
    # ---------- index ----------
 
    def load_chunks(self) -> list[dict]:
        path = self.cfg.chunks_path
        if not path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {path}\nRun python ingestion.py first."
            )
        chunks = json.loads(path.read_text(encoding="utf-8"))
        if not chunks:
            raise ValueError("No chunks found in chunks.json.")
        return chunks
 
    def build_index(self, chunks: list[dict] | None = None, verbose: bool = True):
        """Rebuild the collection from output/chunks.json."""
        chunks = chunks if chunks is not None else self.load_chunks()
 
        for item in chunks:
            if not all(k in item for k in ("id", "text", "metadata")):
                raise ValueError("Every chunk must contain id, text and metadata.")
            if not all(k in item["metadata"] for k in ("source", "page", "chunk_id")):
                raise ValueError(
                    "Chunk metadata must contain source, page and chunk_id."
                )
 
        try:
            self.chroma.delete_collection(name=self.cfg.collection_name)
        except Exception:
            pass
        self._sources = None
        self._raw_cache.clear()
 
        collection = self.chroma.create_collection(
            name=self.cfg.collection_name, metadata={"hnsw:space": "cosine"}
        )
 
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c["text"] for c in batch]
            collection.add(
                ids=[c["id"] for c in batch],
                documents=texts,
                metadatas=[c["metadata"] for c in batch],
                embeddings=self.embed(texts).tolist(),
            )
            if verbose:
                print(f"Indexed {min(start + batch_size, len(chunks))}/{len(chunks)}")
 
        if verbose:
            print("Index rebuilt. Chunks stored:", collection.count())
        return collection
 
    def collection(self):
        try:
            col = self.chroma.get_collection(name=self.cfg.collection_name)
            if col.count() > 0:
                return col
        except Exception:
            pass
        raise RuntimeError(
            "No indexed documents found. Run ingestion.py, then "
            "python rag_pipeline.py --rebuild."
        )
 
    def has_index(self) -> bool:
        try:
            self.collection()
            return True
        except RuntimeError:
            return False
 
    # ---------- metadata filters ----------
 
    def indexed_sources(self) -> list[str]:
        if self._sources is None:
            metas = self.collection().get(include=["metadatas"])["metadatas"]
            self._sources = sorted({m["source"] for m in metas if m and m.get("source")})
        return self._sources
 
    def parse_filters(self, question: str) -> tuple[str | None, int | None]:
        """Recognise 'X.pdf', 'research paper 2' and 'page 5' in a question."""
        by_name = {
            Path(name.replace("\\", "/")).name.lower(): name
            for name in self.indexed_sources()
        }
        source = None
 
        match = re.search(r"([\w.-]+\.pdf)\b", question, flags=re.IGNORECASE)
        if match:
            requested = match.group(1).lower()
            if requested not in by_name:
                raise ValueError(
                    f"'{requested}' was not found in the index. "
                    f"Available: {', '.join(sorted(by_name))}"
                )
            source = by_name[requested]
        else:
            match = PAPER_REF.search(question)
            if match:
                number = re.search(r"\d+", match.group(0)).group(0)
                requested = f"research_paper_{number}.pdf"
                source = by_name.get(requested)
                if source is None:
                    raise ValueError(f"{requested} was not found in the index.")
 
        page = None
        if source:
            match = re.search(r"\bpage\s+(\d+)\b", question, flags=re.IGNORECASE)
            if match:
                page = int(match.group(1))
        return source, page
 
    @staticmethod
    def _where(source, page):
        if source and page is not None:
            return {"$and": [{"source": source}, {"page": page}]}
        return {"source": source} if source else None
 
    # ---------- retrieval ----------
 
    def _candidates(self, query: str, n: int, source, page) -> list[dict]:
        """Vector search. Results are cached per (query, n, filters)."""
        key = (query, n, source, page)
        cached = self._raw_cache.get(key)
        if cached is None:
            col = self.collection()
            args = {
                "query_embeddings": [self.embed_query(query).tolist()],
                "n_results": min(n, col.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            where = self._where(source, page)
            if where:
                args["where"] = where
            res = col.query(**args)
            cached = [
                {
                    "text": text,
                    "source": meta["source"],
                    "page": meta["page"],
                    "chunk_id": meta["chunk_id"],
                    "distance": float(dist),
                }
                for text, meta, dist in zip(
                    res["documents"][0], res["metadatas"][0], res["distances"][0]
                )
            ]
            self._raw_cache[key] = cached
        return [dict(c) for c in cached]
 
    def retrieve(self, question: str, cfg: Settings | None = None) -> Retrieval:
        """Search, rerank, threshold and diversify. `cfg` overrides settings
        for one call (used by the evaluation sweep)."""
        cfg = cfg or self.cfg
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
 
        t0 = time.perf_counter()
        source, page = self.parse_filters(question)
        query = make_semantic_query(question)
        raw = self._candidates(query, cfg.candidates, source, page)
 
        # Hybrid rerank: semantic similarity + small keyword-overlap bonus.
        terms = content_words(query)
        for chunk in raw:
            overlap = len(terms & content_words(chunk["text"])) / len(terms) if terms else 0.0
            chunk["lexical"] = overlap
            chunk["score"] = (1.0 - chunk["distance"]) + cfg.lexical_weight * overlap
        raw.sort(key=lambda c: c["score"], reverse=True)
 
        # Gate: absolute distance limit + relative margin to the best hit.
        limit = cfg.filtered_max_distance if source else cfg.max_distance
        best = min((c["distance"] for c in raw), default=1.0)
        cutoff = min(limit, best + cfg.relative_margin)
 
        technical_limits = asks_technical_rag_limits(question)
        skipped = dict.fromkeys(
            ("too_far", "reference", "review_methodology", "duplicate", "per_source_cap"), 0
        )
        chosen, seen, per_source = [], set(), {}
 
        for chunk in raw:
            if chunk["distance"] > cutoff:
                skipped["too_far"] += 1
                continue
            if looks_like_reference(chunk["text"]):
                skipped["reference"] += 1
                continue
            if technical_limits and looks_like_review_methodology(chunk["text"]):
                skipped["review_methodology"] += 1
                continue
            key = re.sub(r"\s+", " ", chunk["text"]).strip().lower()
            if key in seen:
                skipped["duplicate"] += 1
                continue
            if not source and per_source.get(chunk["source"], 0) >= cfg.max_per_source:
                skipped["per_source_cap"] += 1
                continue
            seen.add(key)
            per_source[chunk["source"]] = per_source.get(chunk["source"], 0) + 1
            chosen.append(chunk)
            if len(chosen) >= cfg.top_k:
                break
 
        return Retrieval(
            question=question,
            semantic_query=query,
            source=source,
            page=page,
            chunks=chosen,
            skipped=skipped,
            seconds=time.perf_counter() - t0,
        )
 
    def explore(self, question: str, n: int = 10) -> list[dict]:
        """Raw matches with pass/reject status. Does not call the LLM."""
        source, page = self.parse_filters(question)
        raw = self._candidates(make_semantic_query(question), n, source, page)
        limit = self.cfg.filtered_max_distance if source else self.cfg.max_distance
        for chunk in raw:
            if chunk["distance"] > limit:
                chunk["status"] = "REJECT: above threshold"
            elif looks_like_reference(chunk["text"]):
                chunk["status"] = "REJECT: likely bibliography"
            else:
                chunk["status"] = "PASSES initial filters"
        return raw
 
    # ---------- prompt ----------
 
    def _fit_chunk(self, text: str, qvec: np.ndarray, limit: int) -> str:
        """Shrink a passage to `limit` chars by keeping its most relevant
        sentences (in original order) instead of blindly cutting the tail."""
        if len(text) <= limit:
            return text
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return text[:limit]
        sims = self.embed(sentences) @ qvec
        chosen, used = set(), 0
        for i in np.argsort(-sims):
            cost = len(sentences[i]) + 1
            if used + cost <= limit:
                chosen.add(int(i))
                used += cost
        if not chosen:
            return text[:limit]
        return " ".join(sentences[i] for i in sorted(chosen))
 
    def build_messages(self, question: str, chunks: list[dict]) -> list[dict]:
        qvec = self.embed_query(make_semantic_query(question))
        labels = ", ".join(f"[S{i}]" for i in range(1, len(chunks) + 1))
        passages = "\n\n".join(
            f"[S{i}] (Source: {c['source']}, page {c['page']})\n"
            f"{self._fit_chunk(c['text'], qvec, self.cfg.max_chars_per_chunk)}"
            for i, c in enumerate(chunks, start=1)
        )
 
        # Short on purpose: every extra rule costs seconds on a CPU.
        system_prompt = (
            "You are a careful research-paper assistant. "
            "Answer ONLY from the passages.\n"
            "Rules:\n"
            "1. Answer directly in at most 4 sentences. Paraphrase; do not quote.\n"
            "2. End every factual sentence with the labels that support it. "
            f"Valid labels: {labels}.\n"
            "3. Never copy the papers' own numeric citations like [14] or [14-18].\n"
            "4. Do not expand acronyms unless a passage does.\n"
            "5. Do not present a review's own methodology limits (search window, "
            "selection bias) as technical limits of RAG.\n"
            "6. If the passages only partly answer, say what is supported "
            "and what is missing.\n"
            f"7. If they do not answer it, reply exactly: \"{INSUFFICIENT}\"\n"
            "8. Do not use outside knowledge or bibliography entries."
        )
        # Passages first, question last: small models follow the end best.
        user_prompt = (
            f"Research-paper passages:\n{passages}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the supplied evidence and cite claims."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
 
    # ---------- LLM ----------
 
    def _llm_options(self, **extra) -> dict:
        return {
            "temperature": self.cfg.temperature,
            "num_predict": self.cfg.num_predict,
            "num_ctx": self.cfg.num_ctx,
            **extra,
        }
 
    def warm_up(self) -> float:
        start = time.perf_counter()
        self.llm.chat(
            model=self.cfg.llm_model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            keep_alive=self.cfg.keep_alive,
            options=self._llm_options(num_predict=1),
        )
        return time.perf_counter() - start
 
    def llm_ready(self) -> bool:
        try:
            self.llm.list()
            return True
        except Exception:
            return False
 
    def _stream_llm(self, messages: list[dict]) -> Iterator[str]:
        """Yield tokens; the generator's return value is a stats dict."""
        t0 = time.perf_counter()
        first = None
        final = None
        for part in self.llm.chat(
            model=self.cfg.llm_model,
            messages=messages,
            stream=True,
            keep_alive=self.cfg.keep_alive,
            options=self._llm_options(),
        ):
            token = part["message"].get("content", "")
            if token:
                if first is None:
                    first = time.perf_counter() - t0
                yield token
            if part.get("done"):
                final = part
 
        def secs(key):
            return round((final.get(key, 0) or 0) / 1e9, 3) if final else 0.0
 
        return {
            "first_token_s": round(first or 0.0, 3),
            "load_s": secs("load_duration"),
            "prompt_eval_s": secs("prompt_eval_duration"),
            "generation_s": secs("eval_duration"),
            "prompt_tokens": (final.get("prompt_eval_count", 0) if final else 0),
            "output_tokens": (final.get("eval_count", 0) if final else 0),
        }
 
    # ---------- grounding ----------
 
    def grounding(self, answer: str, chunks: list[dict]) -> dict:
        """Embedding-based support check (a heuristic, not proof).
 
        faithfulness        share of answer sentences whose best match in the
                            retrieved text reaches support_threshold
        citation_coverage   share of sentences that carry an [S#] label
        citation_precision  share of cited sentences actually supported by
                            the passages they cite
        """
        empty = {
            "faithfulness": None, "citation_coverage": None,
            "citation_precision": None, "rows": [],
        }
        if not chunks or INSUFFICIENT.lower() in answer.lower():
            return empty
 
        rows = []
        for sentence in split_sentences(answer):
            text = strip_citations(sentence)
            if len(text.split()) < 6 or ABSTAIN.search(text):
                continue
            labels = [n for n in cited_labels(sentence) if 1 <= n <= len(chunks)]
            rows.append({"sentence": text, "labels": labels})
 
        evidence, owner = [], []
        for i, chunk in enumerate(chunks, start=1):
            for s in split_sentences(chunk["text"]):
                if len(s.split()) >= 4:
                    evidence.append(s)
                    owner.append(i)
        if not rows or not evidence:
            return empty
 
        sims = self.embed([r["sentence"] for r in rows]) @ self.embed(evidence).T
        owner_arr = np.array(owner)
        threshold = self.cfg.support_threshold
 
        for k, row in enumerate(rows):
            row["support"] = float(sims[k].max())
            row["best_chunk"] = int(owner_arr[sims[k].argmax()])
            if row["labels"]:
                mask = np.isin(owner_arr, row["labels"])
                row["cited_support"] = float(sims[k][mask].max()) if mask.any() else 0.0
            else:
                row["cited_support"] = None
 
        cited_rows = [r for r in rows if r["labels"]]
        return {
            "faithfulness": sum(r["support"] >= threshold for r in rows) / len(rows),
            "citation_coverage": len(cited_rows) / len(rows),
            "citation_precision": (
                sum(r["cited_support"] >= threshold for r in cited_rows) / len(cited_rows)
                if cited_rows else None
            ),
            "rows": rows,
        }
 
    def _warnings(self, grounding: dict) -> list[str]:
        warnings = []
        rows = grounding.get("rows", [])
        threshold = self.cfg.support_threshold
        if rows and grounding["citation_coverage"] == 0:
            warnings.append("The answer contains no [S#] citations.")
        for r in rows:
            short = r["sentence"][:75] + ("..." if len(r["sentence"]) > 75 else "")
            if r["support"] < threshold:
                warnings.append(f"Weak support (similarity {r['support']:.2f}): \"{short}\"")
            elif r["cited_support"] is not None and r["cited_support"] < threshold:
                warnings.append(
                    f"Cited passage does not support it; best match is "
                    f"[S{r['best_chunk']}]: \"{short}\""
                )
        return warnings
 
    @staticmethod
    def format_sources(answer: str, chunks: list[dict]) -> str:
        if not chunks:
            return "Sources: none (no passages passed the retrieval threshold)."
        cited = [n for n in cited_labels(answer) if 1 <= n <= len(chunks)]
        if INSUFFICIENT.lower() in answer.lower():
            header, shown = "Closest passages (not confirmed as evidence):", range(1, len(chunks) + 1)
        elif cited:
            header, shown = "Sources:", cited
        else:
            header, shown = "Retrieved passages (the answer cited none):", range(1, len(chunks) + 1)
        lines = [header]
        for n in shown:
            c = chunks[n - 1]
            lines.append(
                f"  [S{n}] {c['source']}, page {c['page']} "
                f"(cosine similarity: {1.0 - c['distance']:.2f})"
            )
        return "\n".join(lines)
 
    # ---------- end-to-end ----------
 
    def _finish(self, question, text, retrieval, timings, t0, refused) -> Answer:
        chunks = retrieval.chunks
        text = clean_citations(text.strip(), len(chunks)) if chunks else text

        # Citation fallback: cite a sentence only if its best semantic match
        # in the retrieved passages meets the configured support threshold.
        # This is a heuristic, not a guarantee of factual entailment.
        if chunks and INSUFFICIENT.lower() not in text.lower():
            initial_grounding = self.grounding(text, chunks)
            for row in initial_grounding.get("rows", []):
                if row.get("labels") or row.get("support", 0.0) < self.cfg.support_threshold:
                    continue
                sentence = row["sentence"]
                label = f"[S{row['best_chunk']}]"
                match = re.search(re.escape(sentence), text)
                if match:
                    insertion = match.end()
                    text = text[:insertion] + " " + label + text[insertion:]
            text = clean_citations(text, len(chunks))

        grounding = self.grounding(text, chunks)
        timings["total_s"] = round(time.perf_counter() - t0, 3)
        return Answer(
            question=question,
            answer=text,
            chunks=chunks,
            sources_text=self.format_sources(text, chunks),
            warnings=self._warnings(grounding),
            refused=refused or INSUFFICIENT.lower() in text.lower(),
            timings=timings,
            grounding=grounding,
            retrieval=retrieval,
        )
 
    def answer_events(self, question: str) -> Iterator[tuple[str, object]]:
        """Yield ("retrieval", Retrieval), ("token", str)..., ("final", Answer).
 
        Streamed tokens are raw model output; the final Answer has cleaned
        citations. The LLM is never called when nothing passes retrieval.
        """
        t0 = time.perf_counter()
        retrieval = self.retrieve(question)
        timings = {"retrieval_s": round(retrieval.seconds, 3)}
        yield "retrieval", retrieval
 
        if not retrieval.chunks:
            yield "final", self._finish(question, INSUFFICIENT, retrieval, timings, t0, True)
            return
 
        stream = self._stream_llm(self.build_messages(question, retrieval.chunks))
        pieces, stats = [], {}
        while True:
            try:
                token = next(stream)
            except StopIteration as stop:
                stats = stop.value or {}
                break
            pieces.append(token)
            yield "token", token
 
        timings.update(stats)
        yield "final", self._finish(question, "".join(pieces), retrieval, timings, t0, False)
 
    def answer(self, question: str, on_token=None, on_retrieval=None) -> Answer:
        result = None
        for kind, payload in self.answer_events(question):
            if kind == "retrieval" and on_retrieval:
                on_retrieval(payload)
            elif kind == "token" and on_token:
                on_token(payload)
            elif kind == "final":
                result = payload
        return result
 
 
# ============================================================
# 4. COMMAND-LINE APPLICATION
# ============================================================
 
def _print_retrieval(r: Retrieval) -> None:
    print("Semantic query:", r.semantic_query)
    if r.source:
        print("Metadata filter:", r.source, f"| page {r.page}" if r.page is not None else "")
    print(f"Retrieved {len(r.chunks)} chunks | skipped: {r.skipped}")
    for i, c in enumerate(r.chunks, start=1):
        print(f"  [S{i}] {c['source']} | page {c['page']} | "
              f"distance {c['distance']:.4f} | keyword overlap {c['lexical']:.0%}")
    print()
 
 
def _print_timings(t: dict) -> None:
    print(
        f"\nTimings: retrieval {t.get('retrieval_s', 0)}s | "
        f"first token {t.get('first_token_s', 0)}s | "
        f"prompt {t.get('prompt_eval_s', 0)}s ({t.get('prompt_tokens', 0)} tok) | "
        f"generation {t.get('generation_s', 0)}s ({t.get('output_tokens', 0)} tok) | "
        f"total {t.get('total_s', 0)}s"
    )
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Research Intelligence Assistant")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the index first")
    parser.add_argument("--quiet", action="store_true", help="hide retrieval/timing diagnostics")
    args = parser.parse_args()
 
    print("Loading embedding model...")
    pipe = RAGPipeline()
    try:
        _ = pipe.embedder  # load once at startup, not during the first question
    except Exception as error:
        raise RuntimeError(
            "Could not load the embedding model. Check sentence-transformers installation and network/cache access."
        ) from error

    if args.rebuild or not pipe.has_index():
        print("Building the index...")
        pipe.build_index()
    print("Using index:", pipe.collection().count(), "chunks.")
 
    print("Warming up Ollama...")
    try:
        print(f"Warm-up completed in {pipe.warm_up():.2f}s.")
    except Exception as error:
        raise RuntimeError("Could not connect to Ollama. Make sure it is running.") from error
 
    print("\n" + "=" * 60)
    print("RESEARCH INTELLIGENCE ASSISTANT  |  model:", pipe.cfg.llm_model)
    print("Name a PDF (e.g. Research_paper_1.pdf) to filter by source.")
    print("'/explore <question>' shows raw matches. 'exit' quits.")
    print("=" * 60)
 
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue
 
        try:
            if question.lower().startswith("/explore"):
                q = question[len("/explore"):].strip() or "retrieval augmented generation"
                print(f"\nRaw results for: {q}\nMax distance: {pipe.cfg.max_distance:.2f}\n" + "-" * 70)
                for c in pipe.explore(q):
                    snippet = c["text"][:180].replace("\n", " ")
                    print(f"{c['status']} | distance={c['distance']:.4f} | "
                          f"{c['source']} | page {c['page']}\n  {snippet}...\n")
                continue
 
            streamed = []
 
            def on_token(token):
                streamed.append(token)
                print(token, end="", flush=True)
 
            result = pipe.answer(
                question,
                on_token=on_token,
                on_retrieval=None if args.quiet else _print_retrieval,
            )
            if not streamed:
                print(result.answer)
            else:
                print()
                if result.answer.strip() != "".join(streamed).strip():
                    print("\n[Cleaned answer]\n" + result.answer)
 
            if not args.quiet:
                _print_timings(result.timings)
            if result.warnings:
                print("\nGrounding warnings:")
                for w in result.warnings:
                    print(" -", w)
            print("\n" + result.sources_text)
 
        except Exception as error:
            print(f"\nError: {error}")
 
 
if __name__ == "__main__":
    main()
 
