import json
import re
import time
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chromaDB"
CHUNKS_PATH = BASE_DIR / "output" / "chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
COLLECTION_NAME = "research_papers"

# Retrieval settings
CANDIDATES = 30
TOP_K = 4
MAX_DISTANCE = 0.55
MAX_PER_SOURCE = 2
MAX_CHARS_PER_CHUNK = 700      # 4 chunks x 700 chars keeps the prompt well under NUM_CTX

# Ollama settings
KEEP_ALIVE = "30m"
NUM_CTX = 2048
NUM_PREDICT = 300
TEMPERATURE = 0.2

# Lightweight grounding diagnostic; not a factuality guarantee
MIN_SUPPORT = 0.25
SHOW_DIAGNOSTICS = True

INSUFFICIENT = (
    "The retrieved passages do not contain enough information "
    "to answer this reliably."
)

REFERENCE_MARKERS = re.compile(
    r"\barxiv\b|\bdoi\b|\bcrossref\b|\bpreprint\b|"
    r"\bproceedings\b|\bvol\.\b|\bpp\.\b|\bet al\.",
    re.IGNORECASE,
)

CITATION_PATTERN = re.compile(
    r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]"
)

STOPWORDS = {
    "that", "this", "with", "from", "have", "were", "which",
    "their", "there", "these", "those", "been", "also",
    "such", "using", "used", "into", "more", "than", "they",
    "will", "when", "what", "while", "about", "other",
    "paper", "passage", "passages", "provided", "mentions",
    "states", "describes", "answer", "according", "source",
    "sources", "because", "through", "does", "very", "some",
    "many", "each", "most",
}


# ============================================================
# 2. INITIALIZE
# ============================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
_sources_cache = None


# ============================================================
# 3. LOAD AND BUILD INDEX
# ============================================================

def load_chunks():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}\n"
            "Run python ingestion.py first."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found in chunks.json.")

    return chunks


def build_index(chunks=None):
    """Rebuild the collection from output/chunks.json."""

    global _sources_cache

    if chunks is None:
        chunks = load_chunks()

    for item in chunks:
        if not all(k in item for k in ("id", "text", "metadata")):
            raise ValueError(
                "Every chunk must contain id, text and metadata."
            )

        metadata = item["metadata"]

        if not all(k in metadata for k in ("source", "page", "chunk_id")):
            raise ValueError(
                "Chunk metadata must contain source, page and chunk_id."
            )

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    _sources_cache = None

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [item["text"] for item in batch]

        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        collection.add(
            ids=[item["id"] for item in batch],
            documents=texts,
            metadatas=[item["metadata"] for item in batch],
            embeddings=embeddings.tolist(),
        )

        print(
            f"Indexed {min(start + batch_size, len(chunks))}"
            f"/{len(chunks)} chunks."
        )

    print("Index rebuilt successfully.")
    print("Chunks stored:", collection.count())

    return collection


def get_collection():
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        if collection.count() > 0:
            return collection

    except Exception:
        pass

    raise RuntimeError(
        "No indexed documents found. Run ingestion.py and "
        "build the index using python built_index.py."
    )


# ============================================================
# 4. METADATA FILTERING
# ============================================================

def get_indexed_sources(collection):
    global _sources_cache

    if _sources_cache is None:
        result = collection.get(include=["metadatas"])

        _sources_cache = sorted({
            item["source"]
            for item in result["metadatas"]
            if item and item.get("source")
        })

    return _sources_cache


def parse_filters(question, collection):
    """
    Recognize filename and page requests, for example:
      Explain RAG in Research_paper_1.pdf
      What is on page 5 of Research_paper_1.pdf?
      Explain RAG in research paper 2
    """

    sources = get_indexed_sources(collection)

    by_name = {
        Path(name.replace("\\", "/")).name.lower(): name
        for name in sources
    }

    source = None

    match = re.search(r"([\w.-]+\.pdf)\b", question, flags=re.IGNORECASE)

    if match:
        requested = match.group(1).lower()

        if requested not in by_name:
            raise ValueError(
                f"'{requested}' was not found in the index.\n"
                f"Available sources: {', '.join(sorted(by_name))}"
            )

        source = by_name[requested]

    else:
        match = re.search(
            r"research[\s_-]*paper[\s_-]*(\d+)",
            question,
            flags=re.IGNORECASE,
        )

        if match:
            requested = f"research_paper_{match.group(1)}.pdf".lower()
            source = by_name.get(requested)

            if source is None:
                raise ValueError(f"{requested} was not found in the index.")

    page = None

    if source:
        match = re.search(r"\bpage\s+(\d+)\b", question, flags=re.IGNORECASE)

        if match:
            page = int(match.group(1))

    return source, page


def build_where(source, page):
    if source and page is not None:
        return {"$and": [{"source": source}, {"page": page}]}

    if source:
        return {"source": source}

    return None


# ============================================================
# 5. RETRIEVAL
# ============================================================

def looks_like_reference(text):
    markers = len(REFERENCE_MARKERS.findall(text))
    initials = len(re.findall(r"\b[A-Z]\.", text))
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))

    return markers >= 2 or (initials >= 6 and years >= 1)


def make_semantic_query(question):
    """
    Remove filename/page instructions from the text that gets embedded
    (those belong in metadata filters). The LLM still sees the original
    question unchanged.
    """

    query = question.strip()

    query = re.sub(r"[\w.-]+\.pdf", " ", query, flags=re.IGNORECASE)
    query = re.sub(
        r"\bresearch[\s_-]*paper[\s_-]*\d+\b", " ", query, flags=re.IGNORECASE
    )
    query = re.sub(r"\bpage\s+\d+\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\b(please|tell me|explain)\b", " ", query, flags=re.IGNORECASE)

    query = re.sub(r"[^\w\s'-]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    # Drop a dangling preposition left behind, e.g. "... discussed in".
    query = re.sub(
        r"\b(in|from|of|on|at)\s*$", "", query, flags=re.IGNORECASE
    ).strip()

    # If almost nothing is left, fall back to the original question.
    if len(query.split()) < 2:
        return question.strip()

    return query


def looks_like_review_methodology(text):
    """Detect review-methodology caveats, not technical RAG limitations."""

    lower = text.lower()

    explicit_review_limit = any(
        phrase in lower
        for phrase in (
            "limitations of this review",
            "limitations of the review",
            "limitations of this study",
            "limitations of the present study",
            "limitations of our review",
            "limitations of the systematic review",
        )
    )

    methodology_signals = any(
        phrase in lower
        for phrase in (
            "publication window",
            "peer-reviewed, english-language",
            "peer-reviewed english-language",
            "reported findings",
            "review methodology",
            "selection bias",
            "generalizability of the results",
            "completeness of the review",
        )
    )

    return explicit_review_limit or (
        methodology_signals
        and any(term in lower for term in ("review", "studies", "publication"))
    )


def asks_technical_rag_limits(question):
    q = question.lower()

    return (
        "rag" in q
        and "limitation" in q
        and not any(
            phrase in q
            for phrase in (
                "limitations of the review",
                "limitations of this review",
                "limitations of the paper",
                "limitations of this paper",
                "limitations of the study",
            )
        )
    )


def run_search(question, n_results, where=None):
    """Vector search on the semantic query; metadata filters stay separate."""

    collection = get_collection()

    if collection.count() == 0:
        return []

    semantic_query = make_semantic_query(question)

    if SHOW_DIAGNOSTICS:
        print("Semantic query:", semantic_query)

    query_embedding = embedding_model.encode(
        [semantic_query],
        normalize_embeddings=True,
    )[0]

    args = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }

    if where:
        args["where"] = where

    results = collection.query(**args)

    matches = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "chunk_id": metadata["chunk_id"],
            "distance": float(distance),
        })

    return matches


def retrieve_chunks(question, top_k=TOP_K, max_distance=MAX_DISTANCE):
    """Retrieve, threshold, filter, and diversify passages."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    collection = get_collection()
    source, page = parse_filters(question, collection)
    where = build_where(source, page)

    raw = run_search(question, CANDIDATES, where)

    technical_limits = asks_technical_rag_limits(question)

    retrieved = []
    seen = set()
    per_source = {}

    skipped = {
        "too_far": 0,
        "reference": 0,
        "duplicate": 0,
        "per_source_cap": 0,
        "review_methodology": 0,
    }

    for chunk in raw:
        # Enforce the threshold BEFORE any passage reaches the LLM.
        if chunk["distance"] > max_distance:
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

        name = chunk["source"]

        if not source and per_source.get(name, 0) >= MAX_PER_SOURCE:
            skipped["per_source_cap"] += 1
            continue

        seen.add(key)
        per_source[name] = per_source.get(name, 0) + 1
        retrieved.append(chunk)

        if len(retrieved) >= top_k:
            break

    if SHOW_DIAGNOSTICS:
        if source:
            print(
                "Metadata filter:",
                source,
                f"| page {page}" if page is not None else "",
            )

        print(f"Retrieved {len(retrieved)} chunks | skipped: {skipped}")

        for i, chunk in enumerate(retrieved, start=1):
            print(
                f"  [S{i}] {chunk['source']} | "
                f"Page {chunk['page']} | "
                f"Distance {chunk['distance']:.4f}"
            )

    return retrieved


def explore_retrieval(question, n=10):
    """Display raw matches; does not call the LLM."""

    collection = get_collection()
    source, page = parse_filters(question, collection)
    where = build_where(source, page)

    raw = run_search(question, n, where)

    print(f"\nRaw retrieval results for: {question}")
    print(f"Configured maximum distance: {MAX_DISTANCE:.2f}")
    print("-" * 75)

    if not raw:
        print("No matching passages found.")
        return

    for chunk in raw:
        distance = chunk["distance"]

        if distance > MAX_DISTANCE:
            status = "REJECT: above threshold"
        elif looks_like_reference(chunk["text"]):
            status = "REJECT: likely bibliography"
        else:
            status = "PASSES initial filters"

        snippet = chunk["text"][:180].replace("\n", " ")

        print(
            f"{status} | distance={distance:.4f} | "
            f"{chunk['source']} | page {chunk['page']}"
        )
        print(f"  {snippet}...")
        print()


# ============================================================
# 6. OLLAMA
# ============================================================

def llm_options():
    return {
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "num_ctx": NUM_CTX,
    }


def warm_up_llm():
    start = time.perf_counter()

    ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "Reply with OK."}],
        keep_alive=KEEP_ALIVE,
        options={**llm_options(), "num_predict": 1},
    )

    return time.perf_counter() - start


def print_diagnostics(response):
    print(
        "Model load:",
        round(response.get("load_duration", 0) / 1e9, 2),
        "s",
    )
    print(
        "Prompt processing:",
        round(response.get("prompt_eval_duration", 0) / 1e9, 2),
        "s",
        f"({response.get('prompt_eval_count', 0)} tokens)",
    )
    print(
        "Generation:",
        round(response.get("eval_duration", 0) / 1e9, 2),
        "s",
        f"({response.get('eval_count', 0)} tokens)",
    )


# ============================================================
# 7. PROMPT DESIGN
# ============================================================

def build_messages(question, chunks):
    labels = ", ".join(f"[S{i}]" for i in range(1, len(chunks) + 1))

    passages = "\n\n".join(
        (
            f"[S{i}] (Source: {chunk['source']}, page {chunk['page']})\n"
            f"{chunk['text'][:MAX_CHARS_PER_CHUNK]}"
        )
        for i, chunk in enumerate(chunks, start=1)
    )

    # Kept short on purpose: every extra rule costs seconds on a CPU.
    system_prompt = (
        "You are a careful research-paper assistant. "
        "Answer ONLY from the passages.\n"
        "Rules:\n"
        "1. Answer the question directly in at most 4 sentences. "
        "Paraphrase; do not quote.\n"
        "2. End every factual sentence with the source labels that "
        f"support it. Valid labels: {labels}.\n"
        "3. Never copy the papers' own numeric citations like [14] or [14-18].\n"
        "4. Do not present a review's own methodology limits (search window, "
        "selection bias) as technical limits of RAG.\n"
        "5. If the passages only partly answer, say what is supported "
        "and what is missing.\n"
        f"6. If they do not answer it, reply exactly: \"{INSUFFICIENT}\"\n"
        "7. Do not use outside knowledge or bibliography entries."
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


# ============================================================
# 8. CITATIONS AND GROUNDING CHECKS
# ============================================================

def clean_citations(answer, n_sources):
    """
    - Keep valid [S1]-style labels, drop invented ones like [S15]
      (including inside groups such as [S2, S7]).
    - Remove the papers' own numeric citations: [14], [14, 67], [14-18].
    """

    valid = {f"S{i}" for i in range(1, n_sources + 1)}

    def fix_group(match):
        labels = [
            label for label in re.findall(r"S\d+", match.group(0))
            if label in valid
        ]
        return "[" + ", ".join(labels) + "]" if labels else ""

    answer = CITATION_PATTERN.sub(fix_group, answer)

    answer = re.sub(
        r"(?<!\w)\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]",
        "",
        answer,
    )

    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\s+([,.;:!?])", r"\1", answer)

    return answer.strip()


def cited_labels(answer):
    numbers = set()

    for group in CITATION_PATTERN.findall(answer):
        numbers.update(int(n) for n in re.findall(r"S(\d+)", group))

    return sorted(numbers)


def content_words(text):
    return {
        word
        for word in re.findall(r"[a-z]{4,}", text.lower())
        if word not in STOPWORDS
    }


def verify_answer(answer, chunks):
    """
    Heuristic checks for missing citations and low word overlap.
    These checks cannot prove that a claim is factually correct.
    """

    warnings = []

    if INSUFFICIENT.lower() in answer.lower():
        return warnings

    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)

    for sentence in sentences:
        sentence = sentence.strip(" -*•\t")
        words = content_words(sentence)

        if len(sentence.split()) < 6 or not words:
            continue

        labels = []
        for group in CITATION_PATTERN.findall(sentence):
            labels.extend(int(n) for n in re.findall(r"S(\d+)", group))

        labels = sorted({n for n in labels if 1 <= n <= len(chunks)})

        short = sentence[:75] + ("..." if len(sentence) > 75 else "")

        if not labels:
            warnings.append(f"No citation: \"{short}\"")
            continue

        evidence = content_words(
            " ".join(chunks[n - 1]["text"] for n in labels)
        )

        overlap = len(words & evidence) / max(len(words), 1)

        if overlap < MIN_SUPPORT:
            warnings.append(f"Low word overlap ({overlap:.0%}): \"{short}\"")

    return warnings


def format_sources(answer, chunks):
    """Always display source references or an explicit no-source notice."""

    if not chunks:
        return "Sources: none (no passages passed the retrieval threshold)."

    cited = [n for n in cited_labels(answer) if 1 <= n <= len(chunks)]

    if INSUFFICIENT.lower() in answer.lower():
        header = "Closest passages (not confirmed as evidence):"
        shown = list(range(1, len(chunks) + 1))

    elif cited:
        header = "Sources:"
        shown = sorted(set(cited))

    else:
        header = "Retrieved passages (the answer cited none):"
        shown = list(range(1, len(chunks) + 1))

    lines = [header]

    for number in shown:
        chunk = chunks[number - 1]
        similarity = 1.0 - chunk["distance"]

        lines.append(
            f"  [S{number}] {chunk['source']}, page {chunk['page']} "
            f"(cosine similarity: {similarity:.2f})"
        )

    return "\n".join(lines)


# ============================================================
# 9. ANSWER GENERATION
# ============================================================

def generate_answer(question, chunks, stream=False):
    messages = build_messages(question, chunks)

    if not stream:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            stream=False,
            keep_alive=KEEP_ALIVE,
            options=llm_options(),
        )

        if SHOW_DIAGNOSTICS:
            print_diagnostics(response)

        answer = response["message"]["content"]

    else:
        pieces = []
        first_token_time = None
        final_response = None
        start = time.perf_counter()

        for part in ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            keep_alive=KEEP_ALIVE,
            options=llm_options(),
        ):
            token = part["message"].get("content", "")

            if token and first_token_time is None:
                first_token_time = time.perf_counter() - start

            pieces.append(token)
            print(token, end="", flush=True)

            if part.get("done"):
                final_response = part

        print()
        answer = "".join(pieces)

        if SHOW_DIAGNOSTICS:
            if first_token_time is not None:
                print("Time to first token:", round(first_token_time, 2), "s")

            if final_response:
                print_diagnostics(final_response)

    return clean_citations(answer.strip(), len(chunks))


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================

def answer_question(question, stream=False):
    """
    Returns a dict: answer, sources, warnings, chunks, refused.
    """

    start = time.perf_counter()
    chunks = retrieve_chunks(question)

    # Do not call the LLM when no chunks pass the threshold.
    if not chunks:
        if SHOW_DIAGNOSTICS:
            print("LLM not called: no chunks passed retrieval filters.")
            print(
                "Total pipeline time:",
                round(time.perf_counter() - start, 2),
                "s",
            )

        return {
            "answer": INSUFFICIENT,
            "sources": format_sources(INSUFFICIENT, []),
            "warnings": [],
            "chunks": [],
            "refused": True,
        }

    answer = generate_answer(question, chunks, stream=stream)

    result = {
        "answer": answer,
        "sources": format_sources(answer, chunks),
        "warnings": verify_answer(answer, chunks),
        "chunks": chunks,
        "refused": INSUFFICIENT.lower() in answer.lower(),
    }

    if SHOW_DIAGNOSTICS:
        print(
            "Total pipeline time:",
            round(time.perf_counter() - start, 2),
            "s",
        )

    return result


# ============================================================
# 11. COMMAND-LINE APPLICATION
# ============================================================

if __name__ == "__main__":
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        if collection.count() == 0:
            print("The index is empty. Building the index...")
            build_index()

    except Exception:
        print("No existing index found. Building the index...")
        build_index()

    print("\nUsing existing index:", get_collection().count(), "chunks.")

    print("\nWarming up Ollama...")
    try:
        print(f"Warm-up completed in {warm_up_llm():.2f} seconds.")
    except Exception as error:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure it is running."
        ) from error

    print("\n" + "=" * 60)
    print("RESEARCH INTELLIGENCE ASSISTANT")
    print("Model:", LLM_MODEL)
    print("Use a PDF filename to filter by source.")
    print("Use '/explore your question' to inspect raw matches.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            # Handle /explore BEFORE normal answer generation.
            if question.lower().startswith("/explore"):
                explore_question = question[len("/explore"):].strip()

                if not explore_question:
                    explore_question = "retrieval augmented generation"

                explore_retrieval(explore_question)
                continue

            result = answer_question(question, stream=True)

            # Streamed answers are already on screen; only print when the
            # LLM was skipped (nothing passed the threshold).
            if not result["chunks"]:
                print(result["answer"])

            if result["warnings"]:
                print("\nGrounding warnings:")

                for warning in result["warnings"]:
                    print(" -", warning)

            print("\n" + result["sources"])

        except Exception as error:
            print(f"\nError: {error}")