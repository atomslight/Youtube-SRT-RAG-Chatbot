import os
import re

import numpy as np
import pysrt
import yt_dlp
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rank_bm25 import BM25Okapi

# ============================================================
# CONFIG
# ============================================================
EMBED_MODEL     = "dengcao/Qwen3-Embedding-8B:Q4_K_M"
RERANK_MODEL    = "Krakekai/qwen3-reranker-8b"   # only used if USE_CROSS_ENCODER_RERANKER=False
ANSWER_MODEL    = "gemma4:31b-cloud"
OLLAMA_BASE_URL = "http://localhost:11434"

USE_CROSS_ENCODER_RERANKER = True
CROSS_ENCODER_MODEL        = "BAAI/bge-reranker-v2-m3"

# Chunking — fixed windows, no semantic boundary detection
CHUNK_SECONDS   = 75
OVERLAP_SECONDS = 15

# Query expansion
ENABLE_QUERY_EXPANSION = True
NUM_QUERY_EXPANSIONS   = 5

# Token budget — how many tokens of chunk text the LLM context can hold.
# Tune to: model_context_window - prompt_overhead - max_output_tokens.
# chars // 4 is used as the token approximation throughout (standard estimate).
CONTEXT_TOKEN_BUDGET       = 4000

DEDUP_SIMILARITY_THRESHOLD = 0.92

# Disk logging
SAVE_CHUNKS_TO_DISK = True
CHUNK_SAVE_ROOT     = "retrieved_chunks"


# ============================================================
# UTILS
# ============================================================
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def save_chunks_to_folder(stage_name, chunks_with_scores, score_label="score", base_dir=CHUNK_SAVE_ROOT):
    if not SAVE_CHUNKS_TO_DISK:
        return
    stage_dir = os.path.join(base_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)
    summary_lines = []
    for rank, (doc, score) in enumerate(chunks_with_scores, start=1):
        meta      = doc.metadata
        safe_range = meta["subtitle_index_range"].replace(" ", "")
        fname     = f"chunk_{rank:02d}_{safe_range}.txt"
        with open(os.path.join(stage_dir, fname), "w", encoding="utf-8") as f:
            f.write(f"Rank: {rank}\n")
            f.write(f"{score_label}: {score:.6f}\n")
            f.write(f"Video: {meta['video_title']} ({meta['video_id']})\n")
            f.write(f"Time: [{meta['start_time']} - {meta['end_time']}]\n")
            f.write(f"Duration: {meta['duration_seconds']:.0f}s\n")
            f.write(f"Subtitle index range: {meta['subtitle_index_range']}\n")
            f.write("\n--- TEXT ---\n")
            f.write(doc.page_content)
        summary_lines.append(
            f"#{rank} | {score_label}={score:.6f} | "
            f"[{meta['start_time']} - {meta['end_time']}] | file={fname}\n"
            f"    {doc.page_content[:100]}..."
        )
    with open(os.path.join(stage_dir, "_scores.txt"), "w", encoding="utf-8") as f:
        f.write(f"Stage: {stage_name}\n")
        f.write(f"Total chunks: {len(chunks_with_scores)}\n\n")
        f.write("\n".join(summary_lines))
    print(f"  -> Saved {len(chunks_with_scores)} chunks to {stage_dir}/")


# ============================================================
# LANGUAGE DETECTION  (only addition — rest of file is untouched)
# ============================================================
def detect_subtitle_lang(url: str) -> str:
    """Probe available subtitle/auto-caption languages; prefer en then hi."""
    with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    available = (
        set(info.get("subtitles", {}).keys())
        | set(info.get("automatic_captions", {}).keys())
    )
    for lang in ["en", "hi", "en-US", "en-GB"]:
        if lang in available:
            return lang
    return next(iter(available), "en")


# ============================================================
# STEP 1: Download subtitles
# ============================================================
url  = input("ENTER YT URL:\n")
lang = detect_subtitle_lang(url)  # ← dynamic language detection

ydl_opts = {
    "skip_download":    True,
    "writesubtitles":   True,
    "writeautomaticsub": True,
    "subtitleslangs":   [lang],          # ← was ["en"]
    "subtitlesformat":  "srt",
    "outtmpl":          "%(id)s",
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)

video_id    = info["id"]
video_title = info.get("title", "Unknown Title")
srt_file    = f"{video_id}.{lang}.srt"  # ← was f"{video_id}.en.srt"
print("Subtitle file:", srt_file)

subs = pysrt.open(srt_file)
print(f"Loaded {len(subs)} subtitle lines")

# ============================================================
# STEP 2: Embedding model
# ============================================================
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

# ============================================================
# STEP 3: Fixed-window chunking with overlap
#
# Replaces all semantic boundary detection (window embeddings, relative/
# absolute triggers, tight-cluster logic, future-recovery heuristics, drift
# warnings). Those added complexity without measurably outperforming simple
# fixed chunks when a strong cross-encoder reranker + query expansion is
# already in the pipeline.
# ============================================================
def make_chunks(subs, video_title, video_id):
    """
    Slide a CHUNK_SECONDS window across the transcript, stepping by
    CHUNK_SECONDS each time. Each chunk is extended ±OVERLAP_SECONDS
    so adjacent chunks share content at their boundaries.
    """
    if not subs:
        return []

    total_end = subs[-1].end.ordinal / 1000.0
    documents = []
    chunk_idx = 0
    t         = 0.0

    while t < total_end:
        chunk_end = t + CHUNK_SECONDS
        ov_start  = max(0.0, t - OVERLAP_SECONDS) if chunk_idx > 0 else 0.0
        ov_end    = chunk_end + OVERLAP_SECONDS

        group = [
            s for s in subs
            if s.start.ordinal / 1000.0 < ov_end
            and s.end.ordinal   / 1000.0 > ov_start
        ]

        if group:
            text      = " ".join(s.text.replace("\n", " ") for s in group)
            start_sec = group[0].start.ordinal  / 1000.0
            end_sec   = group[-1].end.ordinal   / 1000.0
            documents.append(Document(
                page_content=text,
                metadata={
                    "video_title":          video_title,
                    "video_id":             video_id,
                    "start_time":           str(group[0].start),
                    "end_time":             str(group[-1].end),
                    "start_seconds":        start_sec,
                    "end_seconds":          end_sec,
                    "duration_seconds":     end_sec - start_sec,
                    "subtitle_index_range": f"{group[0].index}-{group[-1].index}",
                },
            ))

        t         += CHUNK_SECONDS
        chunk_idx += 1

    return documents


documents = make_chunks(subs, video_title, video_id)
print(f"\nCreated {len(documents)} chunks  ({CHUNK_SECONDS}s window / {OVERLAP_SECONDS}s overlap)")
for d in documents:
    print(f"  [{d.metadata['start_time']} - {d.metadata['end_time']}] "
          f"({d.metadata['duration_seconds']:.0f}s)  {d.page_content[:80]}...")

save_chunks_to_folder(
    "00_all_chunks",
    [(d, d.metadata["duration_seconds"]) for d in documents],
    score_label="duration_seconds",
)

# ============================================================
# STEP 4: Chunk embeddings + hybrid retrieval (BM25 + Dense + RRF)
# ============================================================
print("\nBuilding hybrid retrieval indexes...")

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents)

chunk_texts      = [d.page_content for d in documents]
chunk_embeddings = embeddings.embed_documents(chunk_texts)

tokenized_corpus = [re.findall(r"\w+", d.page_content.lower()) for d in documents]
bm25             = BM25Okapi(tokenized_corpus)


def hybrid_retrieve_single(query):
    """BM25 + dense retrieval merged via Reciprocal Rank Fusion (RRF).
    Retrieves ALL chunks — token budget downstream decides what gets used,
    not an arbitrary top-k count here.
    """
    n = len(documents)
    # Dense
    dense_ranked = [doc for doc, _ in vector_store.similarity_search_with_score(query, k=n)]

    # BM25
    bm25_scores     = bm25.get_scores(re.findall(r"\w+", query.lower()))
    bm25_ranked_idx = np.argsort(bm25_scores)[::-1]
    bm25_ranked     = [documents[i] for i in bm25_ranked_idx]

    k_rrf      = 60
    rrf_scores = {}
    doc_lookup = {}

    for rank, doc in enumerate(dense_ranked):
        key = doc.metadata["subtitle_index_range"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k_rrf + rank + 1)
        doc_lookup[key] = doc

    for rank, doc in enumerate(bm25_ranked):
        key = doc.metadata["subtitle_index_range"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k_rrf + rank + 1)
        doc_lookup[key] = doc

    return rrf_scores, doc_lookup


def hybrid_retrieve(queries):
    """
    Run hybrid_retrieve_single over one or more query variants and sum
    RRF scores across all variants (multi-query fusion).
    Returns ALL chunks ranked by fused RRF score — no count cutoff.
    """
    if isinstance(queries, str):
        queries = [queries]

    combined_rrf    = {}
    combined_lookup = {}

    for q in queries:
        rrf_scores, doc_lookup = hybrid_retrieve_single(q)
        for key, score in rrf_scores.items():
            combined_rrf[key] = combined_rrf.get(key, 0) + score
        combined_lookup.update(doc_lookup)

    merged = sorted(combined_lookup.items(), key=lambda x: combined_rrf[x[0]], reverse=True)
    return [(doc, combined_rrf[key]) for key, doc in merged]


# ============================================================
# STEP 5: Query expansion
# ============================================================
expansion_llm = ChatOllama(model=ANSWER_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)

QUERY_EXPANSION_PROMPT = (
    "Given the following search query about a video transcript, "
    "generate {n} alternative short search phrases that cover related sub-topics or "
    "phrasings someone might use when discussing the same subject in a tutorial or "
    "walkthrough video. Output ONLY the phrases, one per line, no numbering, no extra text.\n\n"
    "Query: {query}"
)


def expand_query(query, n=NUM_QUERY_EXPANSIONS):
    if not ENABLE_QUERY_EXPANSION:
        return [query]
    response   = expansion_llm.invoke(QUERY_EXPANSION_PROMPT.format(query=query, n=n))
    lines      = [line.strip("-* \t") for line in response.content.strip().split("\n")]
    expansions = [line for line in lines if line]
    return [query] + expansions[:n]


# ============================================================
# STEP 6: Reranking  (cross-encoder preferred, Ollama fallback)
# ============================================================
if USE_CROSS_ENCODER_RERANKER:
    import torch
    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    def rerank(query, candidates):
        """Score ALL candidates in one batched forward pass. Token budget
        downstream decides what gets used — no count cutoff here."""
        pairs  = [(query, doc.page_content[:1000]) for doc, _ in candidates]
        scores = cross_encoder.predict(pairs, activation_fn=torch.nn.Sigmoid())
        scored = sorted(zip([doc for doc, _ in candidates], scores),
                        key=lambda x: x[1], reverse=True)
        return [(doc, float(s)) for doc, s in scored]

else:
    reranker_llm = ChatOllama(
        model=RERANK_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0, num_predict=1,
    )
    RERANK_PROMPT = (
        'Judge whether the Passage is relevant to the Query. Answer only "yes" or "no".\n\n'
        "<Query>: {query}\n<Passage>: {passage}"
    )

    def rerank(query, candidates):
        scored = []
        for doc, rrf_score in candidates:
            response  = reranker_llm.invoke(
                RERANK_PROMPT.format(query=query, passage=doc.page_content[:1000])
            )
            relevance = 1 if response.content.strip().lower().startswith("y") else 0
            scored.append((doc, relevance + rrf_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in scored]


# ============================================================
# STEP 7: Context compression — deduplicate near-identical chunks
# ============================================================
def get_chunk_embedding(doc):
    return chunk_embeddings[chunk_texts.index(doc.page_content)]


def compress_context(candidates):
    kept, kept_embs = [], []
    for doc, score in candidates:
        emb = get_chunk_embedding(doc)
        if not any(cosine_sim(emb, ke) > DEDUP_SIMILARITY_THRESHOLD for ke in kept_embs):
            kept.append((doc, score))
            kept_embs.append(emb)
    return kept


# ============================================================
# STEP 8: Token-budget fill
#
# Greedily admit chunks from the TOP of the reranked list until the
# token budget is exhausted. Using reranker score (strongest signal)
# as the priority — not RRF score — so the best chunks get the
# limited seats. A smaller chunk ranked lower is still admitted if
# it fits, so we never waste leftover budget.
# ============================================================
def token_count(text):
    """chars // 4 is a standard rough token approximation."""
    return len(text) // 4


def fill_token_window(reranked_chunks, budget=CONTEXT_TOKEN_BUDGET):
    selected, used = [], 0
    for doc, score in reranked_chunks:
        tokens = token_count(doc.page_content)
        if used + tokens <= budget:
            selected.append((doc, score))
            used += tokens
        # Don't break — a smaller lower-ranked chunk may still fit
    print(f"  Token budget: {budget} | Used: {used} | Chunks admitted: {len(selected)}")
    return selected


# ============================================================
# STEP 8: Chronological reordering
# ============================================================
def chronological_order(chunks):
    return sorted(chunks, key=lambda x: x[0].metadata["start_seconds"])


# ============================================================
# STEP 9: Answer generation
# ============================================================
answer_llm = ChatOllama(model=ANSWER_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)

ANSWER_PROMPT = """You are answering a question about a YouTube video using transcript excerpts.
The excerpts are provided in CHRONOLOGICAL ORDER (the order events happened in the video),
each with a timestamp range [HH:MM:SS - HH:MM:SS].

STRICT RULES:
- Use ONLY information present in the excerpts below. Never invent commands, steps, or facts.
- If information needed to answer is missing from the excerpts, explicitly say so.
- Merge information from multiple excerpts into a single coherent answer.
- If excerpts repeat the same information, deduplicate — state it once.
- Always cite the relevant timestamp range [HH:MM:SS-HH:MM:SS] next to each fact or step.

FORMAT RULES:
- Process / installation / setup / workflow / steps questions → numbered list in
  chronological order, each step including a short imperative description, the exact
  command if mentioned, and the timestamp range.
- Troubleshooting questions → problem / diagnostic steps / fix, each with timestamps.
- Conceptual questions → clear prose paragraphs with timestamp citations.
- Coding questions → preserve all code and command syntax exactly as in the transcript.

Transcript excerpts (chronological order):
{context}

Question: {question}

Answer:"""


def format_context(chunks):
    return "\n\n---\n\n".join(
        f"[{doc.metadata['start_time']} - {doc.metadata['end_time']}]\n{doc.page_content}"
        for doc, _ in chunks
    )


def generate_answer(query, chunks):
    response = answer_llm.invoke(
        ANSWER_PROMPT.format(context=format_context(chunks), question=query)
    )
    return response.content


# ============================================================
# MAIN QUERY LOOP
# ============================================================
qa = input("\nWhat info are you searching for?\n")

query_slug      = re.sub(r"[^a-zA-Z0-9]+", "_", qa.strip().lower())[:50].strip("_") or "query"
CHUNK_SAVE_ROOT = os.path.join(CHUNK_SAVE_ROOT, query_slug)

print("\n--- Stage 0: Query Expansion ---")
expanded = expand_query(qa, n=NUM_QUERY_EXPANSIONS)
for q in expanded:
    print(f"  - {q}")

print(f"\n--- Stage 1: Hybrid Retrieval (all {len(documents)} chunks, {len(expanded)} query variants) ---")
candidates = hybrid_retrieve(expanded)
print(f"  Retrieved {len(candidates)} chunks")
save_chunks_to_folder("01_hybrid_retrieval", candidates, score_label="rrf_score", base_dir=CHUNK_SAVE_ROOT)

print("\n--- Stage 2: Context Compression (dedup) ---")
compressed = compress_context(candidates)
print(f"  Kept {len(compressed)} / {len(candidates)} chunks after dedup")
save_chunks_to_folder("02_compressed", compressed, score_label="rrf_score", base_dir=CHUNK_SAVE_ROOT)

print(f"\n--- Stage 3: Reranking ({len(compressed)} deduped chunks) ---")
reranked = rerank(qa, compressed)
for rank, (doc, score) in enumerate(reranked, start=1):
    print(f"#{rank} | score={score:.4f} | [{doc.metadata['start_time']} - {doc.metadata['end_time']}] "
          f"{doc.page_content[:80]}...")
save_chunks_to_folder("03_reranked", reranked, score_label="rerank_score", base_dir=CHUNK_SAVE_ROOT)

print(f"\n--- Stage 4: Fill Token Window (budget={CONTEXT_TOKEN_BUDGET} tokens) ---")
windowed = fill_token_window(reranked, budget=CONTEXT_TOKEN_BUDGET)
save_chunks_to_folder("04_token_window", windowed, score_label="rerank_score", base_dir=CHUNK_SAVE_ROOT)

print("\n--- Stage 5: Chronological Reordering ---")
ordered = chronological_order(windowed)
for doc, score in ordered:
    print(f"  [{doc.metadata['start_time']} - {doc.metadata['end_time']}] {doc.page_content[:80]}...")
save_chunks_to_folder("05_final_chronological", ordered, score_label="rerank_score", base_dir=CHUNK_SAVE_ROOT)

print("\n--- Stage 6: Generated Answer ---")
answer = generate_answer(qa, ordered)
print(answer)

if SAVE_CHUNKS_TO_DISK:
    answer_path = os.path.join(CHUNK_SAVE_ROOT, "06_answer.txt")
    with open(answer_path, "w", encoding="utf-8") as f:
        f.write(f"Question: {qa}\n\nAnswer:\n{answer}\n")
    print(f"  -> Saved final answer to {answer_path}")
