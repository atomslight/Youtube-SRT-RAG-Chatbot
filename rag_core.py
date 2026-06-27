"""
rag_core.py — shared RAG logic for the web server.

All algorithms are identical to main.py. The only structural change is
encapsulating per-video state in VideoSession and shared models in RAGEngine
so the server can handle multiple independent sessions.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Generator

import numpy as np
import pysrt
import yt_dlp
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rank_bm25 import BM25Okapi

# ============================================================
# CONFIG  (identical to main.py)
# ============================================================
EMBED_MODEL     = "dengcao/Qwen3-Embedding-8B:Q4_K_M"
RERANK_MODEL    = "Krakekai/qwen3-reranker-8b"
ANSWER_MODEL    = "gemma4:31b-cloud"
OLLAMA_BASE_URL = "http://localhost:11434"

USE_CROSS_ENCODER_RERANKER = True
CROSS_ENCODER_MODEL        = "BAAI/bge-reranker-v2-m3"

CHUNK_SECONDS              = 75
OVERLAP_SECONDS            = 15
ENABLE_QUERY_EXPANSION     = True
NUM_QUERY_EXPANSIONS       = 5
CONTEXT_TOKEN_BUDGET       = 4000
DEDUP_SIMILARITY_THRESHOLD = 0.92
SAVE_CHUNKS_TO_DISK        = True
CHUNK_SAVE_ROOT            = "retrieved_chunks"

# ============================================================
# UTILS  (identical to main.py)
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
        meta       = doc.metadata
        safe_range = meta["subtitle_index_range"].replace(" ", "")
        fname      = f"chunk_{rank:02d}_{safe_range}.txt"
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


def token_count(text: str) -> int:
    return len(text) // 4


def fill_token_window(reranked_chunks, budget: int = CONTEXT_TOKEN_BUDGET):
    selected, used = [], 0
    for doc, score in reranked_chunks:
        tokens = token_count(doc.page_content)
        if used + tokens <= budget:
            selected.append((doc, score))
            used += tokens
    return selected


def chronological_order(chunks):
    return sorted(chunks, key=lambda x: x[0].metadata["start_seconds"])


def format_context(chunks) -> str:
    return "\n\n---\n\n".join(
        f"[{doc.metadata['start_time']} - {doc.metadata['end_time']}]\n{doc.page_content}"
        for doc, _ in chunks
    )


# ============================================================
# PROMPTS  (identical to main.py)
# ============================================================
QUERY_EXPANSION_PROMPT = (
    "Given the following search query about a video transcript, "
    "generate {n} alternative short search phrases that cover related sub-topics or "
    "phrasings someone might use when discussing the same subject in a tutorial or "
    "walkthrough video. Output ONLY the phrases, one per line, no numbering, no extra text.\n\n"
    "Query: {query}"
)

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

# ============================================================
# CHUNKING  (identical to main.py)
# ============================================================
def make_chunks(subs, video_title: str, video_id: str) -> list[Document]:
    if not subs:
        return []
    total_end = subs[-1].end.ordinal / 1000.0
    documents: list[Document] = []
    chunk_idx = 0
    t = 0.0
    while t < total_end:
        chunk_end = t + CHUNK_SECONDS
        ov_start  = max(0.0, t - OVERLAP_SECONDS) if chunk_idx > 0 else 0.0
        ov_end    = chunk_end + OVERLAP_SECONDS
        group = [
            s for s in subs
            if s.start.ordinal / 1000.0 < ov_end and s.end.ordinal / 1000.0 > ov_start
        ]
        if group:
            text      = " ".join(s.text.replace("\n", " ") for s in group)
            start_sec = group[0].start.ordinal / 1000.0
            end_sec   = group[-1].end.ordinal  / 1000.0
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


# ============================================================
# LANGUAGE DETECTION
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
# SESSION  — per-video state
# ============================================================
@dataclass
class VideoSession:
    video_id:         str
    video_title:      str
    documents:        list
    vector_store:     Any
    bm25:             Any
    chunk_texts:      list
    chunk_embeddings: list


# ============================================================
# RAG ENGINE  — models initialized once, shared across sessions
# ============================================================
class RAGEngine:

    def __init__(self):
        self.embeddings    = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        self.expansion_llm = ChatOllama(model=ANSWER_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
        self.answer_llm    = ChatOllama(model=ANSWER_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
        if USE_CROSS_ENCODER_RERANKER:
            import torch
            from sentence_transformers import CrossEncoder
            self._torch        = torch
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        else:
            self.reranker_llm = ChatOllama(
                model=RERANK_MODEL, base_url=OLLAMA_BASE_URL,
                temperature=0, num_predict=1,
            )

    # ----------------------------------------------------------
    # Video processing  (generator → SSE-ready dicts)
    # ----------------------------------------------------------
    def process_video(self, url: str) -> Generator[dict, None, None]:
        yield {"type": "stage", "stage": "lang_detect", "msg": "Detecting subtitle language..."}
        lang = detect_subtitle_lang(url)

        yield {"type": "stage", "stage": "download", "msg": f"Downloading subtitles [{lang.upper()}]..."}
        ydl_opts = {
            "skip_download":     True,
            "writesubtitles":    True,
            "writeautomaticsub": True,
            "subtitleslangs":    [lang],
            "subtitlesformat":   "srt",
            "outtmpl":           "%(id)s",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        video_id    = info["id"]
        video_title = info.get("title", "Unknown Title")
        srt_file    = f"{video_id}.{lang}.srt"
        subs        = pysrt.open(srt_file)

        yield {"type": "stage", "stage": "subs_loaded",
               "msg": f"Loaded {len(subs)} subtitle lines"}

        documents = make_chunks(subs, video_title, video_id)
        yield {"type": "stage", "stage": "chunking",
               "msg": f"Created {len(documents)} chunks ({CHUNK_SECONDS}s / {OVERLAP_SECONDS}s overlap)"}

        yield {"type": "stage", "stage": "embedding",
               "msg": f"Building vector index for {len(documents)} chunks — may take a few minutes..."}
        vector_store = InMemoryVectorStore(self.embeddings)
        vector_store.add_documents(documents)

        chunk_texts      = [d.page_content for d in documents]
        chunk_embeddings = self.embeddings.embed_documents(chunk_texts)

        yield {"type": "stage", "stage": "indexing", "msg": "Building BM25 keyword index..."}
        tokenized_corpus = [re.findall(r"\w+", d.page_content.lower()) for d in documents]
        bm25             = BM25Okapi(tokenized_corpus)

        session = VideoSession(
            video_id=video_id,
            video_title=video_title,
            documents=documents,
            vector_store=vector_store,
            bm25=bm25,
            chunk_texts=chunk_texts,
            chunk_embeddings=chunk_embeddings,
        )
        yield {"type": "session", "session": session}

    # ----------------------------------------------------------
    # Internal helpers  (same logic as main.py, session-scoped)
    # ----------------------------------------------------------
    def _get_chunk_embedding(self, session: VideoSession, doc):
        return session.chunk_embeddings[session.chunk_texts.index(doc.page_content)]

    def _hybrid_retrieve_single(self, session: VideoSession, query: str):
        n            = len(session.documents)
        dense_ranked = [doc for doc, _ in session.vector_store.similarity_search_with_score(query, k=n)]

        bm25_scores     = session.bm25.get_scores(re.findall(r"\w+", query.lower()))
        bm25_ranked_idx = np.argsort(bm25_scores)[::-1]
        bm25_ranked     = [session.documents[i] for i in bm25_ranked_idx]

        k_rrf = 60; rrf_scores = {}; doc_lookup = {}
        for rank, doc in enumerate(dense_ranked):
            key = doc.metadata["subtitle_index_range"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k_rrf + rank + 1)
            doc_lookup[key] = doc
        for rank, doc in enumerate(bm25_ranked):
            key = doc.metadata["subtitle_index_range"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k_rrf + rank + 1)
            doc_lookup[key] = doc
        return rrf_scores, doc_lookup

    def _hybrid_retrieve(self, session: VideoSession, queries):
        if isinstance(queries, str):
            queries = [queries]
        combined_rrf = {}; combined_lookup = {}
        for q in queries:
            rrf_scores, doc_lookup = self._hybrid_retrieve_single(session, q)
            for key, score in rrf_scores.items():
                combined_rrf[key] = combined_rrf.get(key, 0) + score
            combined_lookup.update(doc_lookup)
        merged = sorted(combined_lookup.items(), key=lambda x: combined_rrf[x[0]], reverse=True)
        return [(doc, combined_rrf[key]) for key, doc in merged]

    def _expand_query(self, query: str, n: int = NUM_QUERY_EXPANSIONS) -> list[str]:
        if not ENABLE_QUERY_EXPANSION:
            return [query]
        response   = self.expansion_llm.invoke(QUERY_EXPANSION_PROMPT.format(query=query, n=n))
        lines      = [line.strip("-* \t") for line in response.content.strip().split("\n")]
        expansions = [line for line in lines if line]
        return [query] + expansions[:n]

    def _rerank(self, query: str, candidates):
        if USE_CROSS_ENCODER_RERANKER:
            pairs  = [(query, doc.page_content[:1000]) for doc, _ in candidates]
            scores = self.cross_encoder.predict(pairs, activation_fn=self._torch.nn.Sigmoid())
            scored = sorted(zip([doc for doc, _ in candidates], scores),
                            key=lambda x: x[1], reverse=True)
            return [(doc, float(s)) for doc, s in scored]
        else:
            RERANK_PROMPT = (
                'Judge whether the Passage is relevant to the Query. Answer only "yes" or "no".\n\n'
                "<Query>: {query}\n<Passage>: {passage}"
            )
            scored = []
            for doc, rrf_score in candidates:
                response  = self.reranker_llm.invoke(
                    RERANK_PROMPT.format(query=query, passage=doc.page_content[:1000])
                )
                relevance = 1 if response.content.strip().lower().startswith("y") else 0
                scored.append((doc, relevance + rrf_score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [(doc, score) for doc, score in scored]

    def _compress_context(self, session: VideoSession, candidates):
        kept, kept_embs = [], []
        for doc, score in candidates:
            emb = self._get_chunk_embedding(session, doc)
            if not any(cosine_sim(emb, ke) > DEDUP_SIMILARITY_THRESHOLD for ke in kept_embs):
                kept.append((doc, score))
                kept_embs.append(emb)
        return kept

    # ----------------------------------------------------------
    # Query answering  (generator → SSE-ready dicts + tokens)
    # ----------------------------------------------------------
    def answer_query(self, session: VideoSession, query: str) -> Generator[dict, None, None]:
        yield {"type": "stage", "msg": "Expanding query..."}
        expanded = self._expand_query(query)

        yield {"type": "stage", "msg": f"Searching {len(session.documents)} chunks..."}
        candidates = self._hybrid_retrieve(session, expanded)

        yield {"type": "stage", "msg": "Deduplicating similar chunks..."}
        compressed = self._compress_context(session, candidates)

        yield {"type": "stage", "msg": f"Reranking {len(compressed)} candidates..."}
        reranked = self._rerank(query, compressed)

        windowed = fill_token_window(reranked)
        ordered  = chronological_order(windowed)

        yield {
            "type": "sources",
            "chunks": [
                {
                    "start_time":    doc.metadata["start_time"],
                    "end_time":      doc.metadata["end_time"],
                    "start_seconds": int(doc.metadata["start_seconds"]),
                    "video_id":      doc.metadata["video_id"],
                }
                for doc, _ in ordered
            ],
        }

        yield {"type": "stage", "msg": "Generating answer..."}
        prompt = ANSWER_PROMPT.format(context=format_context(ordered), question=query)

        for chunk in self.answer_llm.stream(prompt):
            if chunk.content:
                yield {"type": "token", "text": chunk.content}

        yield {"type": "done"}
