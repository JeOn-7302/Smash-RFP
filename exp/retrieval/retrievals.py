import os
import re
import time
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
from scipy.special import softmax

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    VECTOR_DB_PATH, BM25_DOCS_PATH, BM25_MAP_PATH,
    META_EMBEDDING_PATH, COLLECTION_NAME, TOP_K,
    HYBRID_ALPHA, HYBRID_BETA, HYBRID_GAMMA,
    NORMALIZATION_METHOD, STRATEGY
)

from embedding_loader import load_embedding_model

from custom_tokenizers import build_bm25_retriever_with_tokenizer, transform_query
from hyde import hyde_expand_query

# --- 유틸 함수 ---
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def try_int(value: str):
    try:
        return int(value)
    except:
        return value

def normalize_source_id(source_id: str) -> str:
    if not isinstance(source_id, str):
        source_id = str(source_id)
    text = source_id.replace(".json", "")
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[^\w]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_chunk_id_to_number(value) -> str:
    s = str(value or "").strip()
    m = re.search(r"(\d+)\s*$", s) or re.search(r"chunk[-_ ]*(\d+)\s*$", s, re.IGNORECASE)
    return m.group(1) if m else s


def normalize_scores(score_dict, method="z_score"):
    values = np.array(list(score_dict.values()))
    keys = list(score_dict.keys())
    if len(values) == 0: return {k: 0.0 for k in keys}
    if method == "z_score":
        mean, std = values.mean(), values.std()
        return {k: 0.0 if std == 0 else (v - mean) / std for k, v in score_dict.items()}
    elif method == "min_max":
        min_v, max_v = values.min(), values.max()
        return {k: 0.0 if max_v - min_v == 0 else (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}
    elif method == "softmax":
        sm = softmax(values)
        return {k: float(v) for k, v in zip(keys, sm)}
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _dedupe_by_chunk(entries: List[Dict], top_k: int) -> List[Dict]:
    """retrieved_chunk_id 기준으로 중복을 제거하며 상위 top_k만 반환"""
    seen = set()
    out = []
    for e in entries:
        cid = e.get("retrieved_chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(e)
        if len(out) >= top_k:
            break
    return out

# --- 검색 초기화 ---
def load_vectorstore(embedding_model):
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

def load_bm25_documents():
    return load_pickle(BM25_DOCS_PATH)

# def init_bm25_retriever(docs): 
#     retriever = BM25Retriever.from_documents(docs)
#     retriever.k = TOP_K
#     return retriever
def init_bm25_retriever(docs, tokenizer_option: str = "charbigram"):
    """
    tokenizer_option: "builtin" | "tiktoken" | "charbigram"
    """
    return build_bm25_retriever_with_tokenizer(docs, tokenizer_option, k=TOP_K)


def _resolve_chunk_id(doc, chunk_id_map: dict | None):
    meta = getattr(doc, "metadata", {}) or {}
    raw = meta.get("chunk_id", None)
    if raw is None:
        # Chroma 내부 id에 규칙이 있다면 백업 복원
        did = getattr(doc, "id", "") or ""
        raw = normalize_chunk_id_to_number(did)
    return normalize_chunk_id_to_number(raw)

# --- 문서 정보 ---
def extract_doc_info(doc, chunk_id_map=None, scores=None):
    meta = getattr(doc, "metadata", {}) or {}
    source_id = normalize_source_id(meta.get("source_id", "unknown"))
    resolved_chunk_id = _resolve_chunk_id(doc, chunk_id_map)
    full_chunk_id = f"{source_id}_{resolved_chunk_id}"

    # ★ 원문이 있으면 원문, 없으면 현재 page_content
    raw_content = getattr(doc, "page_content", "") or ""
    display_content = (meta.get("_orig_page_content") or raw_content)[:1000]

    info = {
        "retrieved_source_id": source_id,
        "retrieved_chunk_id": full_chunk_id,
        "retrieved_content": display_content,
    }
    if scores: info.update(scores)
    return info


# hybrid 내부에서 사용할 고유 키 생성기 (doc.id 대신 사용)
def _doc_key(doc):
    sid = normalize_source_id(doc.metadata.get("source_id", ""))
    cid = try_int(doc.metadata.get("chunk_id", "unknown"))
    return f"{sid}_{cid}"


# --- 단일 전략 검색 ---
def retrieve_documents(
    retriever, queries, use_rerank=False, rerank_model=None, chunk_id_map=None,
    remove_duplicates: bool = False,
    tokenizer_option: str = "builtin"
):
    results = []
    for query in queries:
        q_for_bm25 = transform_query(query, tokenizer_option)
        start = time.perf_counter()
        docs = retriever.invoke(q_for_bm25)
        end = time.perf_counter()

        entries = []
        if use_rerank and rerank_model:
            contents = [doc.page_content[:512] for doc in docs]
            doc_infos = [extract_doc_info(doc, chunk_id_map) for doc in docs]
            scores = rerank_model.predict([(query, c) for c in contents])
            ranked = sorted(zip(doc_infos, contents, scores), key=lambda x: x[2], reverse=True)

            # 우선 충분히 모은 뒤 중복 제거 적용
            for doc_info, content, score in ranked:
                doc_info["retrieved_content"] = content
                doc_info["rerank_score"] = round(score, 4)
                entries.append(doc_info)

            # ★ 중복 제거 옵션 적용
            if remove_duplicates:
                entries = _dedupe_by_chunk(entries, TOP_K)
            else:
                entries = entries[:TOP_K]

        else:
            # rerank 미사용 경로
            for doc in docs:
                entry = extract_doc_info(doc, chunk_id_map)
                entry["dense_score"] = 1.0  # placeholder
                entries.append(entry)

            # ★ 중복 제거 옵션 적용
            if remove_duplicates:
                entries = _dedupe_by_chunk(entries, TOP_K)
            else:
                entries = entries[:TOP_K]

        results.append({"query": query, "results": entries, "latency_sec": round(end - start, 4)})
    return results



# --- Hybrid 검색 ---

# def hybrid_retrieve(
#     query, embedding_model, normalization_method=NORMALIZATION_METHOD,
#     alpha: float | None = None, beta: float | None = None, gamma: float | None = None,
#     tokenizer_option: str = "builtin"
# ):
#     bm25_docs = load_bm25_documents()
#     bm25_retriever = init_bm25_retriever(bm25_docs, tokenizer_option=tokenizer_option)
#     dense_retriever = load_vectorstore(embedding_model).as_retriever(search_kwargs={"k": TOP_K})
#     bm25_chunk_map = load_json(BM25_MAP_PATH)
#     dense_chunk_map = load_json(BM25_MAP_PATH)  # dense 전용이 생기면 교체

#     meta_embedding_dict = load_pickle(META_EMBEDDING_PATH)

#     # 가중치 주입(없으면 기존 상수 사용)
#     a = HYBRID_ALPHA if alpha is None else alpha
#     b = HYBRID_BETA  if beta  is None else beta
#     g = HYBRID_GAMMA if gamma is None else gamma

#     query_embedding = embedding_model.embed_query(query)
#     bm25_results = bm25_retriever.invoke(query)
#     dense_results = dense_retriever.invoke(query)

#     doc_scores = {}
#     all_docs = {}

#     # BM25
#     for i, doc in enumerate(bm25_results):
#         key = _doc_key(doc)
#         doc_scores.setdefault(key, {})["bm25"] = 1 / (i + 1)
#         all_docs[key] = (doc, bm25_chunk_map)

#     # Dense (content_score)
#     dense_embeddings = embedding_model.embed_documents([d.page_content for d in dense_results])
#     for doc, emb in zip(dense_results, dense_embeddings):
#         key = _doc_key(doc)
#         sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)
#         doc_scores.setdefault(key, {})["dense"] = sim
#         all_docs[key] = (doc, dense_chunk_map)

#     # Meta
#     for key in list(doc_scores.keys()):
#         doc, _ = all_docs[key]
#         sid = normalize_source_id(doc.metadata.get("source_id", ""))
#         cid = try_int(doc.metadata.get("chunk_id", "unknown"))
#         meta_vec = meta_embedding_dict.get(f"{sid}_{cid}")
#         if meta_vec is not None:
#             sim = np.dot(query_embedding, meta_vec) / (np.linalg.norm(query_embedding) * np.linalg.norm(meta_vec) + 1e-8)
#             doc_scores[key]["meta"] = sim

#     # 정규화
#     bm25_norm = normalize_scores({k: v.get("bm25", 0.0) for k, v in doc_scores.items()}, normalization_method)
#     dense_norm = normalize_scores({k: v.get("dense", 0.0) for k, v in doc_scores.items()}, normalization_method)
#     meta_norm  = normalize_scores({k: v.get("meta",  0.0) for k, v in doc_scores.items()}, normalization_method)

#     # 결합
#     combined = {
#         key: a * dense_norm.get(key, 0.0) + b * bm25_norm.get(key, 0.0) + g * meta_norm.get(key, 0.0)
#         for key in doc_scores
#     }

#     sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
#     return [
#         extract_doc_info(all_docs[key][0], all_docs[key][1], {
#             "content_score": dense_norm.get(key, 0.0),
#             "bm25_score": bm25_norm.get(key, 0.0),
#             "meta_score":  meta_norm.get(key, 0.0),
#             "hybrid_score": combined.get(key, 0.0)
#         }) for key, _ in sorted_docs[:TOP_K]
#     ]


def hybrid_retrieve(
    query, embedding_model, normalization_method=NORMALIZATION_METHOD,
    alpha: float | None = None, beta: float | None = None, gamma: float | None = None,
    tokenizer_option: str = "builtin",
    remove_duplicates: bool = False   # ★ 추가
):
    bm25_docs = load_bm25_documents()
    bm25_retriever = init_bm25_retriever(bm25_docs, tokenizer_option=tokenizer_option)
    dense_retriever = load_vectorstore(embedding_model).as_retriever(search_kwargs={"k": TOP_K})

    # 메타 임베딩은 당신의 mete_embedding_generate_A.py 규칙과 정합
    meta_embedding_dict = load_pickle(META_EMBEDDING_PATH)

    a = HYBRID_ALPHA if alpha is None else alpha
    b = HYBRID_BETA  if beta  is None else beta
    g = HYBRID_GAMMA if gamma is None else gamma

    query_embedding = embedding_model.embed_query(query)
    bm25_results = bm25_retriever.invoke(query)
    dense_results = dense_retriever.invoke(query)

    doc_scores = {}
    all_docs = {}

    # BM25: 순위 기반 점수
    for i, doc in enumerate(bm25_results):
        key = _doc_key(doc)  # source_id + chunk_id 기반
        doc_scores.setdefault(key, {})["bm25"] = 1 / (i + 1)
        all_docs[key] = (doc, None)

    # Dense: 코사인 유사도
    dense_embeddings = embedding_model.embed_documents([d.page_content for d in dense_results])
    for doc, emb in zip(dense_results, dense_embeddings):
        key = _doc_key(doc)
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)
        doc_scores.setdefault(key, {})["dense"] = sim
        all_docs[key] = (doc, None)

    # Meta: key = "{normalized_source_id}_{chunk_id}"
    for key in list(doc_scores.keys()):
        doc, _ = all_docs[key]
        sid = normalize_source_id(doc.metadata.get("source_id", ""))
        cid = _resolve_chunk_id(doc, None)
        meta_vec = meta_embedding_dict.get(f"{sid}_{cid}")
        if meta_vec is not None:
            sim = np.dot(query_embedding, meta_vec) / (np.linalg.norm(query_embedding) * np.linalg.norm(meta_vec) + 1e-8)
            doc_scores[key]["meta"] = sim

    # 정규화 & 결합
    bm25_norm = normalize_scores({k: v.get("bm25", 0.0) for k, v in doc_scores.items()}, normalization_method)
    dense_norm = normalize_scores({k: v.get("dense", 0.0) for k, v in doc_scores.items()}, normalization_method)
    meta_norm  = normalize_scores({k: v.get("meta",  0.0) for k, v in doc_scores.items()}, normalization_method)
    combined = {k: a*dense_norm.get(k,0.0) + b*bm25_norm.get(k,0.0) + g*meta_norm.get(k,0.0) for k in doc_scores}

    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    entries = [
        extract_doc_info(all_docs[key][0], None, {
            "content_score": dense_norm.get(key, 0.0),
            "bm25_score": bm25_norm.get(key, 0.0),
            "meta_score":  meta_norm.get(key, 0.0),
            "hybrid_score": combined.get(key, 0.0)
        })
        for key, _ in sorted_docs
    ]

    # ★ 중복 제거 옵션 적용
    if remove_duplicates:
        entries = _dedupe_by_chunk(entries, TOP_K)
    else:
        entries = entries[:TOP_K]

    return entries


# --- 최종 실행 함수 ---
# def run_retrieve(query=None, strategy=STRATEGY, normalization_method=NORMALIZATION_METHOD, 
#                  use_cross_encoder=True, embedding_model=None):
#     if embedding_model is None:
#         raise ValueError("embedding_model must be provided to run_retrieve().")

#     print(f"\nRetrieval 시작 → 전략: {strategy}, 정규화: {normalization_method}, rerank: {use_cross_encoder}")
    
#     results = None

#     if strategy == "bm25":
#         retriever = init_bm25_retriever(load_bm25_documents())
#         results = retrieve_documents(
#             retriever, [query],
#             use_rerank=use_cross_encoder,
#             rerank_model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_cross_encoder else None,
#             chunk_id_map=load_json(BM25_MAP_PATH)
#         )
        
#     elif strategy == "dense":
#         retriever = load_vectorstore(embedding_model).as_retriever(search_kwargs={"k": TOP_K})
#         results = retrieve_documents(
#             retriever, [query],
#             use_rerank=use_cross_encoder,
#             rerank_model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_cross_encoder else None,
#             chunk_id_map=load_json(BM25_MAP_PATH)
#         )
        
#     elif strategy == "hybrid":
#         entries = hybrid_retrieve(query, embedding_model, normalization_method)

#         if use_cross_encoder and entries:
#             cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#             pairs = [(query, e["retrieved_content"][:512]) for e in entries]
#             scores = cross_encoder.predict(pairs)
#             ranked = sorted(zip(entries, scores), key=lambda x: x[1], reverse=True)[:TOP_K]
#             entries = [e for e, _ in ranked]
#         results = [{"query": query, "results": entries}]
        
#     else:
#         raise ValueError(f"Unknown strategy: {strategy}")


#     # contents =[]
#     # print(f"\n검색 결과 (Top-{TOP_K}):")
#     # for i, r in enumerate(results[0]["results"]):
#     #     m = r.get("metadata", {}) or {}
#     #     print(f"\n⚡ Top {i+1}:\n{r['retrieved_content'].strip()}")
#     #     contents.append({
#     #         "retrieved_content": r.get("retrieval_content", ""),
#     #         "retrieved_source_id": (m.get("source_id") or m.get("retrieved_source_id")),
#     #         "retrieved_chunk_id": m.get("chunk_id") or m.get("retrieved_chunk_id")
#     #     })
#     # return contents
    
#     print(f"\n검색 결과 (Top-{TOP_K}):")
#     for i, r in enumerate(results[0]["results"]):
#         print(f"\n⚡ Top {i+1}:\n{r['retrieved_content'].strip()}")
    
    
#     result_list = []
#     for i, r in enumerate(results[0]["results"]):
#         result_list.append({
#             "retrieved_content": r.get("retrieved_content",""),
#             "retrieved_source_id": r.get("retrieved_source_id",),
#             "retrieved_chunk_id": r.get("retrieved_chunk_id")
#         })
#     return result_list

def run_retrieve(
    query=None, strategy=STRATEGY, normalization_method=NORMALIZATION_METHOD, 
    use_cross_encoder=True, embedding_model=None,
    tokenizer_option: str = "builtin",
    weights: dict | None = None,
    hyde_config: dict | None = None,
    remove_duplicates: bool = False
):
    print("[DENSE-CHECK] model:", type(embedding_model).__name__)
    try:
        print("[DENSE-CHECK] encode_kwargs:", getattr(embedding_model, "encode_kwargs", None))
    except Exception:
        pass

    if embedding_model is None:
        raise ValueError("embedding_model must be provided to run_retrieve().")

    print(f"\nRetrieval 시작 → 전략: {strategy}, 정규화: {normalization_method}, rerank: {use_cross_encoder}, tokenizer: {tokenizer_option}")

    # HyDE 사전 처리: 쿼리 확장
    original_query = query
    if hyde_config and hyde_config.get("enabled"):
        query = hyde_expand_query(
            query=original_query,
            bm25_docs=load_bm25_documents(),
            bm25_chunk_map=load_json(BM25_MAP_PATH),
            top_n=hyde_config.get("bm25_topN", 3),
            mode=hyde_config.get("mode", "concat")  # "concat" | "replace"
        )

    results = None
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_cross_encoder else None
    bm25_map = load_json(BM25_MAP_PATH)

    if strategy == "bm25":
        retriever = init_bm25_retriever(load_bm25_documents(), tokenizer_option=tokenizer_option)
        results = retrieve_documents(
            retriever, [query],
            use_rerank=use_cross_encoder,
            rerank_model=cross_encoder,
            chunk_id_map=None,
            remove_duplicates=remove_duplicates
        )
    elif strategy == "dense":
        retriever = load_vectorstore(embedding_model).as_retriever(search_kwargs={"k": TOP_K})
        results = retrieve_documents(
            retriever, [query],
            use_rerank=use_cross_encoder,
            rerank_model=cross_encoder,
            chunk_id_map=None,
            remove_duplicates=remove_duplicates
        )
    elif strategy == "hybrid":
        alpha = (weights or {}).get("alpha")
        beta  = (weights or {}).get("beta")
        gamma = (weights or {}).get("gamma")
        entries = hybrid_retrieve(
            query, embedding_model, normalization_method,
            alpha=alpha, beta=beta, gamma=gamma,
            tokenizer_option=tokenizer_option,
            remove_duplicates=remove_duplicates
        )
        if use_cross_encoder and entries:
            pairs = [(query, e["retrieved_content"][:512]) for e in entries]
            scores = cross_encoder.predict(pairs)
            ranked = sorted(zip(entries, scores), key=lambda x: x[1], reverse=True)
            entries = [e for e, _ in ranked]


            if remove_duplicates:
                entries = _dedupe_by_chunk(entries, TOP_K)
            else:
                entries = entries[:TOP_K]

        results = [{"query": query, "results": entries}]
    elif strategy == "hyde":
        retriever = load_vectorstore(embedding_model).as_retriever(search_kwargs={"k": TOP_K})
        results = retrieve_documents(
            retriever, [query],
            use_rerank=use_cross_encoder,
            rerank_model=cross_encoder,
            chunk_id_map=None,
            remove_duplicates=remove_duplicates
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"\n검색 결과 (Top-{TOP_K}):")
    for i, r in enumerate(results[0]["results"]):
        print(f"\n⚡ Top {i+1}:\n{r['retrieved_content'].strip()}")

    result_list = []
    for r in results[0]["results"]:
        result_list.append({
            "retrieved_content": r.get("retrieved_content",""),
            "retrieved_source_id": r.get("retrieved_source_id",),
            "retrieved_chunk_id": r.get("retrieved_chunk_id")
        })
        
        
    for x in result_list[:5]:
        print("[check]", x["retrieved_source_id"], x["retrieved_chunk_id"])

    return result_list