# evaluator.py
import json, os, numpy as np
import re
from typing import List, Dict, Tuple


def _normalize_source_id(source_id: str) -> str:
    if not isinstance(source_id, str):
        source_id = str(source_id)
    text = source_id.replace(".json", "")
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[^\w]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")



def _normalize_chunk_id_to_number(value: str) -> str:
    s = str(value or "").strip()
    m = re.search(r"(\d+)\s*$", s) or re.search(r"chunk[-_ ]*(\d+)\s*$", s, re.IGNORECASE)
    return m.group(1) if m else s

def _normalize_sid_cid_numeric(sid_cid: str) -> str:
    txt = str(sid_cid or "")
    if "_" not in txt:
        return txt
    sid_part, cid_part = txt.rsplit("_", 1)
    return f"{_normalize_source_id(sid_part)}_{_normalize_chunk_id_to_number(cid_part)}"


def _add_example(queries, rel_docs, rel_chunks, query: str, gts: list):
    queries.append(query)
    doc_ids, chunk_ids = set(), set()
    for gt in gts or []:
        sid = _normalize_source_id(gt.get("source_id"))
        cid = gt.get("chunk_id")
        if sid:
            doc_ids.add(sid)
            if cid is not None:
                chunk_ids.add(f"{sid}_{_normalize_chunk_id_to_number(cid)}")
    rel_docs[query] = doc_ids
    rel_chunks[query] = chunk_ids


def load_qrels(path: str):
    queries, rel_docs, rel_chunks = [], {}, {}
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                row = json.loads(line)
                q = row["query"]
                queries.append(q)
                # ★ JSONL도 정규화 적용
                docs = {_normalize_source_id(s) for s in row.get("relevant_source_ids", [])}
                chks = set()
                for c in row.get("relevant_chunks", []):
                    # 이미 "sid_cid" 형태면 sid만 정규화 시도
                    if isinstance(c, str) and "_" in c:
                        sid_part, cid_part = c.rsplit("_", 1)
                        chks.add(f"{_normalize_source_id(sid_part)}_{cid_part}")
                    else:
                        chks.add(str(c))
                rel_docs[q] = docs
                rel_chunks[q] = chks
    elif ext == ".json":
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", [])
        for item in data:
            q = item.get("query")
            if not q: continue
            _add_example(queries, rel_docs, rel_chunks, q, item.get("ground_truths", []))
    else:
        raise ValueError(f"Unsupported qrels format: {path}")

    return queries, rel_docs, rel_chunks

def precision_recall_f1(tp, pred, gold):
    p = tp / max(1, pred); r = tp / max(1, gold)
    f1 = 0.0 if p + r == 0 else 2*p*r/(p+r)
    return p, r, f1

def eval_run(run_results: Dict[str, List[Dict]], rel_docs, rel_chunks, K: int = 10):
    per_query = {}
    for q, items in run_results.items():
        topk = items[:K]
        pred_doc_ids_list   = [_normalize_source_id(str(it.get("retrieved_source_id",""))) for it in topk]
        pred_chunk_ids_list = [_normalize_sid_cid_numeric(str(it.get("retrieved_chunk_id",""))) for it in topk]

        pred_doc_ids_set   = set(pred_doc_ids_list)
        pred_chunk_ids_set = set(pred_chunk_ids_list)

        gold_docs   = set(rel_docs.get(q, set()))
        gold_chunks = set(rel_chunks.get(q, set()))

        # ---- Doc-level (중복 제거) ----
        doc_tp_unique = len(pred_doc_ids_set & gold_docs)
        doc_pred_cnt  = len(pred_doc_ids_set)          # 예측 문서의 유니크 개수
        doc_gold_cnt  = len(gold_docs)

        doc_p = doc_tp_unique / max(1, doc_pred_cnt)
        doc_r = doc_tp_unique / max(1, doc_gold_cnt)
        doc_p = min(1.0, doc_p); doc_r = min(1.0, doc_r)
        doc_f1 = 0.0 if doc_p + doc_r == 0 else 2*doc_p*doc_r/(doc_p+doc_r)

        # ---- Chunk-level (일반적으로 포지션 기반이므로 중복 허용 / 집합 교집합 평가 중 택1)
        # 방법 A) 집합기반 정확/재현 (중복 제거) – 문서와 통일성
        chk_tp_unique = len(pred_chunk_ids_set & gold_chunks)
        chk_pred_cnt  = len(pred_chunk_ids_set)
        chk_gold_cnt  = len(gold_chunks)

        chk_p = chk_tp_unique / max(1, chk_pred_cnt)
        chk_r = chk_tp_unique / max(1, chk_gold_cnt)
        chk_p = min(1.0, chk_p); chk_r = min(1.0, chk_r)
        chk_f1 = 0.0 if chk_p + chk_r == 0 else 2*chk_p*chk_r/(chk_p+chk_r)

        # ---- MRR (포지션 기반, 첫 관련 청크의 순위)
        rr = 0.0
        gold_set = gold_chunks
        for rank, cid in enumerate(pred_chunk_ids_list, start=1):
            if cid in gold_set:
                rr = 1.0 / rank
                break

        per_query[q] = {
            "Doc_P@K": doc_p, "Doc_R@K": doc_r, "Doc_F1@K": doc_f1,
            "Chunk_P@K": chk_p, "Chunk_R@K": chk_r, "Chunk_F1@K": chk_f1,
            "MRR": rr
        }

    if not per_query:
        metrics = ["Doc_P@K","Doc_R@K","Doc_F1@K","Chunk_P@K","Chunk_R@K","Chunk_F1@K","MRR"]
        return per_query, {m: 0.0 for m in metrics}

    metrics = ["Doc_P@K","Doc_R@K","Doc_F1@K","Chunk_P@K","Chunk_R@K","Chunk_F1@K","MRR"]
    macro = {m: float(np.mean([v[m] for v in per_query.values()])) for m in metrics}
    return per_query, macro
