import json, csv, time, hashlib
from pathlib import Path
from typing import List, Dict

def make_run_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return f"{int(time.time())}-{h}"

def ensure_dirs(base_dir: str):
    base = Path(base_dir)
    (base / "per_query").mkdir(parents=True, exist_ok=True)
    (base / "retrieved_json").mkdir(parents=True, exist_ok=True)
    return base

def write_json(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_csv(path: str, row: dict, header_order: list):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if new: w.writeheader()
        w.writerow(row)

def _dedupe_by_chunk(entries: List[Dict], top_k: int) -> List[Dict]:
    """retrieved_chunk_id 기준으로 중복을 제거하며 상위 top_k만 남긴다."""
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