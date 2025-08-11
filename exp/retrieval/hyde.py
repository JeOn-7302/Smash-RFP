from typing import List, Dict

def hyde_expand_query(query: str, bm25_docs: List, bm25_chunk_map: Dict, top_n: int = 3, mode: str = "concat") -> str:
    # 간단한 대체: BM25 상위 N개 문서 내용을 합쳐서 가상 문서로 사용
    # 실제 BM25 상위는 retriever.invoke가 필요하지만 여기서는 외부에서 이미 불러온 docs를 사용한다면,
    # 안전하게 상위 top_n 만큼 슬라이싱
    samples = []
    for d in bm25_docs[:top_n]:
        samples.append(d.page_content[:300])
    pseudo = " ".join(samples)
    if mode == "replace":
        return pseudo
    return f"{query}\n{pseudo}"
