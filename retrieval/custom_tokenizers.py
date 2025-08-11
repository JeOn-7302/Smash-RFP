from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List
import copy
import tiktoken

def _pretokenize_with_tiktoken(docs: List[Document], model_name: str = "cl100k_base") -> List[Document]:
    enc = tiktoken.get_encoding(model_name)
    out = []
    for d in docs:
        # 토큰을 공백으로 join하여 BM25에 입력
        tokens = enc.encode(d.page_content)
        token_str = " ".join(map(str, tokens))
        nd = copy.deepcopy(d)
        nd.page_content = token_str
        # 원문은 메타에 보존
        meta = dict(nd.metadata or {})
        meta["_orig_page_content"] = d.page_content[:1024]  # 샘플 보존
        nd.metadata = meta
        out.append(nd)
    return out


def _pretokenize_char_ngrams(docs: List[Document], n: int = 2) -> List[Document]:
    out = []
    for d in docs:
        txt = (d.page_content or "").replace(" ", "")
        grams = ["".join(txt[i:i+n]) for i in range(len(txt)-n+1)]
        nd = copy.deepcopy(d)
        nd.page_content = " ".join(grams)
        nd.metadata = {**(d.metadata or {}), "_orig_page_content": d.page_content[:1024]}
        out.append(nd)
    return out


def transform_query(query: str, option: str) -> str:
    """문서 전처리 옵션과 동일 규칙으로 쿼리도 변환."""
    if option == "tiktoken":
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(query)
        return " ".join(map(str, toks))
    elif option == "charbigram":
        q = query.replace(" ", "")
        grams = ["".join(q[i:i+2]) for i in range(len(q)-1)]
        return " ".join(grams)
    # builtin 은 그대로
    return query


def build_bm25_retriever_with_tokenizer(docs: List[Document], option: str, k: int = 10):
    if option == "builtin":
        retriever = BM25Retriever.from_documents(docs)
    elif option == "tiktoken":
        retriever = BM25Retriever.from_documents(_pretokenize_with_tiktoken(docs))
    elif option == "charbigram":
        retriever = BM25Retriever.from_documents(_pretokenize_char_ngrams(docs, n=2))
    else:
        raise ValueError(f"Unknown tokenizer option: {option}")
    retriever.k = k
    return retriever
