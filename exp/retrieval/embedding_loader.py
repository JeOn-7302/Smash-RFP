# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings

# def load_embedding_model(model_name: str, provider: str = "huggingface"):
#     if provider == "huggingface":
#         return HuggingFaceEmbeddings(model_name=model_name)
#     elif provider == "openai":
#         return OpenAIEmbeddings(model=model_name)
#     else:
#         raise ValueError(f"Unknown embedding provider: {provider}")

from typing import List
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # Py3.8이면 실패 가능
except Exception:
    HuggingFaceEmbeddings = None

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None

# Fallback: sentence-transformers 기반 간단 어댑터
class SimpleHFEmbeddings:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    # langchain과 동일 시그니처
    def embed_query(self, text: str) -> List[float]:
        import numpy as np
        v = self.model.encode(text, normalize_embeddings=True)
        return v.astype(float).tolist() if hasattr(v, "astype") else v.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vs = self.model.encode(texts, normalize_embeddings=True)
        return [v.astype(float).tolist() if hasattr(v, "astype") else v.tolist() for v in vs]

def load_embedding_model(model_name: str, provider: str = "huggingface"):
    if provider == "huggingface":
        if HuggingFaceEmbeddings is not None:
            return HuggingFaceEmbeddings(model_name=model_name)
        # fallback
        return SimpleHFEmbeddings(model_name=model_name)
    elif provider == "openai":
        if OpenAIEmbeddings is None:
            raise ImportError("langchain-openai가 필요합니다. pip install langchain-openai")
        return OpenAIEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
