# retrieve

# 경로
# import os
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# DATA_DIR = os.path.join(ROOT_DIR, "data2")
# VECTOR_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
# CLEANED_CHUNK_DIR = os.path.join(DATA_DIR, "cleaned_chunks")
# BM25_DOCS_PATH = os.path.join(DATA_DIR, "bm25_docs.pkl")
# BM25_MAP_PATH = os.path.join(DATA_DIR, "bm25_chunk_id_map.json")
# META_EMBEDDING_PATH = os.path.join(DATA_DIR, "meta_embedding_dict.pkl")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]  # 프로젝트 루트 기준 조정
DATA_DIR = BASE_DIR / "data2"

# 당신의 스크립트가 만든 경로들에 맞춤
VECTOR_DB_PATH = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "rfp_documents_bge"   # vectordb.py와 일치

BM25_DOCS_PATH = str(DATA_DIR / "bm25_docs.pkl")
# BM25_MAP_PATH는 더 이상 필수 아님(메타 사용), 남겨두되 사용하지 않음
BM25_MAP_PATH = str(DATA_DIR / "bm25_chunk_id_map.json")

META_EMBEDDING_PATH = str(DATA_DIR / "meta_embedding_dict.pkl")

# 평가 데이터
GROUND_TRUTH_JSON = str(DATA_DIR / "ground_truth.json")
GROUND_TRUTH_QRELS = GROUND_TRUTH_JSON  # 배열 JSON을 그대로 쓰게 설정

TOP_K = 10
HYBRID_ALPHA = 0.6
HYBRID_BETA  = 0.3
HYBRID_GAMMA = 0.1
NORMALIZATION_METHOD = "z_score"
STRATEGY = "hybrid"

COLLECTION_NAME = "rfp_documents"

