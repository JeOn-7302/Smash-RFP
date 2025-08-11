import argparse, json
from pathlib import Path
from itertools import product
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma

from embedding_loader import load_embedding_model
from retrievals import run_retrieve

from evaluator import load_qrels, eval_run
from viz import plot_metrics_csv
from utils import make_run_id, ensure_dirs, write_json, append_csv


# ---- 그리드 정의 ----
def default_grid():
    return {
        "strategies": [
            "bm25", "dense", "hybrid" # "hyde", 
            # 조합 예: HyDE + hybrid (HyDE로 쿼리 확장 + hybrid 전략)
            # ("hybrid", {"hyde": True})
        ],
        "tokenizers": ["builtin", "charbigram"],
        "normalizations": ["min_max", "z_score", "softmax"],
        "use_cross_encoder": [False, True],
        "weights": [  # hybrid 전용 (alpha:content, beta:bm25, gamma:meta)
            {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
            {"alpha": 0.6, "beta": 0.3, "gamma": 0.1},
            {"alpha": 0.4, "beta": 0.4, "gamma": 0.2},
            {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}
        ],
        "embedding_options": [
            {"provider": "huggingface", "model_name": "BAAI/bge-m3"},
            # {"provider": "openai", "model_name": "text-embedding-3-small"}
        ],
        "K": [3, 5, 10]
    }

def run_one_experiment(cfg, queries, rel_docs, rel_chunks, out_base: Path):
    run_id = make_run_id(cfg)
    run_dir = out_base / "retrieved_json" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    emb = load_embedding_model(cfg["model_name"], cfg["provider"])

    run_results = {}
    for q in queries:
        results = run_retrieve(
            query=q,
            strategy=cfg["strategy"],
            normalization_method=cfg["normalization"],
            use_cross_encoder=cfg["use_cross_encoder"],
            embedding_model=emb,
            tokenizer_option=cfg["tokenizer"],
            weights=cfg.get("weights"),
            hyde_config={"enabled": cfg.get("hyde_enabled", False), "mode": "concat", "bm25_topN": 3},
            remove_duplicates = False
        )
        run_results[q] = results
        # 정성평가용 저장
        write_json(str(run_dir / f"{hash(q)}.json"), {
            "query": q,
            "topk": results
        })

    per_query, macro = eval_run(run_results, rel_docs, rel_chunks, K=cfg["K"])

    # 결과 요약 CSV 추가
    csv_row = {
        "run_id": run_id,
        "strategy": cfg["strategy"],
        "tokenizer": cfg["tokenizer"],
        "normalization": cfg["normalization"],
        "use_cross_encoder": cfg["use_cross_encoder"],
        "weights": json.dumps(cfg.get("weights", {}), ensure_ascii=False),
        "provider": cfg["provider"],
        "model_name": cfg["model_name"],
        "K": cfg["K"],
        **macro
    }
    return run_id, csv_row, macro, per_query

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", type=str, default="/home/gcp-JeOn-8/RFP_A/data2/ground_truth.json")
    ap.add_argument("--outdir", type=str, default="experiments/outputs")
    ap.add_argument("--primary_metric", type=str, default="MRR")  # Top-5 선별 기준
    ap.add_argument("--grid", type=str, default="")  # 사용자 정의 그리드 json 경로(옵션)
    args = ap.parse_args()

    queries, rel_docs, rel_chunks = load_qrels(args.qrels)
    out_base = ensure_dirs(args.outdir)

    grid = default_grid() if not args.grid else json.load(open(args.grid,"r",encoding="utf-8"))

    header = ["run_id","strategy","tokenizer","normalization","use_cross_encoder","weights","provider","model_name","K",
              "Doc_P@K","Doc_R@K","Doc_F1@K","Chunk_P@K","Chunk_R@K","Chunk_F1@K","MRR"]
    csv_path = str(Path(args.outdir) / "summary.csv")

    all_runs = []
    # 전략(단일/조합) 전개
    for strategy_entry in grid["strategies"]:
        if isinstance(strategy_entry, tuple):
            strategy, opts = strategy_entry
            hyde_enabled_default = bool(opts.get("hyde"))
        else:
            strategy = strategy_entry
            hyde_enabled_default = (strategy == "hyde")  # HyDE 단독 시 True

        for tokenizer, normalization, use_ce, weights_opt, emb_opt, K in product(
            grid["tokenizers"], grid["normalizations"], grid["use_cross_encoder"],
            grid["weights"], grid["embedding_options"], grid["K"]
        ):
            cfg = {
                "strategy": strategy,
                "hyde_enabled": hyde_enabled_default,
                "tokenizer": tokenizer,
                "normalization": normalization,
                "use_cross_encoder": use_ce,
                "weights": weights_opt if strategy == "hybrid" else {},
                "provider": emb_opt["provider"],
                "model_name": emb_opt["model_name"],
                "K": K
            }
            run_id, row, macro, per_query = run_one_experiment(cfg, queries, rel_docs, rel_chunks, out_base)
            append_csv(csv_path, row, header)
            write_json(str(Path(args.outdir) / "per_query" / f"{run_id}.json"), per_query)
            all_runs.append( (run_id, cfg, macro) )

    # Top-5 추출
    pm = args.primary_metric
    top5 = sorted(all_runs, key=lambda x: x[2][pm], reverse=True)[:5]
    write_json(str(Path(args.outdir) / "top5.json"), [
        {"run_id": rid, "config": cfg, "metrics": mac} for rid, cfg, mac in top5
    ])

    # 그래프 생성
    plot_metrics_csv(csv_path, str(Path(args.outdir) / "charts"))

    print("\n완료. summary.csv / charts/*.png / top5.json / retrieved_json/* / per_query/*.json 확인하세요.")

if __name__ == "__main__":
    # 기본값: qrels, outdir, primary_metric을 미리 지정
    default_qrels = "/home/gcp-JeOn-8/RFP_A/data2/ground_truth.json"
    default_outdir = "/home/gcp-JeOn-8/RFP_A/experiments_results/outputs"
    default_metric = "MRR"

    # CLI 인자 없이 실행하면 기본값 사용
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--qrels", default_qrels,
            "--outdir", default_outdir,
            "--primary_metric", default_metric
        ])

    main()