import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics_csv(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    metrics = ["Doc_P@K","Doc_R@K","Doc_F1@K","Chunk_P@K","Chunk_R@K","Chunk_F1@K","MRR"]
    for m in metrics:
        plt.figure()
        df.plot(x="run_id", y=m)
        plt.title(m)
        plt.tight_layout()
        plt.savefig(str(Path(out_dir) / f"{m}.png"))
        plt.close()
