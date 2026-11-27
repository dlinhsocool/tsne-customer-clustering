"""Chạy thí nghiệm mở rộng t-SNE và ghi CSV + PNG vào results/tsne_scaling/.

Dùng: python run_tsne_scaling_experiment.py [--quick]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import tsne_scalable as T
from models.data_processing import preprocess_data, scale_data
from models.tsne_benchmark import benchmark_methods, repulsion_scaling, sweep_interpolation, sweep_theta
from utils.constants import RECOMMENDED_FEATURES

OUT = Path("results/tsne_scaling")
SRC = Path("New_Data/simulated/customers_simulated_100k.csv")


def load():
    df = pd.read_csv(SRC)
    labels = pd.factorize(df["nhom"])[0] if "nhom" in df else None
    r = preprocess_data(df, feature_columns=list(RECOMMENDED_FEATURES), drop_duplicates=False)
    X = scale_data(r["X_processed"])["X_scaled"]
    return np.asarray(X, float), labels


def main(quick):
    OUT.mkdir(parents=True, exist_ok=True)
    X, labels = load()
    print("X", X.shape)
    sizes = (1000, 3000) if quick else (1000, 3000, 10000, 30000)
    rs = (1000, 5000, 20000) if quick else (1000, 2000, 5000, 10000, 20000, 50000, 100000)
    rep = repulsion_scaling(X, rs); rep.to_csv(OUT / "repulsion_scaling.csv", index=False)
    th = sweep_theta(X); th.to_csv(OUT / "theta_sweep.csv", index=False)
    ip = sweep_interpolation(X); ip.to_csv(OUT / "interpolation_sweep.csv", index=False)
    bm = benchmark_methods(X, labels, sizes=sizes, n_iter=300 if quick else 500); bm.to_csv(OUT / "benchmark.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m, g in rep.groupby("method"):
        ax.loglog(g.n, g.seconds, "o-", label=m)
    ax.set(xlabel="n", ylabel="giây / 1 lần tính lực đẩy", title="Chi phí lực đẩy theo n"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "repulsion_scaling.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].semilogy(th.theta, th.rel_grad_error.clip(lower=1e-12), "o-"); axs[0].set(xlabel="θ", ylabel="sai số gradient tương đối", title="Barnes-Hut")
    axs[1].semilogy(ip.n_interpolation_points, ip.rel_grad_error, "o-"); axs[1].set(xlabel="số điểm nội suy p", title="FIt-SNE")
    for a in axs: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "gradient_error.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for m, g in bm.groupby("method"):
        axs[0].loglog(g.n, g.runtime_s, "o-", label=m)
        axs[1].plot(g.n, g.trustworthiness, "o-", label=m)
        if "kmeans_ari_on_map" in g: axs[2].plot(g.n, g.kmeans_ari_on_map, "o-", label=m)
    axs[0].set(xlabel="n", ylabel="giây (tối ưu hóa)", title="Thời gian chạy"); axs[1].set(xlabel="n", title="Trustworthiness")
    axs[2].set(xlabel="n", title="ARI của K-Means trên bản đồ"); axs[0].legend(fontsize=7)
    for a in axs: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "benchmark.png", dpi=150); plt.close(fig)

    n_map = 3000 if quick else 10000
    idx = np.random.default_rng(1).choice(len(X), n_map, replace=False)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for a, m in zip(axs, T.METHODS if n_map <= 4000 else ("barnes_hut", "fft")):
        r = T.fit_tsne(X[idx], m, n_iter=500)
        a.scatter(*r["embedding"].T, c=labels[idx], s=2, cmap="tab10"); a.set_title(f"{m} (KL={r['kl_divergence']:.3f}, {r['timing']['optimize']:.0f}s)"); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "maps.png", dpi=150); plt.close(fig)

    big = np.random.default_rng(2).choice(len(X), len(X), replace=False)
    r = T.fit_tsne(X[big], "fft", n_iter=500)
    pd.DataFrame({"n": [len(big)], "seconds": [r["timing"]["total"]], "kl": [r["kl_divergence"]]}).to_csv(OUT / "full_run_fft.csv", index=False)
    fig, a = plt.subplots(figsize=(6, 6)); a.scatter(*r["embedding"].T, c=labels[big], s=.5, cmap="tab10"); a.axis("off")
    a.set_title(f"FIt-SNE tự viết, n={len(big):,}, {r['timing']['total']:.0f}s"); fig.tight_layout(); fig.savefig(OUT / "map_full.png", dpi=150)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); main(ap.parse_args().quick)
