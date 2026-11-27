"""models/tsne_benchmark.py - so sánh Exact / Barnes-Hut / FIt-SNE (viết tay) và openTSNE theo kích thước n."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models import tsne_scalable as T
from models.evaluation_cpu import calculate_continuity, calculate_knn_preservation, calculate_trustworthiness


def quality_metrics(X: np.ndarray, Y: np.ndarray, labels: np.ndarray | None, n_eval: int = 3000, seed: int = 0) -> dict:
    """Chất lượng bản đồ trên mẫu con (các thước đo này tốn O(n²))."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_eval, len(X)), replace=False)
    out = {
        "trustworthiness": calculate_trustworthiness(X[idx], Y[idx], 10),
        "continuity": calculate_continuity(X[idx], Y[idx], 10),
        "knn_preservation": calculate_knn_preservation(X[idx], Y[idx], 10),
    }
    if labels is not None:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        k = len(np.unique(labels))
        out["kmeans_ari_on_map"] = float(adjusted_rand_score(labels, KMeans(k, n_init=5, random_state=0).fit_predict(Y)))
    return out


def _run_opentsne(X, method, n_iter, perplexity, seed):
    from openTSNE import TSNE
    t0 = time.perf_counter()
    emb = TSNE(perplexity=perplexity, n_iter=n_iter, negative_gradient_method="bh" if method == "bh" else "fft",
               random_state=seed, n_jobs=-1, verbose=False).fit(X)
    return np.asarray(emb), time.perf_counter() - t0


def benchmark_methods(X, labels=None, sizes=(1000, 3000, 10000, 30000), n_iter=500, perplexity=30.0,
                      exact_max=4000, opentsne_max=30000, seed=0, log=print) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        idx = rng.choice(len(X), n, replace=False) if n < len(X) else np.arange(len(X))
        Xs, ls = X[idx], (None if labels is None else labels[idx])
        P, perp, _ = T.compute_affinities(Xs, perplexity, seed)   # P dùng chung -> so sánh công bằng
        for m in T.METHODS:
            if m == "exact" and n > exact_max:
                continue
            r = T.fit_tsne(Xs, m, perplexity=perp, n_iter=n_iter, P=P, random_state=seed)
            row = {"n": n, "method": f"tự viết · {m}", "runtime_s": r["timing"]["optimize"],
                   "repulsion_s": r["timing"]["repulsion"], "kl": r["kl_divergence"],
                   **quality_metrics(Xs, r["embedding"], ls)}
            rows.append(row); log(row)
        if n <= opentsne_max:
            for m in ("bh", "fft"):
                Y, sec = _run_opentsne(Xs, m, n_iter, perp, seed)
                row = {"n": n, "method": f"openTSNE · {m}", "runtime_s": sec, "repulsion_s": np.nan, "kl": np.nan,
                       **quality_metrics(Xs, Y, ls)}
                rows.append(row); log(row)
    return pd.DataFrame(rows)


def sweep_theta(X, n=3000, thetas=(0.0, 0.2, 0.5, 0.8, 1.2, 2.0), perplexity=30.0, seed=0) -> pd.DataFrame:
    """Sai số gradient & thời gian lực đẩy của BH theo θ, tại một bản đồ trung gian (sau 200 vòng)."""
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    P, _, _ = T.compute_affinities(Xs, perplexity, seed)
    Y = T.fit_tsne(Xs, "barnes_hut", n_iter=200, P=P)["embedding"]
    ge, _ = T.gradient(Y, P, "exact")
    rows = []
    for th in thetas:
        t0 = time.perf_counter(); g, _ = T.gradient(Y, P, "barnes_hut", theta=th); dt = time.perf_counter() - t0
        rows.append({"theta": th, "rel_grad_error": float(np.linalg.norm(g - ge) / np.linalg.norm(ge)), "seconds": dt})
    return pd.DataFrame(rows)


def sweep_interpolation(X, n=3000, points=(1, 2, 3, 4, 5, 7), perplexity=30.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    P, _, _ = T.compute_affinities(Xs, perplexity, seed)
    Y = T.fit_tsne(Xs, "barnes_hut", n_iter=200, P=P)["embedding"]
    ge, _ = T.gradient(Y, P, "exact")
    rows = []
    for p in points:
        t0 = time.perf_counter(); g, _ = T.gradient(Y, P, "fft", n_interpolation_points=p); dt = time.perf_counter() - t0
        rows.append({"n_interpolation_points": p, "rel_grad_error": float(np.linalg.norm(g - ge) / np.linalg.norm(ge)), "seconds": dt})
    return pd.DataFrame(rows)


def repulsion_scaling(X, sizes=(1000, 2000, 5000, 10000, 20000, 50000, 100000), exact_max=10000, seed=0) -> pd.DataFrame:
    """Thời gian MỘT lần tính lực đẩy theo n (không tính kNN) - thể hiện O(n²) / O(n log n) / O(n)."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        if n > len(X):
            continue
        Y = T._initial_embedding(X[rng.choice(len(X), n, replace=False)], "pca", seed) * 1e4 * 0.1
        for m in T.METHODS:
            if m == "exact" and n > exact_max:
                continue
            T.repulsive_forces(Y[:200], m)  # khởi động JIT
            t0 = time.perf_counter(); T.repulsive_forces(Y, m); rows.append({"n": n, "method": m, "seconds": time.perf_counter() - t0})
    return pd.DataFrame(rows)
