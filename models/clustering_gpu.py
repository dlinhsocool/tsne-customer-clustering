"""CUDA-first clustering backend with CPU fallbacks.

K-Means is implemented with PyTorch CUDA. DBSCAN uses cuML when it is
available and otherwise falls back to the original scikit-learn estimator.
GMM intentionally stays on scikit-learn CPU to preserve numerical behavior
and the existing model/result contract.

All public function names, signatures, and result keys match clustering_cpu.py.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from utils.constants import (
    DEFAULT_PARAMS,
    FEATURE_SPACES,
    NOISE_LABEL,
    RANDOM_STATE,
    SUPPORTS_PREDICT,
)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA khong kha dung cho clustering_gpu.")
    return torch.device("cuda")


def _to_numpy(value: Any, dtype=None) -> np.ndarray:
    """Convert NumPy/CuPy/cuDF/PyTorch results to a host NumPy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=dtype)


def _validate_input(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] < 2 or X_arr.shape[1] < 1:
        raise ValueError("X phai co shape (n_samples >= 2, n_features >= 1).")
    if not np.isfinite(X_arr).all():
        raise ValueError("X chua NaN hoac Inf.")
    return X_arr


def estimate_dbscan_eps(
    X: np.ndarray,
    min_samples: int = 5,
    quantile: float = 90.0,
) -> float:
    """Estimate DBSCAN eps using chunked CUDA k-NN distances.

    The calculation preserves the CPU implementation's neighbor-count
    convention and percentile rule while avoiding a full n x n allocation.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or len(X_arr) < 2:
        return 0.5
    if not 0.0 <= float(quantile) <= 100.0:
        raise ValueError("quantile phai nam trong [0, 100].")
    try:
        device = _require_cuda()
        xt = torch.as_tensor(X_arr, dtype=torch.float64, device=device)
        n = len(X_arr)
        k = max(1, min(int(min_samples) - 1, n - 1))
        # Bound temporary memory to roughly chunk_size * n distances.
        chunk_size = max(1, min(2048, int(8_000_000 // max(n, 1))))
        kth_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                distances = torch.cdist(xt[start:stop], xt, p=2)
                row_ids = torch.arange(start, stop, device=device)
                local_ids = torch.arange(stop - start, device=device)
                distances[local_ids, row_ids] = float("inf")
                nearest = torch.topk(distances, k=k, dim=1, largest=False).values
                kth_parts.append(nearest[:, -1].detach().cpu().numpy())
        dists = np.concatenate(kth_parts)
        eps = float(np.percentile(dists, quantile))
        if not np.isfinite(eps) or eps <= 0:
            positive = dists[dists > 0]
            eps = float(np.median(positive)) if len(positive) else 0.5
        return round(eps, 4)
    except Exception as exc:
        warnings.warn(
            f"Uoc luong DBSCAN eps tren GPU that bai ({exc}); dung ban CPU.",
            RuntimeWarning,
        )
        from .clustering_cpu import estimate_dbscan_eps as _cpu_estimate
        return _cpu_estimate(X_arr, min_samples=min_samples, quantile=quantile)


class _TorchKMeansModel:
    """Small sklearn-like model wrapper returned by the PyTorch K-Means."""
    def __init__(self, centers: np.ndarray, labels: np.ndarray, inertia: float, n_iter: int):
        self.cluster_centers_ = np.asarray(centers, dtype=np.float64)
        self.labels_ = np.asarray(labels, dtype=np.int32)
        self.inertia_ = float(inertia)
        self.n_iter_ = int(n_iter)
        self.n_features_in_ = int(self.cluster_centers_.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = _validate_input(X)
        device = _require_cuda()
        with torch.no_grad():
            xt = torch.as_tensor(X_arr, dtype=torch.float64, device=device)
            centers = torch.as_tensor(self.cluster_centers_, dtype=xt.dtype, device=device)
            labels = torch.cdist(xt, centers).argmin(dim=1)
        return labels.cpu().numpy().astype(int)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        raise AttributeError("Model result da fit; hay dung labels_ hoac predict().")


def _kmeans_plus_plus(Xt: torch.Tensor, n_clusters: int, generator: torch.Generator) -> torch.Tensor:
    n_samples = Xt.shape[0]
    first = int(torch.randint(n_samples, (1,), generator=generator, device=Xt.device).item())
    selected = [first]
    centers = [Xt[first]]
    closest_sq = torch.sum((Xt - Xt[first]) ** 2, dim=1)

    for _ in range(1, n_clusters):
        total = closest_sq.sum()
        if torch.isfinite(total) and float(total.item()) > 0.0:
            probs = closest_sq / total
            next_idx = int(torch.multinomial(probs, 1, generator=generator).item())
        else:
            remaining = [i for i in range(n_samples) if i not in selected]
            if remaining:
                choice = int(torch.randint(len(remaining), (1,), generator=generator, device=Xt.device).item())
                next_idx = remaining[choice]
            else:
                next_idx = selected[-1]
        selected.append(next_idx)
        centers.append(Xt[next_idx])
        candidate_sq = torch.sum((Xt - Xt[next_idx]) ** 2, dim=1)
        closest_sq = torch.minimum(closest_sq, candidate_sq)
    return torch.stack(centers, dim=0)


def _fit_torch_kmeans_once(
    Xt: torch.Tensor,
    n_clusters: int,
    max_iter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator(device=Xt.device)
    generator.manual_seed(int(seed))
    centers = _kmeans_plus_plus(Xt, n_clusters, generator)
    n_samples = Xt.shape[0]
    labels = torch.zeros(n_samples, dtype=torch.long, device=Xt.device)
    previous_inertia = None

    for iteration in range(1, max_iter + 1):
        distances = torch.cdist(Xt, centers, p=2)
        labels = distances.argmin(dim=1)
        min_distances = distances.gather(1, labels[:, None]).squeeze(1)
        counts = torch.bincount(labels, minlength=n_clusters)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, Xt)
        new_centers = sums / counts.clamp_min(1).to(Xt.dtype)[:, None]

        empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
        if empty.numel():
            farthest = torch.argsort(min_distances, descending=True)
            for offset, cluster_idx in enumerate(empty.tolist()):
                sample_idx = int(farthest[offset % len(farthest)].item())
                new_centers[cluster_idx] = Xt[sample_idx]

        shift = torch.max(torch.sum((centers - new_centers) ** 2, dim=1))
        centers = new_centers
        inertia_tensor = torch.sum(min_distances ** 2)
        inertia_value = float(inertia_tensor.item())
        if float(shift.item()) <= 1e-10:
            break
        if previous_inertia is not None and abs(previous_inertia - inertia_value) <= 1e-12 * max(1.0, previous_inertia):
            break
        previous_inertia = inertia_value

    final_distances = torch.cdist(Xt, centers, p=2)
    labels = final_distances.argmin(dim=1)
    inertia = torch.sum(final_distances.gather(1, labels[:, None]).squeeze(1) ** 2)
    return centers, labels, inertia, iteration


def make_clustering_result(
    algorithm: str,
    model: Any,
    labels: np.ndarray,
    params: dict,
    runtime_seconds: float | None = None,
    cluster_centers: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    extra: dict | None = None,
) -> dict:
    """Pack clustering results using exactly the CPU result contract."""
    if algorithm not in SUPPORTS_PREDICT:
        raise ValueError(f"algorithm phai thuoc {list(SUPPORTS_PREDICT)}, nhan '{algorithm}'")
    labels = np.asarray(labels).astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    sizes = {int(u): int(c) for u, c in zip(unique, counts)}
    n_noise = int(sizes.get(NOISE_LABEL, 0))
    n_clusters = int(len([u for u in unique if u != NOISE_LABEL]))
    full_extra = {"inertia": None, "bic": None, "aic": None, "core_sample_indices": None}
    full_extra.update(extra or {})
    return {
        "algorithm": algorithm,
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sizes": sizes,
        "cluster_centers": None if cluster_centers is None else _to_numpy(cluster_centers, np.float64),
        "probabilities": None if probabilities is None else _to_numpy(probabilities, np.float64),
        "params": dict(params),
        "feature_space": None,
        "supports_predict": SUPPORTS_PREDICT[algorithm],
        "runtime_seconds": runtime_seconds,
        "extra": full_extra,
    }


def run_clustering(
    X: np.ndarray,
    algorithm: str,
    params: dict | None = None,
    feature_space: str = "original",
    random_state: int = RANDOM_STATE,
) -> dict:
    """CUDA-first dispatcher for K-Means/DBSCAN and CPU GMM fallback."""
    if algorithm not in DEFAULT_PARAMS:
        raise ValueError(f"algorithm phai thuoc {list(DEFAULT_PARAMS)}, nhan '{algorithm}'")
    normalized_space = "original" if feature_space == "scaled" else feature_space
    if normalized_space not in FEATURE_SPACES:
        raise ValueError(f"feature_space phai thuoc {list(FEATURE_SPACES)}, nhan '{feature_space}'")
    merged = {**DEFAULT_PARAMS[algorithm], **(params or {})}
    if algorithm == "kmeans":
        result = run_kmeans(X, random_state=random_state, **merged)
    elif algorithm == "gmm":
        result = run_gmm(X, random_state=random_state, **merged)
    else:
        if merged.get("eps") is None:
            merged["eps"] = estimate_dbscan_eps(X, min_samples=int(merged.get("min_samples", 5)))
        result = run_dbscan(X, **merged)
    result["feature_space"] = normalized_space
    return result


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 5,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> dict:
    """K-Means using a PyTorch CUDA implementation with CPU fallback."""
    X_arr = _validate_input(X)
    n_clusters = int(n_clusters)
    n_init = int(n_init)
    max_iter = int(max_iter)
    if not 1 <= n_clusters <= len(X_arr):
        raise ValueError("n_clusters phai nam trong [1, n_samples].")
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init va max_iter phai >= 1.")

    t0 = time.perf_counter()
    try:
        device = _require_cuda()
        torch.manual_seed(int(random_state))
        torch.cuda.manual_seed_all(int(random_state))
        Xt = torch.as_tensor(X_arr, dtype=torch.float64, device=device)
        best = None
        for init_index in range(n_init):
            candidate = _fit_torch_kmeans_once(
                Xt, n_clusters, max_iter, int(random_state) + init_index
            )
            if best is None or float(candidate[2].item()) < float(best[2].item()):
                best = candidate
        centers_t, labels_t, inertia_t, n_iter_used = best
        model = _TorchKMeansModel(
            centers_t.detach().cpu().numpy(),
            labels_t.detach().cpu().numpy(),
            float(inertia_t.item()),
            n_iter_used,
        )
        backend_name = "pytorch-cuda"
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        warnings.warn(f"K-Means CUDA that bai ({exc}); chuyen sang scikit-learn CPU.", RuntimeWarning)
        from .clustering_cpu import run_kmeans as _cpu_run_kmeans
        return _cpu_run_kmeans(
            X_arr, n_clusters=n_clusters, n_init=n_init,
            max_iter=max_iter, random_state=random_state,
        )
    runtime = time.perf_counter() - t0
    params = {
        "n_clusters": n_clusters,
        "n_init": n_init,
        "max_iter": max_iter,
        "random_state": int(random_state),
    }
    result = make_clustering_result(
        "kmeans", model, model.labels_, params, runtime,
        cluster_centers=model.cluster_centers_,
        probabilities=None,
        extra={"inertia": float(model.inertia_)},
    )
    return result


def run_gmm(
    X: np.ndarray,
    n_components: int = 5,
    covariance_type: str = "full",
    max_iter: int = 100,
    random_state: int = 42,
) -> dict:
    """Gaussian Mixture Model; kept on sklearn CPU for numerical parity."""
    X_arr = _validate_input(X)
    t0 = time.perf_counter()
    model = GaussianMixture(
        n_components=int(n_components),
        covariance_type=covariance_type,
        max_iter=int(max_iter),
        random_state=int(random_state),
    ).fit(X_arr)
    labels = model.predict(X_arr)
    probs = model.predict_proba(X_arr)
    runtime = time.perf_counter() - t0
    params = {
        "n_components": int(n_components),
        "covariance_type": covariance_type,
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    result = make_clustering_result(
        "gmm", model, labels, params, runtime,
        cluster_centers=model.means_, probabilities=probs,
        extra={"bic": float(model.bic(X_arr)), "aic": float(model.aic(X_arr))},
    )
    return result


def _core_sample_indices_torch(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Calculate DBSCAN core indices in bounded CUDA chunks."""
    device = _require_cuda()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = len(X)
    chunk_size = max(1, min(1024, int(4_000_000 // max(n, 1))))
    core_parts = []
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            d = torch.cdist(xt[start:stop], xt, p=2)
            is_core = torch.sum(d <= float(eps), dim=1) >= int(min_samples)
            core_parts.append(is_core.detach().cpu().numpy())
    return np.flatnonzero(np.concatenate(core_parts)).astype(int)


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> dict:
    """DBSCAN uses cuML when available; otherwise uses sklearn CPU."""
    X_arr = _validate_input(X)
    t0 = time.perf_counter()
    backend_name = "scikit-learn-cpu-fallback"
    model = None
    labels = None
    core_indices = None

    try:
        _require_cuda()
        from cuml.cluster import DBSCAN as CuMLDBSCAN
        # cuML is more predictable with float32 input and lower VRAM use.
        X_gpu = X_arr.astype(np.float32, copy=False)
        model = CuMLDBSCAN(eps=float(eps), min_samples=int(min_samples))
        model.fit(X_gpu)
        labels_value = getattr(model, "labels_", None)
        if labels_value is None:
            labels_value = model.fit_predict(X_gpu)
        labels = _to_numpy(labels_value).astype(int)
        core_value = getattr(model, "core_sample_indices_", None)
        if core_value is not None:
            core_indices = _to_numpy(core_value).astype(int)
        else:
            core_indices = _core_sample_indices_torch(X_arr, eps, min_samples)
        backend_name = "cuml-cuda"
    except ImportError:
        from sklearn.cluster import DBSCAN as SklearnDBSCAN
        model = SklearnDBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(X_arr)
        labels = np.asarray(model.labels_, dtype=int)
        core_indices = np.asarray(model.core_sample_indices_, dtype=int)
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        warnings.warn(f"DBSCAN GPU khong kha dung ({exc}); chuyen sang scikit-learn CPU.", RuntimeWarning)
        from sklearn.cluster import DBSCAN as SklearnDBSCAN
        model = SklearnDBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(X_arr)
        labels = np.asarray(model.labels_, dtype=int)
        core_indices = np.asarray(model.core_sample_indices_, dtype=int)

    runtime = time.perf_counter() - t0
    params = {"eps": float(eps), "min_samples": int(min_samples)}
    result = make_clustering_result(
        "dbscan", model, labels, params, runtime,
        cluster_centers=None, probabilities=None,
        extra={"core_sample_indices": core_indices},
    )
    return result
