"""models/tsne_scalable.py - t-SNE mở rộng: Exact, Barnes-Hut và FIt-SNE (FFT) viết tay.

Cả ba phương pháp dùng CHUNG: ma trận xác suất P thưa (k láng giềng gần nhất), khởi tạo PCA và bộ tối ưu
(momentum + adaptive gains + early exaggeration). Chúng chỉ khác nhau ở cách tính LỰC ĐẨY, nên mọi khác biệt
đo được về thời gian/chất lượng đến từ phép xấp xỉ chứ không phải từ bộ tối ưu.

Toán học (q̃_ij = 1 / (1 + ||y_i - y_j||²), Z = Σ_{k≠l} q̃_kl):
    ∂KL/∂y_i = 4 [ Σ_j p_ij q̃_ij (y_i - y_j)  -  (1/Z) Σ_j q̃_ij² (y_i - y_j) ]
                   └──── lực hút: P thưa, O(n·k) ────┘   └──── lực đẩy: n² cặp, cần xấp xỉ ────┘

    exact       : tính mọi cặp, O(n²).
    barnes_hut  : quadtree; một ô đủ xa (độ rộng / khoảng cách < θ) được thay bằng khối tâm, O(n log n).
    fft         : FIt-SNE. Nội suy Lagrange bậc p lên lưới đều rồi tích chập hạt nhân bằng FFT, O(n + G² log G).
                  Dùng 4 kênh điện tích: {1, y_x, y_y} với hạt nhân q̃², và {1} với hạt nhân q̃ (cho Z).

Chỉ hỗ trợ nhúng 2 chiều (quadtree và lưới 2D).
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from scipy import fft as sfft
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

METHODS = ("exact", "barnes_hut", "fft")
_FFT_MAX_INTERVALS = 800  # chặn kích thước lưới để không tràn bộ nhớ khi bản đồ quá rộng


# ---------------------------------------------------------------------------
# 1. Ma trận xác suất P thưa
# ---------------------------------------------------------------------------
def _knn(X: np.ndarray, k: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """k láng giềng gần nhất (không gồm chính điểm). n lớn dùng NN-descent (xấp xỉ), n nhỏ dùng chính xác."""
    n = X.shape[0]
    if n <= 20000:
        dist, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors()
        return dist, idx
    from pynndescent import NNDescent  # đi kèm umap-learn

    index = NNDescent(X, n_neighbors=k + 1, random_state=random_state, n_jobs=-1, low_memory=True)
    idx, dist = index.neighbor_graph
    keep = idx != np.arange(n)[:, None]                   # bỏ chính điểm (nếu có trong danh sách)
    order = np.argsort(~keep, axis=1, kind="stable")[:, :k]
    rows = np.arange(n)[:, None]
    return dist[rows, order], idx[rows, order]


def compute_affinities(
    X: np.ndarray, perplexity: float = 30.0, random_state: int = 42
) -> tuple[sp.csr_matrix, float, dict[str, float]]:
    """P đối xứng, thưa, tổng bằng 1. Trả về (P, perplexity thực dùng, thời gian knn/affinity)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]
    perplexity = float(min(perplexity, (n - 1) / 3.0))
    k = int(min(n - 1, 3 * perplexity))
    t0 = time.perf_counter()
    dist, idx = _knn(X, k, random_state)
    t_knn = time.perf_counter() - t0

    t1 = time.perf_counter()
    d2 = dist ** 2
    d2 = d2 - d2[:, :1]                                    # trừ khoảng cách gần nhất: entropy không đổi, tránh underflow
    target = np.log(perplexity)
    beta = np.ones(n)
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    for _ in range(200):                                   # chia đôi độ chính xác β cho từng hàng, vector hóa
        P = np.exp(-d2 * beta[:, None])
        sumP = np.maximum(P.sum(axis=1), 1e-300)
        H = np.log(sumP) + beta * (d2 * P).sum(axis=1) / sumP
        diff = H - target
        if np.abs(diff).max() < 1e-5:
            break
        up = diff > 0                                      # entropy quá cao -> tăng β
        lo = np.where(up, beta, lo)
        hi = np.where(up, hi, beta)
        beta = np.where(up, np.where(np.isinf(hi), beta * 2.0, (beta + hi) / 2.0), (beta + lo) / 2.0)
    P = np.exp(-d2 * beta[:, None])
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-300)

    rows = np.repeat(np.arange(n), k)
    Pm = sp.csr_matrix((P.ravel(), (rows, idx.ravel())), shape=(n, n))
    Pm = (Pm + Pm.T) / (2.0 * n)
    Pm = Pm.tocsr()
    Pm.sum_duplicates()
    Pm.data = Pm.data / Pm.data.sum()
    return Pm, perplexity, {"knn": t_knn, "affinity": time.perf_counter() - t1}


# ---------------------------------------------------------------------------
# 2. Lực hút (dùng chung): chỉ duyệt các cạnh của P thưa
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _attractive(Y, indptr, indices, data, exaggeration, compute_kl):
    n = Y.shape[0]
    attr = np.zeros((n, 2))
    klpart = np.zeros(n)
    for i in prange(n):
        ax = 0.0
        ay = 0.0
        kl = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            dx = Y[i, 0] - Y[j, 0]
            dy = Y[i, 1] - Y[j, 1]
            q = 1.0 / (1.0 + dx * dx + dy * dy)
            w = data[e] * q
            ax += w * dx
            ay += w * dy
            if compute_kl:
                kl += data[e] * np.log(q)
        attr[i, 0] = exaggeration * ax
        attr[i, 1] = exaggeration * ay
        klpart[i] = kl
    return attr, klpart


# ---------------------------------------------------------------------------
# 3a. Lực đẩy EXACT: O(n²), chia khối để tiết kiệm bộ nhớ
# ---------------------------------------------------------------------------
def _repulsion_exact(Y: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, float]:
    n = len(Y)
    rep = np.zeros_like(Y)
    Z = 0.0
    sq = (Y ** 2).sum(axis=1)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d2 = np.maximum(sq[s:e, None] + sq[None, :] - 2.0 * (Y[s:e] @ Y.T), 0.0)
        q = 1.0 / (1.0 + d2)
        q[np.arange(e - s), np.arange(s, e)] = 0.0          # loại j = i
        Z += q.sum()
        q2 = q * q
        rep[s:e] = Y[s:e] * q2.sum(axis=1, keepdims=True) - q2 @ Y
    return rep, float(Z)


# ---------------------------------------------------------------------------
# 3b. Lực đẩy BARNES-HUT: quadtree + tiêu chí θ
# ---------------------------------------------------------------------------
@njit(cache=True)
def _build_quadtree(Y, max_nodes):
    """Xây quadtree bằng cách chèn từng điểm. Mỗi nút lưu: tâm ô, nửa cạnh, tổng tọa độ, số điểm.

    leaf[node] = chỉ số điểm (lá một điểm) | -1 (nút trong) | -2 (lá chứa nhiều điểm trùng tọa độ).
    """
    n = Y.shape[0]
    xmin = Y[0, 0]
    xmax = xmin
    ymin = Y[0, 1]
    ymax = ymin
    for i in range(1, n):
        xmin = min(xmin, Y[i, 0])
        xmax = max(xmax, Y[i, 0])
        ymin = min(ymin, Y[i, 1])
        ymax = max(ymax, Y[i, 1])
    ccx = np.zeros(max_nodes)
    ccy = np.zeros(max_nodes)
    chalf = np.zeros(max_nodes)
    sx = np.zeros(max_nodes)
    sy = np.zeros(max_nodes)
    mass = np.zeros(max_nodes, np.int64)
    leaf = np.full(max_nodes, -1, np.int64)
    child = np.full((max_nodes, 4), -1, np.int64)
    ccx[0] = 0.5 * (xmin + xmax)
    ccy[0] = 0.5 * (ymin + ymax)
    chalf[0] = 0.5 * max(xmax - xmin, ymax - ymin) + 1e-10
    n_nodes = 1
    for i in range(n):
        node = 0
        while True:
            mass[node] += 1
            sx[node] += Y[i, 0]
            sy[node] += Y[i, 1]
            if mass[node] == 1:                              # nút rỗng -> thành lá chứa điểm i
                leaf[node] = i
                break
            lf = leaf[node]
            if lf >= 0:                                      # lá một điểm: đẩy điểm cũ xuống ô con
                if chalf[node] < 1e-9:                       # hai điểm trùng nhau: gộp thành lá nhiều điểm
                    leaf[node] = -2
                    break
                leaf[node] = -1
                q = 0
                if Y[lf, 0] > ccx[node]:
                    q += 1
                if Y[lf, 1] > ccy[node]:
                    q += 2
                if n_nodes >= max_nodes:
                    return -1, ccx, ccy, chalf, sx, sy, mass, leaf, child
                c = n_nodes
                n_nodes += 1
                h = 0.5 * chalf[node]
                ccx[c] = ccx[node] + (h if (q & 1) else -h)
                ccy[c] = ccy[node] + (h if (q & 2) else -h)
                chalf[c] = h
                mass[c] = 1
                sx[c] = Y[lf, 0]
                sy[c] = Y[lf, 1]
                leaf[c] = lf
                child[node, q] = c
            elif lf == -2:
                break
            q = 0
            if Y[i, 0] > ccx[node]:
                q += 1
            if Y[i, 1] > ccy[node]:
                q += 2
            c = child[node, q]
            if c < 0:
                if n_nodes >= max_nodes:
                    return -1, ccx, ccy, chalf, sx, sy, mass, leaf, child
                c = n_nodes
                n_nodes += 1
                h = 0.5 * chalf[node]
                ccx[c] = ccx[node] + (h if (q & 1) else -h)
                ccy[c] = ccy[node] + (h if (q & 2) else -h)
                chalf[c] = h
                child[node, q] = c
            node = c
    return n_nodes, ccx, ccy, chalf, sx, sy, mass, leaf, child


@njit(parallel=True, cache=True)
def _bh_forces(Y, ccx, ccy, chalf, sx, sy, mass, leaf, child, theta2):
    """Với mỗi điểm i, duyệt cây: ô đủ xa (độ rộng² < θ²·d²) được tính bằng khối tâm, ngược lại mở rộng."""
    n = Y.shape[0]
    rep = np.zeros((n, 2))
    zarr = np.zeros(n)
    for i in prange(n):
        stack = np.empty(1024, np.int64)
        yi0 = Y[i, 0]
        yi1 = Y[i, 1]
        fx = 0.0
        fy = 0.0
        zi = 0.0
        stack[0] = 0
        top = 1
        while top > 0:
            top -= 1
            node = stack[top]
            m = mass[node]
            if m == 0:
                continue
            lf = leaf[node]
            if lf == i:
                continue
            dx = yi0 - sx[node] / m
            dy = yi1 - sy[node] / m
            d2 = dx * dx + dy * dy
            if lf != -1:                                     # lá: tính trực tiếp
                m_eff = m
                if lf == -2 and d2 < 1e-18:
                    m_eff = m - 1                            # i nằm trong lá trùng điểm
                if m_eff <= 0:
                    continue
                q = 1.0 / (1.0 + d2)
                zi += m_eff * q
                q2 = q * q * m_eff
                fx += q2 * dx
                fy += q2 * dy
            else:
                h = chalf[node]
                inside = abs(yi0 - ccx[node]) <= h and abs(yi1 - ccy[node]) <= h
                if (not inside) and 4.0 * h * h < theta2 * d2:   # đủ xa: dùng khối tâm
                    q = 1.0 / (1.0 + d2)
                    zi += m * q
                    q2 = q * q * m
                    fx += q2 * dx
                    fy += q2 * dy
                else:
                    for c in range(4):
                        ch = child[node, c]
                        if ch >= 0:
                            stack[top] = ch
                            top += 1
        rep[i, 0] = fx
        rep[i, 1] = fy
        zarr[i] = zi
    return rep, zarr


def _repulsion_bh(Y: np.ndarray, theta: float = 0.5) -> tuple[np.ndarray, float]:
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    max_nodes = 12 * len(Y) + 128
    while True:
        out = _build_quadtree(Y, max_nodes)
        if out[0] >= 0:
            break
        max_nodes *= 2
    _, ccx, ccy, chalf, sx, sy, mass, leaf, child = out
    rep, zarr = _bh_forces(Y, ccx, ccy, chalf, sx, sy, mass, leaf, child, float(theta) ** 2)
    return rep, float(zarr.sum())


# ---------------------------------------------------------------------------
# 3c. Lực đẩy FIt-SNE: nội suy lên lưới + tích chập bằng FFT
# ---------------------------------------------------------------------------
def _lagrange_weights(u: np.ndarray, p: int) -> np.ndarray:
    """Trọng số nội suy Lagrange của p nút đều (r+0.5)/p trong một đoạn đơn vị, tại tọa độ cục bộ u ∈ [0, 1)."""
    nodes = (np.arange(p) + 0.5) / p
    w = np.ones((len(u), p))
    for r in range(p):
        for s in range(p):
            if s != r:
                w[:, r] *= (u - nodes[s]) / (nodes[r] - nodes[s])
    return w


def _repulsion_fft(
    Y: np.ndarray,
    n_interpolation_points: int = 3,
    min_intervals: int = 50,
    intervals_per_integer: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Tính Σ_j q̃_ij² (y_i - y_j) và Z bằng nội suy bậc p lên lưới đều + FFT.

    Σ_j q̃²_ij (y_i - y_j) = y_i · φ₁(i) − φ_y(i), với φ_c(i) = Σ_j K₂(y_i, y_j) · charge_c(j).
    Số hạng j = i triệt tiêu chính xác trong hiệu này; riêng Z phải trừ n (mỗi điểm tự đóng góp q̃_ii = 1).
    """
    n = len(Y)
    p = int(n_interpolation_points)
    lo = Y.min(axis=0)
    span = max(float((Y.max(axis=0) - lo).max()), 1e-12)
    n_int = int(min(_FFT_MAX_INTERVALS, max(int(min_intervals), np.ceil(span / intervals_per_integer))))
    w = span * (1.0 + 1e-9) / n_int                          # độ rộng mỗi đoạn
    t = (Y - lo) / w
    a = np.minimum(np.floor(t).astype(np.int64), n_int - 1)
    u = t - a
    Lx, Ly = _lagrange_weights(u[:, 0], p), _lagrange_weights(u[:, 1], p)
    G = n_int * p
    h = w / p

    pairs = []                                               # (chỉ số nút phẳng, trọng số) cho từng cặp nút (r, s)
    for r in range(p):
        for s in range(p):
            pairs.append(((a[:, 0] * p + r) * G + (a[:, 1] * p + s), Lx[:, r] * Ly[:, s]))

    charges = [np.ones(n), Y[:, 0], Y[:, 1], np.ones(n)]     # 3 kênh cho K₂ và 1 kênh cho K₁ (tính Z)
    grids = []
    for ch in charges:
        g = np.zeros(G * G)
        for idx, wt in pairs:
            g += np.bincount(idx, weights=ch * wt, minlength=G * G)
        grids.append(g.reshape(G, G))

    ofs = np.arange(2 * G)
    ofs = np.where(ofs <= G, ofs, ofs - 2 * G).astype(np.float64) * h   # nhúng tuần hoàn 2G để tích chập tuyến tính
    d2 = ofs[:, None] ** 2 + ofs[None, :] ** 2
    K1 = 1.0 / (1.0 + d2)
    F1, F2 = sfft.rfft2(K1, workers=-1), sfft.rfft2(K1 * K1, workers=-1)

    phis = []
    for c, g in enumerate(grids):
        pad = np.zeros((2 * G, 2 * G))
        pad[:G, :G] = g
        conv = sfft.irfft2(sfft.rfft2(pad, workers=-1) * (F1 if c == 3 else F2), s=(2 * G, 2 * G), workers=-1)
        phis.append(conv[:G, :G].ravel())

    vals = [np.zeros(n) for _ in range(4)]
    for idx, wt in pairs:
        for c in range(4):
            vals[c] += wt * phis[c][idx]
    rep = Y * vals[0][:, None] - np.stack([vals[1], vals[2]], axis=1)
    Z = float(vals[3].sum() - n)
    return rep, Z


def repulsive_forces(Y: np.ndarray, method: str = "barnes_hut", theta: float = 0.5,
                     n_interpolation_points: int = 3) -> tuple[np.ndarray, float]:
    """Trả về (rep, Z) với rep_i = Σ_j q̃_ij² (y_i - y_j) và Z = Σ_{i≠j} q̃_ij."""
    if method == "exact":
        return _repulsion_exact(Y)
    if method == "barnes_hut":
        return _repulsion_bh(Y, theta)
    if method == "fft":
        return _repulsion_fft(Y, n_interpolation_points)
    raise ValueError(f"method phải thuộc {METHODS}, nhận '{method}'.")


def gradient(Y: np.ndarray, P: sp.csr_matrix, method: str = "exact", theta: float = 0.5,
             n_interpolation_points: int = 3, exaggeration: float = 1.0) -> tuple[np.ndarray, float]:
    """Gradient KL đầy đủ (lực hút + lực đẩy) tại Y. Dùng để kiểm tra độ đúng của phép xấp xỉ."""
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    attr, _ = _attractive(Y, P.indptr, P.indices, P.data, float(exaggeration), False)
    rep, Z = repulsive_forces(Y, method, theta, n_interpolation_points)
    return 4.0 * (attr - rep / Z), Z


# ---------------------------------------------------------------------------
# 4. Bộ tối ưu chung
# ---------------------------------------------------------------------------
def _initial_embedding(X: np.ndarray, init: str, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if init == "pca" and X.shape[1] >= 2:
        Y = PCA(n_components=2, random_state=random_state).fit_transform(X)
        return np.ascontiguousarray(Y / max(float(Y[:, 0].std()), 1e-12) * 1e-4)
    return rng.normal(0.0, 1e-4, size=(X.shape[0], 2))


def fit_tsne(
    X: np.ndarray,
    method: str = "barnes_hut",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    learning_rate: float | str = "auto",
    early_exaggeration: float = 12.0,
    exaggeration_iters: int = 250,
    theta: float = 0.5,
    n_interpolation_points: int = 3,
    init: str = "pca",
    random_state: int = 42,
    kl_every: int = 50,
    snapshot_iters: tuple[int, ...] = (),
    P: sp.csr_matrix | None = None,
    callback: Callable[[int, np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """t-SNE 2D với lực đẩy `method`. Trả về dict gồm embedding, KL, lịch sử KL và thời gian từng bước."""
    if method not in METHODS:
        raise ValueError(f"method phải thuộc {METHODS}, nhận '{method}'.")
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 10:
        raise ValueError("X phải là ma trận 2 chiều với ít nhất 10 dòng.")
    if not np.isfinite(X).all():
        raise ValueError("X chứa NaN hoặc Inf.")
    n = X.shape[0]

    t_start = time.perf_counter()
    timing = {"knn": 0.0, "affinity": 0.0}
    if P is None:
        P, perplexity, timing = compute_affinities(X, perplexity, random_state)
    p_log_p = float((P.data * np.log(np.maximum(P.data, 1e-300))).sum())
    lr = max(n / early_exaggeration / 4.0, 50.0) if isinstance(learning_rate, str) else float(learning_rate)

    Y = _initial_embedding(X, init, random_state)
    update = np.zeros_like(Y)
    gains = np.ones_like(Y)
    kl_history: list[tuple[int, float]] = []
    snapshots: dict[int, np.ndarray] = {}
    t_opt = time.perf_counter()
    t_repulse = 0.0

    def step(it: int, want_kl: bool, apply: bool) -> float | None:
        nonlocal Y, update, gains, t_repulse
        exag = early_exaggeration if it < exaggeration_iters else 1.0
        tr = time.perf_counter()
        rep, Z = repulsive_forces(Y, method, theta, n_interpolation_points)
        t_repulse += time.perf_counter() - tr
        attr, klpart = _attractive(Y, P.indptr, P.indices, P.data, float(exag), want_kl)
        kl = (p_log_p - float(klpart.sum()) + np.log(Z)) if want_kl else None
        if apply:
            grad = 4.0 * (attr - rep / Z)
            gains = np.where(np.sign(grad) != np.sign(update), gains + 0.2, gains * 0.8)
            np.maximum(gains, 0.01, out=gains)
            mom = 0.5 if it < exaggeration_iters else 0.8
            update = mom * update - lr * gains * grad
            Y = Y + update
            Y -= Y.mean(axis=0)
        return kl

    for it in range(n_iter):
        want = (it % max(kl_every, 1) == 0)
        kl = step(it, want, True)
        if kl is not None:
            kl_history.append((it, float(kl)))
        if it in snapshot_iters:
            snapshots[it] = Y.copy()
        if callback is not None:
            callback(it, Y)
    final_kl = float(step(n_iter, True, False))             # KL cuối, không cập nhật Y
    kl_history.append((n_iter, final_kl))
    t_end = time.perf_counter()

    return {
        "embedding": Y,
        "n_components": 2,
        "perplexity": float(perplexity),
        "learning_rate": float(lr),
        "n_iter": int(n_iter),
        "kl_divergence": final_kl,
        "tsne_model": None,
        "backend": f"custom-{method}",
        "device": "cpu",
        "runtime_seconds": float(t_end - t_start),
        "method": method,
        "kl_history": kl_history,
        "snapshots": snapshots,
        "timing": {**timing, "optimize": float(t_end - t_opt), "repulsion": float(t_repulse),
                   "total": float(t_end - t_start)},
        "params": {"theta": float(theta), "n_interpolation_points": int(n_interpolation_points),
                   "early_exaggeration": float(early_exaggeration), "init": init, "n_neighbors": int(P.nnz / n)},
    }


def run_tsne_scalable(
    X_input: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float | str = "auto",
    n_iter: int = 1000,
    random_state: int = 42,
    method: str = "barnes_hut",
    theta: float = 0.5,
    n_interpolation_points: int = 3,
) -> dict[str, Any]:
    """Cùng hợp đồng đầu ra với `run_tsne` (embedding, kl_divergence, backend, runtime_seconds...)."""
    if int(n_components) != 2:
        raise ValueError("t-SNE mở rộng (Barnes-Hut/FFT) chỉ hỗ trợ nhúng 2 chiều.")
    return fit_tsne(X_input, method=method, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate,
                    theta=theta, n_interpolation_points=n_interpolation_points, random_state=random_state)
