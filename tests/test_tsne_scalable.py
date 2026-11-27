import numpy as np
import pytest

from models import tsne_scalable as T


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c * 4, 1, (120, 8)) for c in range(4)])
    y = np.repeat(np.arange(4), 120)
    P, _, _ = T.compute_affinities(X, 20)
    Y = rng.normal(0, 5, (len(X), 2))
    ge, Ze = T.gradient(Y, P, "exact")
    return X, y, P, Y, ge, Ze


def _err(g, ge):
    return np.linalg.norm(g - ge) / np.linalg.norm(ge)


def test_affinities_normalised(data):
    P = data[2]
    assert abs(P.sum() - 1) < 1e-9 and abs(P - P.T).max() < 1e-12


def test_bh_theta_small_matches_exact(data):
    _, _, P, Y, ge, Ze = data
    g, Z = T.gradient(Y, P, "barnes_hut", theta=0.0)
    assert _err(g, ge) < 1e-8 and abs(Z - Ze) / Ze < 1e-8


def test_bh_error_grows_with_theta(data):
    _, _, P, Y, ge, _ = data
    e = [_err(T.gradient(Y, P, "barnes_hut", theta=t)[0], ge) for t in (0.1, 0.5, 1.0)]
    assert e[0] < e[1] < e[2] and e[1] < 0.05


def test_fft_error_shrinks_with_points(data):
    _, _, P, Y, ge, _ = data
    e3 = _err(T.gradient(Y, P, "fft", n_interpolation_points=3)[0], ge)
    e5 = _err(T.gradient(Y, P, "fft", n_interpolation_points=5)[0], ge)
    assert e5 < e3 < 0.05


@pytest.mark.parametrize("method", T.METHODS)
def test_kl_decreases_and_separates(data, method):
    X, y, *_ = data
    r = T.fit_tsne(X, method, perplexity=20, n_iter=300)
    kl = [v for _, v in r["kl_history"]]
    assert kl[-1] < kl[1] and np.isfinite(r["embedding"]).all()
    E = r["embedding"]
    cent = np.array([E[y == c].mean(0) for c in range(4)])
    within = np.mean([np.linalg.norm(E[y == c] - cent[c], axis=1).mean() for c in range(4)])
    between = np.mean([np.linalg.norm(cent[a] - cent[b]) for a in range(4) for b in range(a)])
    assert between > 2 * within


def test_contract_keys(data):
    r = T.run_tsne_scalable(data[0], n_iter=50)
    for k in ("embedding", "n_components", "perplexity", "learning_rate", "n_iter", "kl_divergence",
              "tsne_model", "backend", "device", "runtime_seconds"):
        assert k in r
