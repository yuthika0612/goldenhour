"""
Metrics for comparing a quantum kernel against classical kernels.

Three questions, in the order they should be asked:

  1. Is the quantum representation GEOMETRICALLY DIFFERENT from the classical
     one?  -> geometric_difference (Huang et al., Nature Comms 2021)
     If g is small, a classical model can reproduce whatever the quantum
     kernel does, for ANY labelling. Advantage is ruled out before training.

  2. Does the representation ALIGN with the labels we actually care about?
     -> kernel_target_alignment. A kernel can be exotic and useless.

  3. Does it help on the TASK?  -> cross-validated accuracy / ROC-AUC of a
     kernel classifier, with the same splits for every kernel.

Reporting only (3) is how quantum-ML papers mislead. Reporting (1) first is
what makes a negative result meaningful rather than an anecdote.
"""
import numpy as np
from scipy.linalg import sqrtm


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Centered KTA in [-1, 1]. How much of the kernel's similarity structure
    lines up with the label structure yy^T.
    """
    Kc = center_kernel(K)
    Y = np.outer(y, y).astype(float)
    Yc = center_kernel(Y)
    denom = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return float((Kc * Yc).sum() / denom) if denom > 0 else 0.0


def normalise_kernel(K: np.ndarray) -> np.ndarray:
    """
    Put a kernel in correlation form (unit diagonal).

    This matters. A rank-deficient or badly scaled classical kernel (the
    linear kernel here has effective dimension 12 out of 300) has tiny
    eigenvalues, and inverting it inflates the geometric difference for any
    quantum kernel at all. Without this step the metric reports a large g
    even for a product-state encoding that is trivially simulable
    classically, which is a false positive.
    """
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    return K / np.outer(d, d)


def geometric_difference(K_c: np.ndarray, K_q: np.ndarray,
                         lam: float = 1e-2) -> float:
    """
    g(K_C || K_Q) = sqrt( || sqrt(K_Q) (K_C + lam I)^-1 sqrt(K_Q) ||_inf )

    Huang et al., Nature Communications 12, 2631 (2021).

    With N training points, a quantum advantage requires g >> sqrt(N). If g
    is at or below sqrt(N), the classical kernel can match the quantum one
    for ANY labelling of this data, so no accuracy result can demonstrate
    advantage.

    Both kernels are normalised to unit diagonal first, and lam regularises
    the inverse. Report g against the BEST classical kernel available (see
    geometric_difference_vs_family): advantage must beat every classical
    option, not just a conveniently weak one.
    """
    n = K_c.shape[0]
    Kc = normalise_kernel(K_c)
    Kq = normalise_kernel(K_q)

    sq = np.real(sqrtm(Kq + lam * np.eye(n)))
    inv = np.linalg.inv(Kc + lam * np.eye(n))
    M = sq @ inv @ sq
    ev = np.linalg.eigvalsh((M + M.T) / 2)
    return float(np.sqrt(max(ev.max(), 0.0)))


def geometric_difference_vs_family(K_classical: dict, K_q: np.ndarray,
                                   lam: float = 1e-2):
    """
    g against a whole family of classical kernels. The operative number is
    the MINIMUM: if any classical kernel is geometrically close to the
    quantum one, that classical kernel can do the same job.
    """
    per = {name: geometric_difference(K, K_q, lam) for name, K in K_classical.items()}
    best = min(per, key=per.get)
    return {"per_kernel": per, "min_g": per[best], "closest_classical": best}


def effective_dimension(K: np.ndarray, thresh: float = 0.99) -> int:
    """How many eigenvalues carry `thresh` of the kernel's spectral mass."""
    ev = np.linalg.eigvalsh(K)[::-1]
    ev = np.clip(ev, 0, None)
    if ev.sum() <= 0:
        return 0
    c = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(c, thresh) + 1)


def expressivity_spread(K: np.ndarray) -> float:
    """
    Mean off-diagonal kernel value. Near 0 means every state is nearly
    orthogonal to every other: the kernel matrix approaches the identity,
    the model memorises and cannot generalise. This is the classic failure
    mode of over-expressive quantum feature maps.
    """
    n = K.shape[0]
    off = K[~np.eye(n, dtype=bool)]
    return float(off.mean())


def cv_scores(K: np.ndarray, y: np.ndarray, n_splits: int = 5,
              C: float = 1.0, seed: int = 0) -> dict:
    """Stratified CV with a precomputed-kernel SVM. Same splits for all kernels."""
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for tr, te in skf.split(np.zeros(len(y)), y):
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        pred = clf.predict(K[np.ix_(te, tr)])
        dec = clf.decision_function(K[np.ix_(te, tr)])
        accs.append(balanced_accuracy_score(y[te], pred))
        try:
            aucs.append(roc_auc_score(y[te], dec))
        except ValueError:
            aucs.append(np.nan)
    return {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
    }


def learning_curve(K: np.ndarray, y: np.ndarray, sizes, n_reps: int = 5,
                   seed: int = 0) -> dict:
    """
    Accuracy vs training-set size. Huang et al.'s central empirical point is
    that classical models catch up as data grows, so a single-N comparison
    can be misleading.
    """
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {}
    for m in sizes:
        scores = []
        for _ in range(n_reps):
            idx = rng.permutation(n)
            tr, te = idx[:m], idx[m:m + 60]
            if len(np.unique(y[tr])) < 2 or len(te) < 10:
                continue
            clf = SVC(kernel="precomputed", C=1.0)
            clf.fit(K[np.ix_(tr, tr)], y[tr])
            pred = clf.predict(K[np.ix_(te, tr)])
            scores.append(balanced_accuracy_score(y[te], pred))
        if scores:
            out[m] = (float(np.mean(scores)), float(np.std(scores)))
    return out
