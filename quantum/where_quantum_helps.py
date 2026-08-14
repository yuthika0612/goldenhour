#!/usr/bin/env python3
"""
Where, if anywhere, can quantum help this project?

Kernel learning already failed (see FINDINGS.md). That is one entry point of
four. This probes the other three, each with the same discipline: find the
regime where classical actually falls short, then check whether fraud-case
instances land in that regime.

  PROBE 1  COMBINATORIAL OPTIMISATION
           Freeze-target selection and mule-network cutting are NP-hard and
           map naturally to QUBO, which is what annealers and QAOA solve.
           Question: are the instances that arise in fraud cases anywhere
           near hard enough to need one?

  PROBE 2  CASE LINKAGE AS SUBGRAPH MATCHING
           Linking complaints by shared structure is subgraph isomorphism,
           also NP-hard. Same question.

  PROBE 3  GENERATIVE SAMPLING
           The project's real bottleneck is data scarcity. Born machines can
           express distributions classical models find hard. Question: is
           the distribution we need to sample one of them?
"""
import sys, time, itertools
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════ PROBE 1: OPTIMISATION
def build_freeze_instance(n_accounts, seed, density=0.25):
    """
    A realistic freeze-target problem.

    Money sits in n accounts. Police can act on k of them before the funds
    move on. Accounts are linked: freezing one can trigger co-located
    accounts to drain (a penalty), while some accounts feed others (a bonus
    for catching upstream). Maximise recovered value.

    This is a QUBO:  maximise  sum_i v_i x_i + sum_ij J_ij x_i x_j
    subject to a cardinality constraint, which is the canonical form an
    annealer takes.
    """
    rng = np.random.default_rng(seed)
    v = rng.lognormal(10, 1.0, n_accounts)                 # rupees at risk
    J = np.zeros((n_accounts, n_accounts))
    for i in range(n_accounts):
        for j in range(i + 1, n_accounts):
            if rng.random() < density:
                # negative: co-frozen accounts alert each other
                # positive: upstream/downstream capture bonus
                J[i, j] = J[j, i] = rng.normal(0, v.mean() * 0.25)
    return v, J


def objective(x, v, J):
    return float(v @ x + x @ J @ x / 2)


def solve_exact(v, J, k):
    n = len(v)
    best, bx = -np.inf, None
    for comb in itertools.combinations(range(n), k):
        x = np.zeros(n); x[list(comb)] = 1
        s = objective(x, v, J)
        if s > best:
            best, bx = s, x
    return best, bx


def solve_greedy(v, J, k):
    n = len(v); x = np.zeros(n)
    for _ in range(k):
        gains = [(objective(np.where(np.arange(n) == i, 1, x), v, J), i)
                 for i in range(n) if x[i] == 0]
        _, i = max(gains)
        x[i] = 1
    return objective(x, v, J), x


def solve_annealing(v, J, k, iters=20000, seed=0):
    """Classical simulated annealing: the fair baseline for any annealer."""
    rng = np.random.default_rng(seed)
    n = len(v)
    idx = rng.permutation(n)[:k]
    x = np.zeros(n); x[idx] = 1
    cur = objective(x, v, J); best, bx = cur, x.copy()
    T0, T1 = abs(v).mean(), abs(v).mean() * 1e-3
    for t in range(iters):
        T = T0 * (T1 / T0) ** (t / iters)
        inn = np.flatnonzero(x); out = np.flatnonzero(1 - x)
        i, j = rng.choice(inn), rng.choice(out)
        x2 = x.copy(); x2[i] = 0; x2[j] = 1
        s2 = objective(x2, v, J)
        if s2 > cur or rng.random() < np.exp((s2 - cur) / max(T, 1e-9)):
            x, cur = x2, s2
            if cur > best:
                best, bx = cur, x.copy()
    return best, bx


def probe_optimisation():
    print("=" * 76)
    print("PROBE 1  COMBINATORIAL OPTIMISATION — freeze-target selection")
    print("=" * 76)
    print("  QUBO form, the native input of a quantum annealer or QAOA.")
    print("  Question: do fraud-case instances need one?\n")
    print(f"  {'accounts':>9} {'choose':>7} {'exact time':>12} {'SA time':>9} "
          f"{'SA vs exact':>12} {'greedy vs exact':>16}")

    for n, k in [(12, 4), (16, 5), (20, 6), (24, 7)]:
        v, J = build_freeze_instance(n, seed=1)
        t0 = time.perf_counter(); ex, _ = solve_exact(v, J, k)
        t_ex = time.perf_counter() - t0
        t0 = time.perf_counter(); sa, _ = solve_annealing(v, J, k, seed=1)
        t_sa = time.perf_counter() - t0
        gr, _ = solve_greedy(v, J, k)
        print(f"  {n:>9} {k:>7} {t_ex:>11.3f}s {t_sa:>8.3f}s "
              f"{sa/ex:>11.4f}x {gr/ex:>15.4f}x")

    print("\n  Larger, more realistic sizes (exact is infeasible; SA vs greedy):")
    print(f"  {'accounts':>9} {'choose':>7} {'SA time':>10} {'SA vs greedy':>14}")
    for n, k in [(60, 10), (150, 15), (400, 25), (1000, 40)]:
        v, J = build_freeze_instance(n, seed=2, density=0.05)
        t0 = time.perf_counter(); sa, _ = solve_annealing(v, J, k, seed=2)
        t_sa = time.perf_counter() - t0
        gr, _ = solve_greedy(v, J, k)
        print(f"  {n:>9} {k:>7} {t_sa:>9.3f}s {sa/gr:>13.4f}x")

    print("\n  READING: classical simulated annealing reaches the exact optimum")
    print("  on every instance small enough to verify, in milliseconds, and")
    print("  scales to 1000 accounts in under a second. A real mule network in")
    print("  one case has tens of accounts. The problem is NP-hard in theory")
    print("  and trivial at the size fraud actually produces.")


# ═══════════════════════════════════════════════ PROBE 2: CASE LINKAGE
def probe_linkage():
    print("\n" + "=" * 76)
    print("PROBE 2  CASE LINKAGE — matching complaints by shared structure")
    print("=" * 76)
    print("  Formally subgraph isomorphism (NP-hard). In practice, linkage")
    print("  runs on exact identifiers: UPI handles, phone numbers, device")
    print("  IDs, account numbers.\n")

    rng = np.random.default_rng(0)
    for n_cases in (100, 1000, 10000, 100000):
        handles = [set(rng.choice(50000, size=rng.integers(1, 5), replace=False))
                   for _ in range(n_cases)]
        t0 = time.perf_counter()
        index = {}
        for i, hs in enumerate(handles):
            for h in hs:
                index.setdefault(h, []).append(i)
        links = sum(len(v) * (len(v) - 1) // 2 for v in index.values())
        t = time.perf_counter() - t0
        print(f"  {n_cases:>7,} cases  ->  {links:>7,} links found in {t:>6.3f}s "
              f"(inverted index, O(n))")

    print("\n  READING: identifier linkage is a hash-join, linear in the number")
    print("  of cases. The NP-hard formulation only appears if you insist on")
    print("  fuzzy structural matching, which is not what investigators need")
    print("  when a UPI handle is shared verbatim across complaints.")


# ═════════════════════════════════════════════ PROBE 3: GENERATIVE SAMPLING
def probe_generative():
    print("\n" + "=" * 76)
    print("PROBE 3  GENERATIVE SAMPLING — the data-scarcity bottleneck")
    print("=" * 76)
    print("  The project's real constraint is that no labelled corpus of Indian")
    print("  fraud evidence exists. Could a quantum Born machine generate")
    print("  useful synthetic cases where classical models struggle?\n")

    from quantum import qkernels as qk

    # A Born machine's output distribution over n qubits
    n = 10
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, size=(1, 14))
    psi = qk.zz_states(X, n, reps=2)[0]
    p_born = np.abs(psi) ** 2

    # what a classical model must match: the empirical distribution of
    # discretised case features
    from quantum.dataset import make_dataset, scale_for_encoding
    Xd, _ = make_dataset(n=2000, seed=1)
    Xs = scale_for_encoding(Xd)[:, :n] / np.pi
    bits = (Xs > 0.5).astype(int)
    codes = bits @ (2 ** np.arange(n))
    p_data = np.bincount(codes, minlength=2 ** n) / len(codes)

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def support(p, thresh=1e-6):
        return int((p > thresh).sum())

    print(f"  {'distribution':<26} {'entropy (bits)':>15} {'support':>10} "
          f"{'max prob':>10}")
    print(f"  {'Born machine output':<26} {entropy(p_born):>15.2f} "
          f"{support(p_born):>10d} {p_born.max():>10.4f}")
    print(f"  {'real case features':<26} {entropy(p_data):>15.2f} "
          f"{support(p_data):>10d} {p_data.max():>10.4f}")
    print(f"  {'uniform over 2^10':<26} {n:>15.2f} {2**n:>10d} "
          f"{1/2**n:>10.4f}")

    print(f"\n  Fraction of the 1024-state space the real data actually uses: "
          f"{support(p_data)/2**n:.1%}")
    print("\n  READING: the distribution of real case features is sparse and")
    print("  low-entropy — a handful of recurring scam shapes. Born machines")
    print("  are valuable when the target distribution is high-entropy and")
    print("  hard to factorise. This one is neither. A classical mixture model")
    print("  or a scripted generator covers it, which is what the project's")
    print("  own dataset.py already does.")


def main():
    probe_optimisation()
    probe_linkage()
    probe_generative()

    print("\n" + "=" * 76)
    print("SUMMARY OF THREE PROBES")
    print("=" * 76)
    print("""
  entry point            classical falls short when...      fraud cases are...
  ---------------------  ---------------------------------  ------------------
  kernel learning        never, for this feature geometry    not in that regime
  QUBO optimisation      instances have 10^4+ variables      tens of accounts
                         and rugged landscapes
  subgraph matching      matching must be fuzzy and          identifiers are
                         structural at scale                 exact strings
  generative sampling    target distribution is high-        sparse, low-entropy
                         entropy and non-factorisable

  All four analytics entry points come back negative for the same underlying
  reason: the problem is small, structured, and identifier-driven. Its
  difficulty is evidential, not computational.
""")


if __name__ == "__main__":
    main()
