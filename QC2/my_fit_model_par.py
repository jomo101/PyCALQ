"""
my_fit_model_par.py — fast, parallel S-D mixing fit engine (drop-in study module).

Use it from any run JSON by setting:

    "model_file": "./my_fit_model_par.py"

Everything else (quantum_numbers, data blocks, plots) works unchanged: this
module loads my_fit_model.py for all plotting/diagnostic functions and only
replaces the computational core (_model_datap2 / _chi2 / minimizechi2 / vij).

Why it is fast
--------------
The old code re-evaluates the box matrix B(E) — the expensive Lüscher
zeta-function part — at every grid point of every energy scan for every
optimizer step.  But B(E) does not depend on the fit parameters at all: only
the K-matrix does.  So this module

  1. tabulates B(E) ONCE per (P², irrep) block on Chebyshev nodes inside each
     inter-NI-level window (poles at the free levels are divided out first, so
     the tabulated function is smooth and the interpolant is accurate to
     ~machine precision),
  2. rebuilds Ktilde^-1(E) vectorized in numpy from the JSON channel config at
     each optimizer step, and
  3. finds the quantization-condition roots of
         Omega(mu, E) = det(Q) / det(sqrt(mu^2 + Q Qdag)),   Q = Ktilde^-1 - B
     from a batched eigen-decomposition on a probe grid followed by Brent
     refinement on the smooth interpolant (xtol 1e-13 — no more step-size /
     n_refine resolution limits).

At setup the numpy reconstruction is verified numerically against the C++
BoxQuantization (basis ordering, K-matrix elements, and Omega values).  Any
block that fails verification automatically falls back to the original
my_fit_model._findspectrum_irrep path, so results are never silently wrong.

Parallelism
-----------
Independent (P², irrep) blocks and independent parameter vectors are evaluated
across a spawn-based process pool (each worker owns its own tabulated engine):

  * least-squares jacobian columns          (strategy "least_squares")
  * multistart local fits                   (strategy "multistart")
  * block-chunked chi^2 evaluations         (strategy "nelder-mead")
  * vij / Hessian finite differences

JSON knobs (all under "fit": {...}, all optional)
-------------------------------------------------
  strategy      : "least_squares" (recommended) | "nelder-mead" | "multistart"
  n_starts      : multistart count (default 20)
  param_bounds  : [[lo,hi], ...]   used by least_squares/multistart
  xatol, fatol  : tolerance knobs (mapped to xtol/ftol for least_squares)

Environment:
  HPW_FIT_WORKERS : number of worker processes (default: all cores, capped 8;
                    set 0 to disable multiprocessing entirely)
"""

import importlib.util
import os
import sys
import time
import atexit
import traceback
from collections import Counter
from itertools import permutations

import numpy as np
from numpy.polynomial import chebyshev as _np_cheb
from scipy.optimize import brentq, minimize, least_squares, linear_sum_assignment
from scipy.linalg import cholesky as _sp_cholesky, solve_triangular

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_THIS_PATH = os.path.abspath(__file__)

# ── load my_fit_model.py into an isolated namespace ──────────────────────────
_BASE_PATH = os.path.join(_THIS_DIR, "my_fit_model.py")
_bspec = importlib.util.spec_from_file_location("_fit_par_base", _BASE_PATH)
_base = importlib.util.module_from_spec(_bspec)
sys.modules["_fit_par_base"] = _base
_bspec.loader.exec_module(_base)

# Re-export everything public (plots, build_energy_cm_dict, helpers, ...).
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

BMat = _base.BMat

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ── module state ──────────────────────────────────────────────────────────────
_PROGRESS_LOGGER = None
_ENGINE = None            # cached _SpectrumEngine for the current fit structure
_ENGINE_KEY = None
_POOL = None              # cached _EvalPool
_POOL_KEY = None

DEFAULT_OPTS = {
    "n_cheb": 96,          # Chebyshev nodes per window (doubles on retry)
    "n_cheb_max": 384,
    "n_probe_uniform": 321,   # uniform probe points per window for sign scan
    "n_probe_edge": 45,       # geometric probe points stacked at each NI wall
    "edge_probe_min": 1e-8,   # innermost probe distance as fraction of window
    "interp_tol": 1e-8,       # max allowed |Omega_interp - Omega_exact| off-node
    "mu": 1.0,                # Omega regulator (old code used -1.0 -> mu^2 = 1)
    "verify": True,
}


def set_progress_logger(logger_fn):
    global _PROGRESS_LOGGER
    _PROGRESS_LOGGER = logger_fn
    _base.set_progress_logger(logger_fn)


def _plog(msg):
    if callable(_PROGRESS_LOGGER):
        _PROGRESS_LOGGER(msg)
    else:
        print(msg, flush=True)


def set_quantum_numbers(qn):
    global _ENGINE, _ENGINE_KEY
    _base.set_quantum_numbers(qn)
    _ENGINE = None
    _ENGINE_KEY = None


def clear_warm_root_cache():
    _base.clear_warm_root_cache()


# ══════════════════════════════════════════════════════════════════════════════
#  Vectorized K-matrix (mirrors _base._evaluate_k_matrix / _k_matrix_coupled)
# ══════════════════════════════════════════════════════════════════════════════

def _kvec_eval(mode, coeffs, p2, ecm, MN, MK):
    """Vectorized clone of _base._evaluate_k_matrix for uncoupled modes."""
    mode = str(mode or "ere").strip().lower()
    p2 = np.asarray(p2, dtype=float)
    ecm = np.asarray(ecm, dtype=float)
    c = [float(v) for v in np.atleast_1d(coeffs)]
    if mode == "zero":
        return np.zeros_like(ecm)
    if mode in ("polynomial", "poly"):
        out = np.zeros_like(p2)
        for i, ci in enumerate(c):
            out = out + ci * p2 ** i
        return out
    if mode == "constant":
        return np.full_like(p2, c[0] if c else 0.0)
    if mode == "linear":
        if not c:
            return np.zeros_like(p2)
        if len(c) == 1:
            return np.full_like(p2, c[0])
        return c[0] + c[1] * p2
    if mode == "ere":
        a0 = c[0]
        r0 = c[1] if len(c) > 1 else 0.0
        if a0 == 0.0:
            raise ValueError("k_matrix='ere' received a0=0")
        return (-1.0 / a0) + 0.5 * r0 * p2
    if mode == "ere_inv":
        return 1.0 / _kvec_eval("ere", c, p2, ecm, MN, MK)
    if mode == "ere1_vs":
        a1 = c[0] if len(c) > 0 else 0.0
        r1 = c[1] if len(c) > 1 else 0.0
        r2 = c[2] if len(c) > 2 else 0.0
        x = ecm ** 2 - 4.0
        return ecm * (a1 + x * r1 + x ** 2 * r2)
    if mode == "ere1_vs_inv":
        return 1.0 / _kvec_eval("ere1_vs", c, p2, ecm, MN, MK)
    if mode == "epsilon":
        # theta = c0*p2 + c1*p2^2 + ...  (single coefficient -> linear, as before)
        out = np.zeros_like(p2)
        for i, ci in enumerate(c):
            out = out + ci * p2 ** (i + 1)
        return out
    if mode == "poly_s":
        # Polynomial in s = ecm^2 of arbitrary order: c0 + c1*s + c2*s^2 + ...
        s = ecm ** 2
        out = np.zeros_like(s)
        for i, ci in enumerate(c):
            out = out + ci * s ** i
        return out
    raise ValueError(f"Unsupported k_matrix mode '{mode}' in vectorized evaluator")


def _kvec_coupled(coeffs, Lr, Lc, p2, ecm, MN, MK, param_groups, modes, L_values):
    """Vectorized clone of _base._k_matrix_coupled (same p2 barrier factors)."""
    lv = sorted(int(v) for v in (L_values if L_values is not None else [0, 2]))
    La, Lb = lv[0], lv[1]
    coeffs = list(np.atleast_1d(coeffs))
    if param_groups is None or len(param_groups) < 3:
        param_groups = [len(coeffs) - 2, 1, 1]
    o1 = param_groups[0]
    o2 = o1 + param_groups[1]
    g1, g2, g3 = coeffs[:o1], coeffs[o1:o2], coeffs[o2:]
    if modes and len(modes) >= 3:
        m1, m2, m3 = modes[0], modes[1], modes[2]
    else:
        m1, m2, m3 = "ere1_vs", "ere", "epsilon"
    ka = _kvec_eval(m1, g1, p2, ecm, MN, MK)
    kb = _kvec_eval(m2, g2, p2, ecm, MN, MK)
    th = _kvec_eval(m3, g3, p2, ecm, MN, MK)
    ct, st = np.cos(th), np.sin(th)
    Lr, Lc = int(Lr), int(Lc)
    if Lr == La and Lc == La:
        return ka * ct ** 2 + kb * st ** 2 / p2 ** 2
    if Lr == Lb and Lc == Lb:
        return kb * ct ** 2 + ka * st ** 2 * p2 ** 2
    if {Lr, Lc} == {La, Lb}:
        return ct * st * (p2 * ka - kb / p2)
    return np.zeros_like(ecm)


def _qcm_sq(ecm, MN, MK):
    """CM momentum squared over mref^2 (continuum dispersion, two masses)."""
    ecm = np.asarray(ecm, dtype=float)
    if abs(MN - MK) < 1e-12:
        return ecm ** 2 / 4.0 - MN ** 2
    return (ecm ** 2 - (MN + MK) ** 2) * (ecm ** 2 - (MN - MK) ** 2) / (4.0 * ecm ** 2)


class _KinvBuilder:
    """Builds Ktilde^-1(E) as (nE, N, N) arrays for a fixed basis ordering."""

    def __init__(self, basis, MN, MK):
        # basis: list of (chan, twoS, twoJ, L, occ) in matrix index order
        self.basis = list(basis)
        self.N = len(basis)
        self.MN = float(MN)
        self.MK = float(MK)
        # Resolve, once, which channel-config (or None) supplies each element.
        self.elem_specs = {}   # (r, c) -> (channel_dict, twoJ, Lr, Lc) for r <= c
        for r in range(self.N):
            for c in range(r, self.N):
                cr = self.basis[r]
                cc = self.basis[c]
                if cr[2] != cc[2] or cr[4] != cc[4]:
                    continue  # J or occurrence mismatch -> exact zero
                J = cr[2]
                Lr, Lc = cr[3], cc[3]
                Sr, Sc = cr[1], cc[1]
                chr_, chc = cr[0], cc[0]
                if _base.isZero(J, Lr, Sr, chr_, Lc, Sc, chc):
                    continue
                match = None
                for ch in _base._normalize_qn_channels():
                    if ch["enabled"] and _base._match_qn(ch, J, Lr, Sr, chr_, Lc, Sc, chc):
                        match = ch
                        break
                if match is None:
                    continue
                self.elem_specs[(r, c)] = (match, J, Lr, Lc)

    def build(self, par, ecm):
        ecm = np.asarray(ecm, dtype=float)
        p2 = _qcm_sq(ecm, self.MN, self.MK)
        K = np.zeros((len(ecm), self.N, self.N), dtype=float)
        par = np.asarray(par, dtype=float)
        for (r, c), (ch, J, Lr, Lc) in self.elem_specs.items():
            coeffs = par[ch["slice"]]
            kmode = ch.get("k_matrix", "polynomial")
            if isinstance(kmode, list) or str(kmode).strip().lower().startswith("coupled"):
                val = _kvec_coupled(coeffs, Lr, Lc, p2, ecm, self.MN, self.MK,
                                    ch.get("_param_groups"), ch.get("_k_matrix_modes"),
                                    ch.get("L_values"))
            elif Lr == Lc and Lr in (0, 1, 2, 3):
                val = _kvec_eval(kmode, coeffs, p2, ecm, self.MN, self.MK)
            else:
                val = np.zeros_like(ecm)
            K[:, r, c] = val
            if r != c:
                K[:, c, r] = val
        return K

    def verify_against_base(self, par_list, ecm_list):
        """Element-wise check vs _base.calcFunc_param. Raises on mismatch."""
        for par in par_list:
            for e in ecm_list:
                Kv = self.build(par, np.array([e]))[0]
                p2fn = [lambda x: float(_qcm_sq(np.array([x]), self.MN, self.MK)[0])]
                for r in range(self.N):
                    for c in range(r, self.N):
                        cr, cc = self.basis[r], self.basis[c]
                        if cr[2] != cc[2] or cr[4] != cc[4]:
                            ref = 0.0
                        else:
                            ref = _base.calcFunc_param(
                                cr[2], cr[3], cr[1], cr[0], cc[3], cc[1], cc[0],
                                float(e), p2fn, self.MN, self.MK, *par)
                        got = Kv[r, c]
                        scale = 1.0 + abs(ref)
                        if abs(got - ref) > 1e-9 * scale:
                            raise RuntimeError(
                                f"Kinv element mismatch basis[{r}]x[{c}] "
                                f"{cr}x{cc} at Ecm={e}: numpy={got!r} base={ref!r}")


# ══════════════════════════════════════════════════════════════════════════════
#  Per-(P², irrep) block solver
# ══════════════════════════════════════════════════════════════════════════════

def _ni_levels_exact(mL, nP, ecm_max, n_shells=4):
    """Free levels like _base._generate_ni_levels but WITHOUT rounding.

    The rounded values displace the pole-factor zeros from the true B-matrix
    poles by ~5e-11, which leaves a near-cancelled pole in the regularized
    function and contaminates the Chebyshev fit at the 1e-4 level.  Pole
    factors need the exact double-precision positions.
    """
    import itertools as _it
    nP = np.asarray(nP, dtype=int)
    rng = np.arange(-n_shells, n_shells + 1)
    energies = []
    for i in _it.product(rng, rng, rng):
        ecm = _base._free_energy_single(mL, np.array(i, dtype=float), nP)
        if ecm >= ecm_max:
            continue
        if any(abs(ecm - e) < 1e-9 for e in energies):
            continue
        energies.append(float(ecm))
    energies.sort()
    return np.asarray(energies, dtype=float)


def _max_L_active():
    max_L = 0
    for ch in _base._normalize_qn_channels():
        if ch["enabled"]:
            lv = ch.get("L_values")
            if lv is not None:
                for v in lv:
                    max_L = max(max_L, int(v))
            else:
                max_L = max(max_L, int(ch.get("L", 0)))
    return max_L


def _make_box_q(frame_label, psq, irrep, max_L_bmat, mL, MN, MK, calc, is_zero):
    """Build a BoxQuantization; returns (box_q, kmat) — keep BOTH alive."""
    kmat = BMat.KMatrix(calc, is_zero)
    chan_list = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
    box_q = BMat.BoxQuantization(frame_label, psq, irrep, chan_list,
                                 [max_L_bmat], kmat, True)
    box_q.setRefMassL(mL)
    box_q.setMassesOverRef(0, MN, MK)
    return box_q, kmat


class _BlockSolver:
    """One (P², irrep) block: B(E) tables + vectorized QC root finding."""

    def __init__(self, mom, irrep, data_ecm, mL, MN, MK, opts, verify_par=None):
        self.mom = np.asarray(mom, dtype=int)
        self.psq = int(np.sum(self.mom ** 2))
        self.irrep = str(irrep)
        self.frame_label = _base._frame_label_from_psq(self.psq)
        self.mL = float(mL)
        self.MN = float(MN)
        self.MK = float(MK)
        self.opts = dict(opts)
        self.mu = float(self.opts.get("mu", 1.0))
        self.data_ecm = np.asarray(data_ecm, dtype=float)
        self.nsols = len(self.data_ecm)
        self.fast_ok = False
        self.fallback_reason = None
        self.interp_err = np.nan
        self.max_L = _max_L_active()
        self.max_L_bmat = self.max_L + 1

        if self.nsols == 0:
            raise ValueError(f"Block {self.irrep} PSQ={self.psq} has no data levels")

        try:
            self._setup_fast_path(verify_par)
            self.fast_ok = True
        except Exception as exc:
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            self.fast_ok = False

    # ── setup ────────────────────────────────────────────────────────────────
    def _setup_fast_path(self, verify_par):
        _base._require_bmat()
        ecm_max = float(np.max(self.data_ecm)) + 0.35
        self.ni_levels = _ni_levels_exact(self.mL, self.mom, ecm_max + 0.3)

        self._discover_basis()
        self.kinv_builder = _KinvBuilder(self.basis, self.MN, self.MK)

        # verification parameter sets
        if verify_par is None:
            p0 = _base._p0_from_config()
            if p0.size == 0:
                p0 = np.ones(_base._expected_param_count(default_len=1))
            verify_par = [np.asarray(p0, dtype=float)]
        self.verify_par = [np.asarray(p, dtype=float) for p in verify_par]

        self._build_windows()
        self._tabulate()
        if self.opts.get("verify", True):
            self.kinv_builder.verify_against_base(
                self.verify_par, [w["nodes"][len(w["nodes"]) // 3] for w in self.windows])
            self._verify_omega()

    def _discover_basis(self):
        """Determine basis size, content, and index ordering of Ktilde^-1 - B."""
        records = []

        def probe_calc(J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF):
            records.append((int(J), int(Lp), int(Sp), int(chanp),
                            int(L), int(S), int(chan)))
            return 0.17 if (Lp == L and Sp == S and chanp == chan) else 0.03

        box_q, _k = _make_box_q(self.frame_label, self.psq, self.irrep,
                                self.max_L_bmat, self.mL, self.MN, self.MK,
                                probe_calc, _base.isZero)
        # test energy away from any NI pole
        e_test = None
        lo = 2.0 + 1e-4
        for cand in np.linspace(lo + 0.01, lo + 0.5, 200):
            if len(self.ni_levels) == 0 or np.min(np.abs(self.ni_levels - cand)) > 5e-3:
                e_test = float(cand)
                break
        if e_test is None:
            e_test = lo + 0.013

        records.clear()
        eigs = box_q.getEigenvaluesFromEcm(e_test)
        self.N = len(eigs)
        if self.N == 0:
            raise RuntimeError("Empty basis (all states excluded by isZero)")

        # diagonal-in-type record count == occurrence multiplicity of the type
        diag_counts = Counter()
        types = set()
        for (J, Lp, Sp, chanp, L, S, chan) in records:
            types.add((chanp, Sp, J, Lp))
            types.add((chan, S, J, L))
            if (Lp, Sp, chanp) == (L, S, chan):
                diag_counts[(chan, S, J, L)] += 1
        if sum(diag_counts.values()) != self.N:
            raise RuntimeError(
                f"Basis multiplicity inference failed: sum(mult)="
                f"{sum(diag_counts.values())} != N={self.N} "
                f"(types={dict(diag_counts)})")

        # group types into (chan, twoS) blocks; order within a block is
        # ascending (twoJ, L, occurrence) — the C++ std::set ordering.
        blocks = {}
        for t in sorted(diag_counts):
            blocks.setdefault((t[0], t[1]), []).append(t)
        block_keys = sorted(blocks)
        self._basis_candidates = []
        for perm in permutations(block_keys):
            basis = []
            for bk in perm:
                for (chan, S, J, L) in blocks[bk]:
                    for occ in range(diag_counts[(chan, S, J, L)]):
                        basis.append((chan, S, J, L, occ))
            self._basis_candidates.append(basis)
        self.basis = self._basis_candidates[0]

    def _build_windows(self):
        """One window per contiguous inter-NI-wall interval holding data, plus
        (by default) the adjacent empty intervals on either side.

        The neighbors matter for weakly interacting waves: a level measured
        just below an NI wall can have its model root pushed just *above* the
        wall (repulsive shift), i.e. into the next inter-NI interval. Those
        intervals hold no data — they only contribute candidate roots, and
        predict() assigns roots to levels block-globally.
        """
        ni = self.ni_levels
        win_map = {}
        for k, e in enumerate(np.sort(self.data_ecm)):
            below = ni[ni < e]
            above = ni[ni > e]
            # No NI wall below happens for sub-threshold (bound-state) levels:
            # open the window down to a margin under the data level instead.
            lo = float(below[-1]) if len(below) else float(e) - 0.08
            hi = float(above[0]) if len(above) else float(e) + 0.15
            win_map.setdefault((lo, hi), []).append(float(e))

        if bool(self.opts.get("scan_neighbor_windows", True)):
            ni_sorted = np.sort(ni)
            for (lo, hi) in list(win_map.keys()):
                below = ni_sorted[ni_sorted < lo - 1e-12]
                if len(below) and (float(below[-1]), lo) not in win_map:
                    if lo - float(below[-1]) > 1e-8:
                        win_map.setdefault((float(below[-1]), lo), [])
                above = ni_sorted[ni_sorted > hi + 1e-12]
                if len(above) and (hi, float(above[0])) not in win_map:
                    if float(above[0]) - hi > 1e-8:
                        win_map.setdefault((hi, float(above[0])), [])

        self.windows = []
        for (lo, hi), levels in sorted(win_map.items()):
            width = hi - lo
            if width <= 0:
                raise RuntimeError(f"Degenerate window [{lo}, {hi}]")
            # pole factors: every NI level near the window (harmless if not a
            # true pole of this irrep's B — the product stays smooth)
            mask = (ni > lo - 0.75 * width) & (ni < hi + 0.75 * width)
            poles = np.unique(np.concatenate([ni[mask], [lo, hi]]))
            self.windows.append({
                "lo": lo, "hi": hi,
                "levels": sorted(levels),
                "poles": poles,
            })

    def _elab_from_ecm(self, ecm):
        psq_phys = self.psq * (2.0 * np.pi / self.mL) ** 2
        return np.sqrt(np.asarray(ecm, dtype=float) ** 2 + psq_phys)

    def _tabulate_window(self, w, n_cheb):
        lo, hi = w["lo"], w["hi"]
        width = hi - lo
        delta = max(1e-7 * width, 1e-12)
        a, b = lo + delta, hi - delta
        k = np.arange(n_cheb)
        x = np.cos(np.pi * k / (n_cheb - 1))[::-1]          # Lobatto nodes in [-1, 1]
        nodes = 0.5 * (a + b) + 0.5 * (b - a) * x

        dummy = lambda J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF: 0.0
        kmat = BMat.KMatrix(dummy, _base.isZero)
        chan_list = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
        fbq = BMat.FastBoxQuantization(self.frame_label, self.psq, self.irrep,
                                       chan_list, [self.max_L_bmat], kmat, True)
        elab = self._elab_from_ecm(nodes)
        nE = len(nodes)
        fbq.initialize([float(v) for v in elab],
                       [self.mL] * nE,
                       [([self.MN] * nE, [self.MK] * nE)])
        ecm_actual = np.asarray(fbq.getEcmOverMrefList(), dtype=float)
        if np.max(np.abs(ecm_actual - nodes)) > 1e-9:
            raise RuntimeError("Elab->Ecm round trip mismatch in FastBoxQuantization")

        B = np.zeros((nE, self.N, self.N), dtype=complex)
        for r in range(self.N):
            for c in range(r, self.N):
                vals = np.asarray(fbq.getBoxMatrixElementList(r, c), dtype=complex)
                B[:, r, c] = vals
                if r != c:
                    B[:, c, r] = np.conjugate(vals)

        # regularize: h(E) = B(E) * prod (E - pole) is smooth across the window
        pf = np.ones(nE)
        for p in w["poles"]:
            pf *= (ecm_actual - p)
        h = B * pf[:, None, None]

        xs = (2.0 * ecm_actual - (a + b)) / (b - a)
        coeffs = _np_cheb.chebfit(xs, h.reshape(nE, -1), deg=n_cheb - 1)
        w["a"], w["b"] = a, b
        w["nodes"] = ecm_actual
        w["coeffs"] = coeffs           # (n_cheb, N*N) complex
        w["B_nodes"] = B
        # B on the probe grid is parameter-independent: precompute it once so
        # the per-step sign scan needs no interpolation at all.
        w["probe"] = self._probe_grid(w)
        w["B_probe"] = self._B_interp(w, w["probe"])

    def _tabulate(self):
        n_cheb = int(self.opts.get("n_cheb", 96))
        for w in self.windows:
            self._tabulate_window(w, n_cheb)

    # ── evaluation ───────────────────────────────────────────────────────────
    def _B_interp(self, w, ecm):
        ecm = np.atleast_1d(np.asarray(ecm, dtype=float))
        xs = (2.0 * ecm - (w["a"] + w["b"])) / (w["b"] - w["a"])
        deg = w["coeffs"].shape[0] - 1
        vander = _np_cheb.chebvander(xs, deg)                 # (nE, deg+1)
        h = (vander @ w["coeffs"]).reshape(len(ecm), self.N, self.N)
        pf = np.ones(len(ecm))
        for p in w["poles"]:
            pf *= (ecm - p)
        B = h / pf[:, None, None]
        return 0.5 * (B + np.conjugate(np.swapaxes(B, 1, 2)))   # enforce hermiticity

    def _omega_from_Q(self, Q):
        lam = np.linalg.eigvalsh(Q)
        return np.prod(lam / np.sqrt(self.mu ** 2 + lam * lam), axis=-1)

    def omega_window(self, w, par, ecm):
        """Omega on arbitrary energies inside window w (vectorized)."""
        ecm = np.atleast_1d(np.asarray(ecm, dtype=float))
        B = self._B_interp(w, ecm)
        K = self.kinv_builder.build(par, ecm)
        return self._omega_from_Q(K - B)

    def _probe_grid(self, w):
        lo, hi = w["a"], w["b"]
        width = hi - lo
        nu = int(self.opts.get("n_probe_uniform", 321))
        ne = int(self.opts.get("n_probe_edge", 45))
        fmin = float(self.opts.get("edge_probe_min", 1e-8))
        pts = [np.linspace(lo, hi, nu)]
        if ne > 0:
            g = width * np.geomspace(fmin, 0.45, ne)
            pts.append(lo + g)
            pts.append(hi - g)
        grid = np.unique(np.concatenate(pts))
        return grid[(grid >= lo) & (grid <= hi)]

    def _roots_in_window(self, w, par):
        grid = w["probe"]
        K = self.kinv_builder.build(par, grid)
        om = self._omega_from_Q(K - w["B_probe"])
        ok = np.isfinite(om)
        idx = np.where(ok[:-1] & ok[1:] & (om[:-1] * om[1:] < 0.0))[0]
        if len(idx) == 0:
            return []

        # Batched bracket refinement: subdivide every bracket 33-fold per round
        # in a single vectorized Omega call. 4 rounds shrink each bracket by
        # 32^4 ≈ 1e6 (probe spacing ~1e-4 -> ~1e-10), then a secant step lands
        # at the interpolant root to ~1e-13.
        lo = grid[idx].astype(float)
        hi = grid[idx + 1].astype(float)
        flo = om[idx].astype(float)
        fhi = om[idx + 1].astype(float)
        m = 33
        for _ in range(4):
            E = np.linspace(lo, hi, m, axis=1)              # (nb, m)
            F = self.omega_window(w, par, E.ravel()).reshape(len(lo), m)
            F[:, 0], F[:, -1] = flo, fhi
            sign = F[:, :-1] * F[:, 1:] < 0.0
            first = np.argmax(sign, axis=1)
            has = sign.any(axis=1)
            rows = np.arange(len(lo))
            lo = np.where(has, E[rows, first], lo)
            hi = np.where(has, E[rows, first + 1], hi)
            flo = np.where(has, F[rows, first], flo)
            fhi = np.where(has, F[rows, first + 1], fhi)
        denom = fhi - flo
        root = np.where(np.abs(denom) > 0, lo - flo * (hi - lo) / np.where(denom == 0, 1, denom),
                        0.5 * (lo + hi))
        roots = sorted(float(r) for r in root if np.isfinite(r))
        merged = []
        for r in roots:
            if merged and abs(r - merged[-1]) < 1e-10:
                continue
            merged.append(r)
        return merged

    def predict(self, par):
        """QC roots assigned to this block's data levels (ascending-data order).

        Returns an array of len(data_ecm) sorted-ascending predictions with
        NaN for levels whose root could not be found.
        """
        if not self.fast_ok:
            return self._predict_fallback(par)
        # block-global optimal assignment: candidate roots from every window
        # (including empty neighbor windows) vs all of this block's levels —
        # a level may match a root just beyond its own window's NI wall
        d_all, r_all = [], []
        for w in self.windows:
            d_all.extend(w["levels"])
            r_all.extend(self._roots_in_window(w, par))
        d = np.asarray(sorted(d_all), dtype=float)     # ascending-data order
        r = np.asarray(r_all, dtype=float)
        pred = np.full(len(d), np.nan)
        if len(r) > 0:
            cost = np.abs(r[:, None] - d[None, :])
            ri, di = linear_sum_assignment(cost)
            for a_, b_ in zip(ri, di):
                pred[b_] = r[a_]
        return pred

    def _predict_fallback(self, par):
        """Original slow path from my_fit_model (per-block safety net)."""
        sols = _base._findspectrum_irrep(
            0.1 / 3000.0, self.nsols, self.mom, self.mL, par, self.irrep,
            self.MN, self.MK, data_ecm=list(self.data_ecm), debug=False,
            energy_aware=False, step_mode="wave_adaptive", n_refine=3000,
            mN_err=None, fit_mode="omega")
        pred = np.full(self.nsols, np.nan)
        for i, s in enumerate(sorted(sols)[: self.nsols]):
            pred[i] = s
        return pred

    # ── verification ─────────────────────────────────────────────────────────
    def _exact_omega_fn(self, par):
        par = np.asarray(par, dtype=float)
        calc = lambda J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF: _base.calcFunc_param(
            J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF, self.MN, self.MK, *par)
        box_q, kmat = _make_box_q(self.frame_label, self.psq, self.irrep,
                                  self.max_L_bmat, self.mL, self.MN, self.MK,
                                  calc, _base.isZero)
        holder = (box_q, kmat)   # keep the KMatrix alive alongside box_q

        def f(e):
            return float(holder[0].getOmegaFromEcm(-self.mu, float(e)))
        return f

    def _verify_omega(self):
        """Pick the basis ordering that reproduces the C++ Omega; measure the
        off-node interpolation error; retry with denser nodes if needed."""
        tol_node = 1e-6
        best = None
        for cand in self._basis_candidates:
            self.basis = cand
            self.kinv_builder = _KinvBuilder(cand, self.MN, self.MK)
            max_err = 0.0
            ok = True
            for par in self.verify_par:
                exact = self._exact_omega_fn(par)
                for w in self.windows:
                    nodes = w["nodes"]
                    sel = nodes[np.linspace(2, len(nodes) - 3, 5).astype(int)]
                    B = w["B_nodes"][np.linspace(2, len(nodes) - 3, 5).astype(int)]
                    K = self.kinv_builder.build(par, sel)
                    om_py = self._omega_from_Q(K - B)
                    for e, o in zip(sel, om_py):
                        o_cpp = exact(e)
                        err = abs(o - o_cpp) / (1.0 + abs(o_cpp))
                        max_err = max(max_err, err)
                        if err > tol_node:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                best = (cand, max_err)
                break
        if best is None:
            raise RuntimeError(
                f"Omega verification failed for all {len(self._basis_candidates)} "
                f"basis ordering candidates (last max_err={max_err:.3e})")
        self.basis, node_err = best

        # Off-node interpolation error. Densification decisions use the middle
        # 60% of each window: near the NI walls Omega is pole-steep, so a tiny
        # B-interpolation error shows up as a large Omega difference there while
        # the ROOT position error stays tiny (error_E = error_Omega / |dOmega/dE|).
        # The end-of-fit polish_roots() check quantifies the true root error.
        tol = float(self.opts.get("interp_tol", 1e-7))
        n_cheb = int(self.opts.get("n_cheb", 96))
        n_max = int(self.opts.get("n_cheb_max", 384))
        par = self.verify_par[0]
        exact = self._exact_omega_fn(par)
        while True:
            max_err = 0.0
            for w in self.windows:
                nodes = w["nodes"]
                mids = 0.5 * (nodes[:-1] + nodes[1:])
                inner = mids[(mids > w["a"] + 0.2 * (w["b"] - w["a"]))
                             & (mids < w["b"] - 0.2 * (w["b"] - w["a"]))]
                if len(inner) == 0:
                    continue
                sel = inner[np.linspace(0, len(inner) - 1,
                                        min(7, len(inner))).astype(int)]
                om_i = self.omega_window(w, par, sel)
                for e, o in zip(sel, om_i):
                    o_cpp = exact(e)
                    max_err = max(max_err, abs(o - o_cpp) / (1.0 + abs(o_cpp)))
            self.interp_err = max_err
            if max_err <= tol or n_cheb >= n_max:
                break
            n_cheb = min(2 * n_cheb, n_max)
            for w in self.windows:
                self._tabulate_window(w, n_cheb)
        if self.interp_err > 1e-4:
            raise RuntimeError(
                f"Interpolation error {self.interp_err:.3e} too large even with "
                f"{n_cheb} Chebyshev nodes")

    def polish_roots(self, par, roots, delta=5e-5):
        """Re-solve each root with the exact C++ Omega. Returns (exact, |diff|).

        The bracket is clamped inside the root's inter-NI window so it never
        crosses a pole (near-wall roots sit ~1e-7 from the wall).
        """
        exact = self._exact_omega_fn(par)
        ni = self.ni_levels
        out = []
        for r in roots:
            if not np.isfinite(r):
                out.append((np.nan, np.nan))
                continue
            below = ni[ni < r]
            above = ni[ni > r]
            d = delta
            if len(below):
                d = min(d, 0.45 * (r - float(below[-1])))
            if len(above):
                d = min(d, 0.45 * (float(above[0]) - r))
            d = max(d, 1e-12)
            try:
                fa, fb = exact(r - d), exact(r + d)
                if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                    re = brentq(exact, r - d, r + d,
                                xtol=1e-13, rtol=4e-15, maxiter=200)
                    out.append((float(re), abs(re - r)))
                else:
                    out.append((np.nan, np.nan))
            except Exception:
                out.append((np.nan, np.nan))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Whole-spectrum engine
# ══════════════════════════════════════════════════════════════════════════════

class _SpectrumEngine:
    def __init__(self, spec):
        self.spec = spec
        frames = spec["frames"]
        irreps = spec["irreps"]
        levels = spec["levels"]
        self.mL = float(spec["mL"])
        self.MN = float(spec["MN"])
        self.MK = float(spec["MK"])
        opts = dict(DEFAULT_OPTS)
        opts.update(spec.get("opts", {}))
        self.opts = opts

        # per-(frame, irrep) data-Ecm lists in the base _model_datap2 order
        data2cm = spec.get("data2cm")
        kept_mom2 = spec.get("kept_mom2")
        kept_irreps = spec.get("kept_irreps")
        lookup = {}
        if data2cm is not None and kept_mom2 is not None and kept_irreps is not None:
            for k, (m2, irr) in enumerate(zip(kept_mom2, kept_irreps)):
                key = (tuple(int(round(x)) for x in np.asarray(m2).ravel()), str(irr))
                lookup.setdefault(key, []).append(float(np.asarray(data2cm[k]).ravel()[0]))

        verify_par = [np.asarray(p, dtype=float) for p in spec.get("verify_par", [])] or None

        self.blocks = []
        self.block_slices = []
        self.block_perms = []     # ascending-sort permutation of each block's data
        pos = 0
        t0 = time.perf_counter()
        for i, mom in enumerate(frames):
            for j, irrep in enumerate(irreps[i]):
                n_levels = int(levels[i][j])
                key = (tuple(int(round(x)) for x in np.asarray(mom).ravel()), str(irrep))
                d = lookup.get(key, [])
                if len(d) != n_levels:
                    raise ValueError(
                        f"Block {irrep} P={key[0]}: {len(d)} data levels found, "
                        f"levels struct says {n_levels}. The fast engine needs "
                        "data2cm/kept_mom2/kept_irreps.")
                order = np.argsort(d)                      # data ascending
                inv = np.empty_like(order)
                inv[order] = np.arange(len(order))
                blk = _BlockSolver(mom, irrep, np.asarray(d)[order],
                                   self.mL, self.MN, self.MK, opts,
                                   verify_par=verify_par)
                self.blocks.append(blk)
                self.block_perms.append(inv)
                self.block_slices.append(slice(pos, pos + n_levels))
                pos += n_levels
        self.n_levels = pos
        self.setup_time = time.perf_counter() - t0

    def summary(self):
        lines = []
        for blk in self.blocks:
            mode = "fast" if blk.fast_ok else f"FALLBACK ({blk.fallback_reason})"
            lines.append(
                f"  {blk.irrep:>4s} PSQ={blk.psq}  N={getattr(blk, 'N', '?')}  "
                f"windows={len(getattr(blk, 'windows', []))}  levels={blk.nsols}  "
                f"interp_err={blk.interp_err:.2e}  [{mode}]")
        return "\n".join(lines)

    def predict_ecm(self, par, block_idx=None):
        """Ecm/mref predictions in the global data ordering.

        block_idx: optional list of block indices (for chunked parallel eval);
        entries outside those blocks are NaN.
        """
        out = np.full(self.n_levels, np.nan)
        indices = range(len(self.blocks)) if block_idx is None else block_idx
        for bi in indices:
            blk = self.blocks[bi]
            pred_sorted = blk.predict(np.asarray(par, dtype=float))
            out[self.block_slices[bi]] = pred_sorted[self.block_perms[bi]]
        return out


def build_engine_from_spec(spec):
    return _SpectrumEngine(spec)


def predict_energy_cm_dict(par, data2cm, mom2, irreps, levels, mL, MN, MK,
                           n_refine=1000, step_mode="wave_adaptive",
                           fit_mode="omega"):
    """Engine-based override of my_fit_model.predict_energy_cm_dict.

    Same signature and return shape ({PSQxx: {irrep: {level: [Ecm]}}}), but the
    roots come from the tabulated-B probe grid + exact-Omega polish instead of
    the serial grid scan.  The serial path structurally excludes roots within
    its ni_buffer of an NI wall (e.g. the T1g 3D1 needle 1.5e-4 above the n^2=1
    wall), which produced spurious level-assignment cascades in fixed-param
    comparisons; the engine's geometric edge probes find them.
    """
    par = np.asarray(par, dtype=float)

    # rebuild the frames/irreps/levels structure in input encounter order,
    # remembering which flat input rows belong to each (frame, irrep) block
    frames, irr_per, lvl_counts = [], [], []
    forder, rows = {}, {}
    for i, (mom, irrep) in enumerate(zip(mom2, irreps)):
        fkey = tuple(int(x) for x in np.asarray(mom).ravel())
        if fkey not in forder:
            forder[fkey] = len(frames)
            frames.append(list(fkey))
            irr_per.append([])
            lvl_counts.append([])
        fi = forder[fkey]
        ir = str(irrep)
        if ir not in irr_per[fi]:
            irr_per[fi].append(ir)
            lvl_counts[fi].append(0)
        lvl_counts[fi][irr_per[fi].index(ir)] += 1
        rows.setdefault((fkey, ir), []).append(i)

    spec = _make_spec(frames, irr_per, lvl_counts, mL, MN, MK,
                      data2cm, mom2, irreps, verify_par=[par])
    engine = _SpectrumEngine(spec)
    pred = engine.predict_ecm(par)     # ordered by block_slices construction

    out = {}
    pos = 0
    for fi, frame in enumerate(frames):
        fkey = tuple(int(x) for x in frame)
        psq = int(sum(v * v for v in fkey))
        for ir, n_lvl in zip(irr_per[fi], lvl_counts[fi]):
            block_rows = rows[(fkey, ir)]      # input rows, encounter order
            for k, row_idx in enumerate(block_rows):
                lvl = int(levels[row_idx])
                v = pred[pos + k]
                out.setdefault(f"PSQ{psq}", {}).setdefault(ir, {})[lvl] = \
                    [float(v)] if np.isfinite(v) else []
            pos += n_lvl
    return out


def _engine_key(frames, irreps, levels, mL, MN, MK, data2cm, kept_mom2, kept_irreps):
    def _tt(x):
        return tuple(np.asarray(x, dtype=float).ravel().round(10))
    key = [round(float(mL), 10), float(MN), float(MK), id(_base._QN_CHANNELS_CACHE)]
    for i, f in enumerate(frames):
        key.append((_tt(f), tuple(irreps[i]), tuple(int(v) for v in levels[i])))
    if data2cm is not None:
        key.append(tuple(round(float(np.asarray(r).ravel()[0]), 10) for r in data2cm))
    return tuple(map(str, key))


def _make_spec(frames, irreps, levels, mL, MN, MK, data2cm, kept_mom2, kept_irreps,
               verify_par=None, opts=None):
    return {
        "frames": [np.asarray(f).tolist() for f in frames],
        "irreps": [list(x) for x in irreps],
        "levels": [list(map(int, x)) for x in levels],
        "mL": float(mL), "MN": float(MN), "MK": float(MK),
        "data2cm": [[float(np.asarray(r).ravel()[0])] for r in data2cm]
                   if data2cm is not None else None,
        "kept_mom2": [np.asarray(m).tolist() for m in kept_mom2]
                     if kept_mom2 is not None else None,
        "kept_irreps": [str(s) for s in kept_irreps] if kept_irreps is not None else None,
        "verify_par": [np.asarray(p, dtype=float).tolist() for p in (verify_par or [])],
        "quantum_numbers": dict(_base._ACTIVE_QN),
        "opts": dict(opts or {}),
        "module_path": _THIS_PATH,
    }


def _get_engine(frames, irreps, levels, mL, MN, MK,
                data2cm=None, kept_mom2=None, kept_irreps=None, verify_par=None):
    global _ENGINE, _ENGINE_KEY
    key = _engine_key(frames, irreps, levels, mL, MN, MK,
                      data2cm, kept_mom2, kept_irreps)
    if _ENGINE is not None and _ENGINE_KEY == key:
        return _ENGINE
    spec = _make_spec(frames, irreps, levels, mL, MN, MK,
                      data2cm, kept_mom2, kept_irreps, verify_par=verify_par)
    t0 = time.perf_counter()
    _plog("[par] building fast spectrum engine (one-time B-matrix tabulation)...")
    _ENGINE = _SpectrumEngine(spec)
    _ENGINE_KEY = key
    _plog(f"[par] engine ready in {time.perf_counter() - t0:.2f} s")
    _plog(_ENGINE.summary())
    return _ENGINE


# ══════════════════════════════════════════════════════════════════════════════
#  Worker pool (parameter-vector and block-chunk parallelism)
# ══════════════════════════════════════════════════════════════════════════════

def _n_workers_default():
    try:
        env = os.environ.get("HPW_FIT_WORKERS")
        if env is not None:
            return max(0, int(env))
    except ValueError:
        pass
    import multiprocessing as mp
    return min(8, mp.cpu_count())


class _EvalPool:
    """Process pool over fit_par_worker using multiprocessing.Pool (spawn).

    Note: concurrent.futures.ProcessPoolExecutor was tried first and showed
    lost-result stalls on macOS (all tasks completed in the workers, parent
    never woken). multiprocessing.Pool's delivery path has been reliable.
    """

    def __init__(self, spec, n_workers):
        import fit_par_worker
        import multiprocessing as mp
        # single-threaded BLAS in the workers (batched small matrices)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(var, "1")
        # Children re-import __main__ (run_HPW_fit) and my_fit_model on spawn;
        # this flag makes those imports skip matplotlib TeX probing, which is
        # slow and deadlock-prone when 8 workers race on the TeX cache.
        os.environ["HPW_FIT_WORKER"] = "1"
        ctx = mp.get_context("spawn")
        self.n_workers = n_workers
        self.pool = ctx.Pool(processes=n_workers,
                             initializer=fit_par_worker.init_worker,
                             initargs=(spec,))
        self._worker_mod = fit_par_worker

    def eval_many(self, par_list, timeout=120):
        """Predict Ecm for many parameter vectors in parallel.

        A finite default timeout turns any worker hang into an exception the
        caller can recover from (serial fallback) instead of a silent stall.
        """
        args = [np.asarray(p).tolist() for p in par_list]
        res = self.pool.map_async(self._worker_mod.eval_epred, args)
        return [np.asarray(r) for r in res.get(timeout=timeout)]

    def eval_blocks_chunked(self, par, chunks, timeout=600):
        args = [(np.asarray(par).tolist(), list(c)) for c in chunks]
        res = self.pool.map_async(self._worker_mod.eval_epred_blocks, args)
        return [np.asarray(r) for r in res.get(timeout=timeout)]

    def run_local_fits(self, jobs, timeout=None):
        res = self.pool.map_async(self._worker_mod.run_local_fit, list(jobs))
        try:
            return res.get(timeout=timeout)
        except Exception as exc:
            return [{"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    for _ in jobs]

    def shutdown(self):
        try:
            self.pool.terminate()
            self.pool.join()
        except Exception:
            pass


def _get_pool(spec, fit_ctx=None):
    global _POOL, _POOL_KEY
    n_workers = _n_workers_default()
    if n_workers <= 1:
        return None
    key = (str(spec.get("quantum_numbers")), str(spec.get("data2cm")),
           spec.get("mL"), n_workers,
           str(fit_ctx.get("data").tolist()) if fit_ctx else None)
    if _POOL is not None and _POOL_KEY == key:
        return _POOL
    if _POOL is not None:
        _POOL.shutdown()
        _POOL = None
    full_spec = dict(spec)
    if fit_ctx is not None:
        full_spec["fit_ctx"] = {
            "data": np.asarray(fit_ctx["data"]).tolist(),
            "cov": np.asarray(fit_ctx["cov"]).tolist(),
            "observable": fit_ctx["observable"],
            "ni_sum": (np.asarray(fit_ctx["ni_sum"]).tolist()
                       if fit_ctx.get("ni_sum") is not None else None),
        }
    try:
        _plog(f"[par] starting {n_workers} worker processes...")
        t0 = time.perf_counter()
        _POOL = _EvalPool(full_spec, n_workers)
        # warm-up: force worker initialization now, and verify one eval.
        # A hard timeout guards against worker-startup hangs — on timeout we
        # fall back to serial evaluation rather than blocking the fit.
        p0 = full_spec.get("verify_par") or [[0.0]]
        _POOL.eval_many([p0[0]], timeout=300)
        _plog(f"[par] workers ready in {time.perf_counter() - t0:.2f} s")
        _POOL_KEY = key
    except Exception as exc:
        _plog(f"[par] WARNING: worker pool unavailable ({exc}); running serial")
        _POOL = None
        _POOL_KEY = None
    return _POOL


@atexit.register
def _shutdown_pool():
    global _POOL
    if _POOL is not None:
        _POOL.shutdown()
        _POOL = None


# ══════════════════════════════════════════════════════════════════════════════
#  Study-module interface: _model_datap2 / _chi2 / minimizechi2 / vij
# ══════════════════════════════════════════════════════════════════════════════

def _to_observable(epred, observable, MN, ni_sum):
    epred = np.asarray(epred, dtype=float)
    if observable == "p2":
        return epred ** 2 / 4.0 - 1.0
    if observable == "ecm":
        return epred
    if observable == "elab":
        # True lab energy: E_lab/mN = sqrt((E_cm/mN)^2 + (P/mN)^2).
        # The per-level lab boost (P/mN)^2 is carried in the ni_sum slot
        # (set by the runner for fit_observable='elab').
        if ni_sum is None:
            raise ValueError("observable='elab' requires (P/mN)^2 per level "
                             "via ni_sum")
        return np.sqrt(epred ** 2 + np.asarray(ni_sum, dtype=float))
    if observable == "shift":
        if ni_sum is None:
            raise ValueError("observable='shift' requires ni_sum")
        ni = np.asarray(ni_sum, dtype=float)
        if ni.ndim == 2:
            # rows: [ni_sum_lab/mN, (P/mN)^2] — lab-frame shift with the
            # cm->lab boost of the model root (matches the lab-frame dE_NN
            # data): shift/mN = sqrt(ecm^2 + (P/mN)^2) - ni_lab/mN.
            return np.sqrt(epred ** 2 + ni[1]) - ni[0]
        return epred * MN - ni   # legacy rest-frame-only formula
    raise ValueError(f"Unknown observable '{observable}'")


def _model_datap2(par, frames, irreps, levels, mL, MN, MK, n=400,
                  debug=False, data2cm=None, kept_mom2=None, kept_irreps=None,
                  step_mode='adaptive', n_refine=1, mN_err=None, fit_mode='omega',
                  observable='p2', ni_sum=None):
    engine = _get_engine(frames, irreps, levels, mL, MN, MK,
                         data2cm=data2cm, kept_mom2=kept_mom2,
                         kept_irreps=kept_irreps, verify_par=[np.asarray(par, float)])
    epred = engine.predict_ecm(np.asarray(par, dtype=float))
    return _to_observable(epred, observable, MN, ni_sum)


def _chi2_from_model(model_arr, data_arr, cov_arr):
    """Chi^2 with the same missing-level penalty semantics as my_fit_model."""
    model_arr = np.asarray(model_arr, dtype=float)
    data_arr = np.asarray(data_arr, dtype=float)
    bad = ~np.isfinite(model_arr)
    n_missing = int(bad.sum())
    if n_missing == 0:
        r = data_arr - model_arr
        return float(r @ np.linalg.solve(cov_arr, r))
    penalty = 1e6 * n_missing
    good = ~bad
    if good.any():
        r = (data_arr - model_arr)[good]
        sub = cov_arr[np.ix_(good, good)]
        return float(r @ np.linalg.solve(sub, r)) + penalty
    return float(penalty)


def _chi2(par, data, frames, irreps, levels, mL, cov, MN, MK, n=400,
          data2cm=None, kept_mom2=None, kept_irreps=None,
          step_mode='adaptive', n_refine=1, mN_err=None, fit_mode='omega',
          observable='shift', ni_sum=None):
    try:
        model = _model_datap2(par, frames, irreps, levels, mL, MN, MK, n=n,
                              data2cm=data2cm, kept_mom2=kept_mom2,
                              kept_irreps=kept_irreps, observable=observable,
                              ni_sum=ni_sum)
    except Exception as exc:
        _plog(f"[par] chi2 model failure: {exc}")
        return 1e12
    return _chi2_from_model(model, data, np.asarray(cov, dtype=float))


def _whitener(cov):
    cov = np.asarray(cov, dtype=float)
    try:
        L = _sp_cholesky(cov, lower=True)
    except Exception:
        jitter = 1e-12 * np.mean(np.diag(cov))
        L = _sp_cholesky(cov + jitter * np.eye(len(cov)), lower=True)

    def whiten(resid):
        return solve_triangular(L, resid, lower=True)
    return whiten


_MISSING_RESID = 1e3   # whitened residual assigned to an unfound level


def _resid_whitened(epred, data, whiten, observable, MN, ni_sum):
    model = _to_observable(epred, observable, MN, ni_sum)
    bad = ~np.isfinite(model)
    m = np.where(bad, data, model)
    r = whiten(data - m)
    r[bad] = _MISSING_RESID
    return r


def minimizechi2(data, frames, irreps, levels, mL, cov, p0, MN, MK,
                 n=400, max_iter=None, n_maxiter=1000, strategy="nelder-mead",
                 n_starts=20, param_bounds=None,
                 data2cm=None, kept_mom2=None, kept_irreps=None,
                 step_mode='adaptive', n_refine=1,
                 xatol=1e-6, fatol=1e-6, mN_err=None, fit_mode='omega',
                 observable='shift', ni_sum=None, bootstrap=None,
                 multistart_seed=42, multistart_dump=None):
    p0 = np.asarray(p0, dtype=float)
    data_arr = np.asarray(data, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    expected = _base._expected_param_count(default_len=len(p0))
    if expected != len(p0):
        raise ValueError(
            f"Initial parameter length ({len(p0)}) does not match expected ({expected}).")

    engine = _get_engine(frames, irreps, levels, mL, MN, MK,
                         data2cm=data2cm, kept_mom2=kept_mom2,
                         kept_irreps=kept_irreps, verify_par=[p0])
    whiten = _whitener(cov_arr)
    fit_ctx = {"data": data_arr, "cov": cov_arr,
               "observable": observable, "ni_sum": ni_sum}
    spec = _make_spec(frames, irreps, levels, mL, MN, MK,
                      data2cm, kept_mom2, kept_irreps, verify_par=[p0])
    pool = _get_pool(spec, fit_ctx=fit_ctx)

    eval_count = {"n": 0}

    def model_ecm(par):
        eval_count["n"] += 1
        return engine.predict_ecm(par)

    def resid(par):
        return _resid_whitened(model_ecm(par), data_arr, whiten,
                               observable, MN, ni_sum)

    def chi2_of(par):
        return float(np.sum(resid(par) ** 2))

    def parallel_resids(par_list):
        nonlocal pool
        if pool is None:
            return [resid(p) for p in par_list]
        try:
            epreds = pool.eval_many(par_list)
        except Exception as exc:
            _plog(f"[par] WARNING: parallel eval failed ({type(exc).__name__}: "
                  f"{exc}); switching to serial for the rest of this fit")
            pool = None
            return [resid(p) for p in par_list]
        eval_count["n"] += len(par_list)
        return [_resid_whitened(e, data_arr, whiten, observable, MN, ni_sum)
                for e in epreds]

    def parallel_jac(par, f0=None):
        par = np.asarray(par, dtype=float)
        h = np.maximum(1e-6, 1e-6 * np.abs(par))
        pts, signs = [], []
        for i in range(len(par)):
            pp = par.copy(); pp[i] += h[i]
            pm = par.copy(); pm[i] -= h[i]
            pts.extend([pp, pm])
        rs = parallel_resids(pts)
        J = np.empty((len(data_arr), len(par)))
        for i in range(len(par)):
            J[:, i] = (rs[2 * i] - rs[2 * i + 1]) / (2.0 * h[i])
        return J

    t0 = time.perf_counter()
    chi2_init = chi2_of(p0)
    _plog(f"[par] initial chi2 = {chi2_init:.8g}  "
          f"({(time.perf_counter() - t0) * 1e3:.0f} ms/eval)")

    strategy = str(strategy).strip().lower().replace("-", "_")
    truncated = False

    ls_kwargs = dict(
        method="trf",
        jac=parallel_jac,
        x_scale="jac",
        ftol=max(fatol, 1e-14),
        xtol=max(xatol * 1e-3, 1e-14),
        gtol=1e-12,
        max_nfev=int(n_maxiter),
    )
    if param_bounds:
        lo = np.array([b[0] for b in param_bounds], dtype=float)
        hi = np.array([b[1] for b in param_bounds], dtype=float)
        ls_kwargs["bounds"] = (lo, hi)
    elif strategy in ("least_squares", "trf", "lm"):
        # Loose default bounds keep the trust region from wandering off into
        # the missing-root penalty plateau (flat residuals, no gradient).
        lo = np.array([v - max(10.0 * abs(v), 10.0) for v in p0])
        hi = np.array([v + max(10.0 * abs(v), 10.0) for v in p0])
        ls_kwargs["bounds"] = (lo, hi)
        _plog(f"[par] no param_bounds given; using loose defaults p0 ± max(10|p0|, 10)")

    if strategy in ("least_squares", "trf", "lm"):
        _plog("[par] minimization started [least_squares/trf, parallel jacobian]")
        res = least_squares(resid, p0, **ls_kwargs)
        best_x, best_chi2 = res.x, float(np.sum(res.fun ** 2))
        converged = bool(res.status > 0)
        _plog(f"[par] least_squares status={res.status} nfev={res.nfev} "
              f"njev={res.njev} ({res.message})")

    elif strategy == "multistart":
        if param_bounds is None:
            param_bounds = [(v - max(abs(v) * 2.0, 1.0), v + max(abs(v) * 2.0, 1.0))
                            for v in p0]
        # Propagate these bounds into ls_kwargs: it was built above before
        # `strategy == "multistart"` was checked, so it never picked up any
        # bounds (the `elif strategy in (...)` branch that sets loose
        # defaults only fires for strategy in {least_squares, trf, lm}).
        # Only the parallel-pool per-start jobs below received `param_bounds`
        # explicitly; the serial per-start loop (`local_kwargs = dict(ls_kwargs)`)
        # and the final polish (`least_squares(resid, best_x, **ls_kwargs)`)
        # were both running fully unbounded, letting the optimizer wander
        # into the missing-root NaN plateau and blow parameters up to
        # unphysical values. setdefault so an explicitly-passed param_bounds
        # (already handled earlier) is never overridden.
        ls_kwargs.setdefault("bounds", (
            np.array([b[0] for b in param_bounds], dtype=float),
            np.array([b[1] for b in param_bounds], dtype=float),
        ))
        rng = np.random.default_rng(int(multistart_seed))
        starts = [p0] + [
            np.array([rng.uniform(lo, hi) for (lo, hi) in param_bounds])
            for _ in range(int(n_starts))]
        _plog(f"[par] multistart: {len(starts)} starts "
              f"({'parallel' if pool else 'serial'})")
        results = []
        if pool is not None:
            jobs = [{"p0": s.tolist(),
                     "bounds": [list(b) for b in param_bounds],
                     "n_maxiter": int(n_maxiter),
                     "ftol": max(fatol, 1e-14), "xtol": max(xatol * 1e-3, 1e-14)}
                    for s in starts]
            for i, out in enumerate(pool.run_local_fits(jobs)):
                if out.get("ok"):
                    results.append((float(out["chi2"]), np.asarray(out["x"])))
                    _plog(f"  start {i + 1}/{len(starts)}: chi2={out['chi2']:.6g}")
                else:
                    _plog(f"  start {i + 1}/{len(starts)} failed: {out.get('error')}")
        else:
            local_kwargs = dict(ls_kwargs)
            local_kwargs["jac"] = "2-point"
            for i, s in enumerate(starts):
                try:
                    r = least_squares(resid, s, **local_kwargs)
                    results.append((float(np.sum(r.fun ** 2)), r.x))
                    _plog(f"  start {i + 1}/{len(starts)}: chi2={results[-1][0]:.6g}")
                except Exception as exc:
                    _plog(f"  start {i + 1} failed: {exc}")
        if not results:
            raise RuntimeError("All multistart fits failed")
        results.sort(key=lambda t: t[0])
        if multistart_dump:
            # dump every start's converged (chi2, params) for basin analysis
            os.makedirs(os.path.dirname(os.path.abspath(multistart_dump)),
                        exist_ok=True)
            np.save(multistart_dump,
                    np.array([[c] + list(np.asarray(x, dtype=float))
                              for c, x in results], dtype=float))
            _plog(f"[par] multistart dump ({len(results)} starts) -> "
                  f"{multistart_dump}")
        best_chi2, best_x = results[0]
        # final polish from the best start with the parallel jacobian
        res = least_squares(resid, best_x, **ls_kwargs)
        best_x, best_chi2 = res.x, float(np.sum(res.fun ** 2))
        converged = True

    elif strategy in ("nelder_mead", "neldermead"):
        # chi2 evals parallelized across blocks when a pool is available
        if pool is not None and len(engine.blocks) > 1:
            nb = len(engine.blocks)
            nchunk = min(pool.n_workers, nb)
            chunks = [list(range(k, nb, nchunk)) for k in range(nchunk)]

            def chi2_nm(par):
                nonlocal pool
                eval_count["n"] += 1
                if pool is not None:
                    try:
                        parts = pool.eval_blocks_chunked(par, chunks)
                        epred = np.full(engine.n_levels, np.nan)
                        for arr in parts:
                            good = np.isfinite(arr)
                            epred[good] = arr[good]
                        model = _to_observable(epred, observable, MN, ni_sum)
                        return _chi2_from_model(model, data_arr, cov_arr)
                    except Exception as exc:
                        _plog(f"[par] WARNING: parallel chi2 failed "
                              f"({type(exc).__name__}: {exc}); going serial")
                        pool = None
                try:
                    model = _to_observable(model_ecm(par), observable, MN, ni_sum)
                except Exception:
                    return 1e12
                return _chi2_from_model(model, data_arr, cov_arr)
        else:
            def chi2_nm(par):
                try:
                    model = _to_observable(model_ecm(par), observable, MN, ni_sum)
                except Exception:
                    return 1e12
                return _chi2_from_model(model, data_arr, cov_arr)

        iter_count = {"n": 0}
        class _EarlyStop(Exception):
            pass

        def callback(xk):
            iter_count["n"] += 1
            if max_iter is not None and iter_count["n"] >= int(max_iter):
                raise _EarlyStop()
            if iter_count["n"] % 25 == 0:
                _plog(f"  NM iter {iter_count['n']}  chi2={chi2_nm(xk):.6g}  "
                      f"params={np.array2string(np.asarray(xk))}")

        _nm_bounds = [tuple(b) for b in param_bounds] if param_bounds else None
        _plog("[par] minimization started [nelder-mead]")
        try:
            res = minimize(chi2_nm, p0, method="Nelder-Mead", bounds=_nm_bounds,
                           callback=callback,
                           options={"maxiter": int(n_maxiter),
                                    "xatol": xatol, "fatol": fatol})
            best_x, best_chi2, converged = res.x, float(res.fun), bool(res.success)
        except _EarlyStop:
            truncated = True
            converged = False
            res = minimize(chi2_nm, p0, method="Nelder-Mead", bounds=_nm_bounds,
                           options={"maxiter": iter_count["n"],
                                    "xatol": xatol, "fatol": fatol})
            best_x, best_chi2 = res.x, float(res.fun)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use least_squares, multistart, "
            "or nelder-mead.")

    elapsed = time.perf_counter() - t0
    status = "TRUNCATED" if truncated else ("converged" if converged else "did NOT converge")
    _plog(f"[par] minimization ended [{status}] chi2={best_chi2:.8g} "
          f"params={np.array2string(np.asarray(best_x))}")
    _plog(f"[par] {eval_count['n']} model evaluations in {elapsed:.1f} s "
          f"({elapsed / max(eval_count['n'], 1) * 1e3:.1f} ms/eval effective)")

    # ── exact-QC polish check: quantify interpolation error at the minimum ────
    try:
        epred = engine.predict_ecm(np.asarray(best_x, dtype=float))
        max_dev, n_checked = 0.0, 0
        for bi, blk in enumerate(engine.blocks):
            if not blk.fast_ok:
                continue
            roots = epred[engine.block_slices[bi]]
            for _, dev in blk.polish_roots(best_x, roots):
                if np.isfinite(dev):
                    max_dev = max(max_dev, dev)
                    n_checked += 1
        _plog(f"[par] exact-QC validation: {n_checked} roots re-solved with the "
              f"C++ Omega; max |dE/mref| = {max_dev:.3e}")
    except Exception as exc:
        _plog(f"[par] exact-QC validation skipped: {exc}")

    # ── bootstrap: refit every bootstrap sample, in parallel across workers ──
    # JSON: "fit": {"bootstrap": {"enabled": true, "n_samples": 200,
    #               "n_maxiter": 400, "save_path": "./results/bs_params.npy"}}
    # Each sample b uses data column b of data2cm (col 0 = central), the
    # central covariance, and starts from the central best fit. Errors are
    # meant to be quoted as the 68% percentile interval (16th-84th) of the
    # resulting parameter distribution.
    bs_cfg = bootstrap if isinstance(bootstrap, dict) else {}
    if bs_cfg.get("enabled", False):
        if data2cm is None:
            _plog("[par][bootstrap] WARNING: data2cm not provided — skipping")
        elif observable not in ("p2", "ecm"):
            _plog(f"[par][bootstrap] observable '{observable}' not supported "
                  "for bootstrap refits — skipping")
        else:
            d2 = np.asarray([np.asarray(d, dtype=float).ravel()
                             for d in data2cm])
            n_avail = d2.shape[1] - 1
            n_bs = min(int(bs_cfg.get("n_samples", 200)), n_avail)
            bs_maxiter = int(bs_cfg.get("n_maxiter", 400))
            bs_save = str(bs_cfg.get("save_path", "./results/bootstrap_params.npy"))
            if param_bounds:
                blo = [float(b[0]) for b in param_bounds]
                bhi = [float(b[1]) for b in param_bounds]
            else:
                blo = [float(v - max(5.0 * abs(v), 5.0)) for v in best_x]
                bhi = [float(v + max(5.0 * abs(v), 5.0)) for v in best_x]
            jobs = []
            for b in range(1, n_bs + 1):
                ecm_b = d2[:, b]
                data_b = ecm_b ** 2 / 4.0 - 1.0 if observable == "p2" else ecm_b
                jobs.append({"p0": np.asarray(best_x, dtype=float).tolist(),
                             "bounds": [[l, h] for l, h in zip(blo, bhi)],
                             "n_maxiter": bs_maxiter,
                             "ftol": 1e-10, "xtol": 1e-12,
                             "data": data_b.tolist()})
            _plog(f"[par][bootstrap] refitting {n_bs} samples "
                  f"(of {n_avail} available), start = central best fit, "
                  f"{'parallel' if pool else 'serial'}")
            t_bs = time.perf_counter()
            if pool is not None:
                outs = pool.run_local_fits(jobs)
            else:
                from scipy.optimize import least_squares as _ls
                outs = []
                for job in jobs:
                    dat_b = np.asarray(job["data"], dtype=float)
                    def _resid_b(p, _d=dat_b):
                        return _resid_whitened(engine.predict_ecm(p), _d,
                                               whiten, observable, MN, ni_sum)
                    try:
                        r = _ls(_resid_b, np.asarray(job["p0"]), method="trf",
                                jac="2-point", x_scale="jac",
                                bounds=(np.array(blo), np.array(bhi)),
                                ftol=1e-10, xtol=1e-12, gtol=1e-12,
                                max_nfev=bs_maxiter)
                        outs.append({"ok": True, "x": r.x.tolist(),
                                     "chi2": float(np.sum(r.fun ** 2))})
                    except Exception as exc:
                        outs.append({"ok": False,
                                     "error": f"{type(exc).__name__}: {exc}"})
            good = [o for o in outs if o.get("ok")]
            n_fail = len(outs) - len(good)
            par_bs = np.array([o["x"] for o in good], dtype=float)
            chi2_bs = np.array([o["chi2"] for o in good], dtype=float)
            _plog(f"[par][bootstrap] {len(good)}/{n_bs} samples converged "
                  f"({n_fail} failed) in {time.perf_counter() - t_bs:.1f} s")
            if len(good):
                os.makedirs(os.path.dirname(os.path.abspath(bs_save)),
                            exist_ok=True)
                np.save(bs_save, par_bs)
                _plog(f"[par][bootstrap] saved {par_bs.shape[0]}x"
                      f"{par_bs.shape[1]} params -> {bs_save}")
                p16 = np.percentile(par_bs, 16, axis=0)
                p50 = np.percentile(par_bs, 50, axis=0)
                p84 = np.percentile(par_bs, 84, axis=0)
                _plog("[par][bootstrap] 68% percentile intervals "
                      "(median [p16, p84], sigma68=(p84-p16)/2):")
                for i in range(par_bs.shape[1]):
                    _plog(f"    p{i}: {p50[i]:.6g}  [{p16[i]:.6g}, "
                          f"{p84[i]:.6g}]  sigma68={(p84[i]-p16[i])/2:.4g}")
                _plog(f"[par][bootstrap] sample chi2: median="
                      f"{np.median(chi2_bs):.4g}  max={np.max(chi2_bs):.4g}")

    return np.asarray(best_x), float(best_chi2), bool(converged)


def vij(par, dataE2, frames, irreps, levels, mL, cov, MN, MK, n=150, epsilon=1e-5,
        data2cm=None, kept_mom2=None, kept_irreps=None,
        step_mode='adaptive', n_refine=1, mN_err=None, fit_mode='omega',
        observable='shift', ni_sum=None):
    par = np.asarray(par, dtype=float)
    data_arr = np.asarray(dataE2, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    engine = _get_engine(frames, irreps, levels, mL, MN, MK,
                         data2cm=data2cm, kept_mom2=kept_mom2,
                         kept_irreps=kept_irreps, verify_par=[par])

    def model(p):
        return _to_observable(engine.predict_ecm(p), observable, MN, ni_sum)

    m0 = model(par)
    r0 = data_arr - m0
    chi2_at = _chi2_from_model(m0, data_arr, cov_arr)
    _plog(f"vij: chi2 at input params = {chi2_at:.8g}")
    dof = len(data_arr) - len(par)
    _plog(f"vij: dof={dof}  reduced_chi2="
          f"{(chi2_at / dof if dof > 0 else np.nan):.8g}")

    h = np.maximum(epsilon, epsilon * np.abs(par))
    jac = np.zeros((len(data_arr), len(par)))
    for i in range(len(par)):
        pp = par.copy(); pp[i] += h[i]
        pm = par.copy(); pm[i] -= h[i]
        mp_, mm_ = model(pp), model(pm)
        col = (mp_ - mm_) / (2.0 * h[i])
        col[~np.isfinite(col)] = 0.0
        jac[:, i] = col

    cov_inv = np.linalg.inv(cov_arr)
    fisher = jac.T @ cov_inv @ jac
    return np.linalg.pinv(fisher)
