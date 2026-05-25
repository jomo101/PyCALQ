import logging
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
import sys
import gvar as gv
import itertools
import plotting as pt

pt.apply_plot_style(use_tex=True)

try:
    sys.path.append('/Users/jomoscoso/Desktop/Analysis_codes/pythib_jo')
    import BMat
except Exception as exc:
    BMat = None
    _BMAT_IMPORT_ERROR = exc
else:
    _BMAT_IMPORT_ERROR = None


_ACTIVE_QN = {
    "channels": []
}

# ── Performance caches ────────────────────────────────────────────────────────
# _QN_CHANNELS_CACHE : result of _normalize_qn_channels() — rebuilt only when
#                      set_quantum_numbers() is called (once per run).
# _NI_LEVELS_CACHE   : free-spectrum levels — purely geometric, never change
#                      during a fit.
# _BOX_Q_CACHE       : BMAT BoxQuantization objects — the expensive zeta-function
#                      setup depends only on (frame, irrep, mL, MN, MK, max_L).
#                      K-matrix params are injected via a mutable holder so the
#                      cached object stays live across minimizer steps.
_QN_CHANNELS_CACHE = None
_NI_LEVELS_CACHE   = {}   # key: (mL_r8, mom_tuple, ecm_max_r8)
_BOX_Q_CACHE       = {}   # key: (frame_label, psq, irrep, mL_r8, MN, MK, max_L)


class _MutableParams:
    """Thin mutable wrapper letting a cached BoxQuantization see updated parameters."""
    __slots__ = ("par",)
    def __init__(self, par):
        self.par = np.asarray(par, dtype=float)


_PROGRESS_LOGGER = None


def set_progress_logger(logger_fn):
    global _PROGRESS_LOGGER
    _PROGRESS_LOGGER = logger_fn


def _progress_log(message):
    if callable(_PROGRESS_LOGGER):
        _PROGRESS_LOGGER(message)
    else:
        print(message)



def set_quantum_numbers(qn):
    global _ACTIVE_QN, _QN_CHANNELS_CACHE, _BOX_Q_CACHE
    if not isinstance(qn, dict):
        raise TypeError("quantum_numbers must be a dict")
    merged = dict(_ACTIVE_QN)
    merged.update(qn)
    print(f"Setting quantum numbers: {merged}")
    _ACTIVE_QN = merged
    _QN_CHANNELS_CACHE = None   # quantum numbers changed — rebuild next call
    _BOX_Q_CACHE.clear()        # max_L may change, invalidates BoxQ geometry keys


def _normalize_qn_channels():
    global _QN_CHANNELS_CACHE
    if _QN_CHANNELS_CACHE is not None:
        return _QN_CHANNELS_CACHE
    channels = _ACTIVE_QN.get("channels")
    if channels is None:
        channels = _ACTIVE_QN.get("sets", [])

    if not channels:
        channels = [
            {
                "name": "main",
                "twoJ": _ACTIVE_QN.get("twoJ", 0),
                "L": _ACTIVE_QN.get("L", 0),
                "Lp": _ACTIVE_QN.get("Lp", 0),
                "twoS": _ACTIVE_QN.get("twoS", 0),
                "twoSp": _ACTIVE_QN.get("twoSp", 0),
                "chan": _ACTIVE_QN.get("chan", 0),
                "chanp": _ACTIVE_QN.get("chanp", 0),
                "enabled": True,
            }
        ]

    global_params = _ACTIVE_QN.get("params", _ACTIVE_QN.get("calc_terms", []))
    normalized = []
    cursor = 0
    for idx, qn_channel in enumerate(channels):
        item = dict(qn_channel)
        enabled = bool(item.get("enabled", True))
        k_matrix_raw = item.get("k_matrix", "polynomial")
        # k_matrix can be a string (uncoupled) or a list (coupled: one mode per param group)
        if isinstance(k_matrix_raw, list):
            k_matrix_mode = "coupled_sd"  # auto-detect coupled from list
            item["_k_matrix_modes"] = [str(m).strip().lower() for m in k_matrix_raw]
        else:
            k_matrix_mode = str(k_matrix_raw).strip().lower()
            item["_k_matrix_modes"] = None
        if k_matrix_mode == "zero":
            n_params = 0
            item["_param_groups"] = []
        else:
            local_params = item.get("params", item.get("terms", item.get("calc_terms", [])))
            if isinstance(local_params, list) and local_params:
                # Check for nested lists (grouped params for coupled channels)
                if isinstance(local_params[0], list):
                    flat_params = []
                    group_sizes = []
                    for group in local_params:
                        group_sizes.append(len(group))
                        flat_params.extend(group)
                    n_params = len(flat_params)
                    item["_flat_params"] = flat_params
                    item["_param_groups"] = group_sizes
                else:
                    n_params = len(local_params)
                    item["_param_groups"] = [n_params]
            else:
                n_params = int(item.get("n_params", len(global_params)))
                item["_param_groups"] = [n_params] if n_params > 0 else []
        if n_params < 0:
            raise ValueError(f"quantum_numbers.channels[{idx}].n_params must be >= 0")

        item["enabled"] = enabled
        item["n_params"] = n_params
        if enabled:
            item["slice"] = slice(cursor, cursor + n_params)
            cursor += n_params
        else:
            item["slice"] = slice(0, 0)

        normalized.append(item)

    _QN_CHANNELS_CACHE = normalized
    return normalized


def _expected_param_count(default_len=None):
    qn_channels = _normalize_qn_channels()
    if qn_channels:
        total = sum(item["n_params"] for item in qn_channels if item["enabled"])
        return total

    params = _ACTIVE_QN.get("params", _ACTIVE_QN.get("calc_terms", []))
    if params:
        return len(params)

    if default_len is not None:
        return int(default_len)

    return 0
def _n_waves_from_config():
    """
    Determine number of partial waves from the quantum numbers config.

    Rules:
      - Uncoupled single channel  (one L, no L_values) : +1 per enabled channel
      - Uncoupled multi-channel   (N channels, each one L): N total
      - Coupled channel           (L_values list)       : +len(L_values) per channel

    Examples from your configs:
      3D1 only                  → 1
      3D1 + 3S1 (uncoupled)     → 2
      coupled S-D (L_values:[0,2]) → 2
    """
    qn_channels = _normalize_qn_channels()
    n_waves = 0
    for ch in qn_channels:
        if not ch["enabled"]:
            continue
        lv = ch.get("L_values")
        if lv is not None:
            # Coupled: one wave per distinct L value in the coupling
            n_waves += len(set(int(v) for v in lv))
        else:
            # Uncoupled: one wave per channel
            n_waves += 1
    return max(n_waves, 1)


def _get_bmat_size(box_q, ecm_test):
    """Always derive from config — never probe the C++ object."""
    return _n_waves_from_config()

def _wave_label_from_config():
    """
    Build a LaTeX-formatted title string from enabled channels.
    e.g.  3D1_channel  →  $^3D_1$
          3S1 + 3D1    →  $^3S_1 + {}^3D_1$
    """
    _L_LABELS = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H", 6: "I"}

    qn_channels = _normalize_qn_channels()
    labels = []

    def _spectroscopic_latex(twoS, L, twoJ):
        """Build ^{2S+1}L_J in LaTeX."""
        L_str = _L_LABELS.get(L, f"L{{{L}}}")
        return rf"${{}}^{{{twoS+1}}}{L_str}_{{{int(twoJ/2)}}}$"

    for ch in qn_channels:
        if not ch["enabled"]:
            continue

        name = ch.get("name", "")
        if name:
            # Try to parse twoS, L, twoJ from the channel fields
            # and render in LaTeX — fall back to cleaned name if missing
            twoJ = ch.get("twoJ")
            L    = ch.get("L")
            twoS = ch.get("twoS")
            if twoJ is not None and L is not None and twoS is not None:
                labels.append(_spectroscopic_latex(int(twoS), int(L), int(twoJ)))
            else:
                # Fallback: clean up the name string
                clean = (name.replace("_channel", "")
                             .replace("_chan", "")
                             .replace("_wave", "")
                             .strip())
                labels.append(clean)
            continue

        # No name — build from quantum numbers directly
        lv = ch.get("L_values")
        if lv is not None:
            twoS = int(ch.get("twoS", 0))
            twoJ = int(ch.get("twoJ", 0))
            for v in sorted(set(int(x) for x in lv)):
                labels.append(_spectroscopic_latex(twoS, v, twoJ))
        else:
            twoJ = int(ch.get("twoJ", 0))
            L    = int(ch.get("L",    0))
            twoS = int(ch.get("twoS", 0))
            labels.append(_spectroscopic_latex(twoS, L, twoJ))

    return " $+$ ".join(labels) if labels else "Unknown wave"


def _format_irrep_label(irrep):
    irrep = str(irrep)
    if len(irrep) <= 1:
        return irrep
    return rf"${irrep[0]}_{{{irrep[1:]}}}$"

def _match_qn(qn_dict, J, Lp, Sp, chanp, L, S, chan):
    # If L_values is present (coupled channel), match L and Lp against that set.
    # Otherwise use the single L / Lp fields (uncoupled, existing behaviour).
    L_values = qn_dict.get("L_values")
    if L_values is not None:
        L_set = {int(v) for v in L_values}
        l_match = (int(L) in L_set and int(Lp) in L_set)
    else:
        l_match = (
            int(qn_dict.get("L", _ACTIVE_QN.get("L", L))) == int(L)
            and int(qn_dict.get("Lp", _ACTIVE_QN.get("Lp", Lp))) == int(Lp)
        )
    return (
        l_match
        and int(qn_dict.get("twoJ", _ACTIVE_QN.get("twoJ", J))) == int(J)
        and int(qn_dict.get("twoS", _ACTIVE_QN.get("twoS", S))) == int(S)
        and int(qn_dict.get("twoSp", _ACTIVE_QN.get("twoSp", Sp))) == int(Sp)
        and int(qn_dict.get("chan", _ACTIVE_QN.get("chan", chan))) == int(chan)
        and int(qn_dict.get("chanp", _ACTIVE_QN.get("chanp", chanp))) == int(chanp)
    )

def _p0_from_config():
    """
    Extract initial parameter values directly from the config file.
    Reads the 'initial' field of each param in each enabled channel.
    Returns a flat np.ndarray matching the full parameter layout.
    """
    qn_channels = _normalize_qn_channels()
    p0 = []
    for ch in qn_channels:
        if not ch["enabled"]:
            continue
        local_params = ch.get("params", ch.get("terms", ch.get("calc_terms", [])))
        if isinstance(local_params, list) and local_params:
            # nested groups (coupled channels)
            if isinstance(local_params[0], list):
                for group in local_params:
                    for pm in group:
                        p0.append(float(pm.get("initial", 0.0)))
            else:
                for pm in local_params:
                    p0.append(float(pm.get("initial", 0.0)))
        else:
            # fallback: n_params zeros
            n = ch.get("n_params", 0)
            p0.extend([0.0] * n)
    return np.array(p0, dtype=float)

def _free_energy_single(mL, n, nP):
    """
    Single-particle moving-frame energy for momentum n, total momentum nP.
    mL = mN * L  (dimensionless, mN = reference mass = 1)
    Returns Ecm/mN.
    """
    tpomL  = 2.0 * np.pi / mL
    n      = np.asarray(n,  dtype=float)
    nP     = np.asarray(nP, dtype=float)
    E1     = np.sqrt(1.0 + np.dot(n,  n)  * tpomL ** 2)
    E2     = np.sqrt(1.0 + np.dot(nP - n, nP - n) * tpomL ** 2)
    Etot   = E1 + E2
    Psq    = np.dot(nP, nP) * tpomL ** 2
    Ecm_sq = Etot ** 2 - Psq
    return float(np.sqrt(Ecm_sq)) 

def _generate_ni_levels(mL, nP, ecm_max, n_shells=4):
    """
    Free (non-interacting) spectrum in CM frame, in units of mN.

    mL      : mN * L  (dimensionless)
    nP      : total momentum vector (integer array, e.g. [0,0,1])
    ecm_max : upper cutoff in units of mN
    n_shells: max integer for single-particle momentum components

    Returns sorted list of unique Ecm/mN values below ecm_max.
    """
    _cache_key = (
        round(float(mL), 8),
        tuple(int(x) for x in np.asarray(nP).ravel()),
        round(float(ecm_max), 8),
    )
    if _cache_key in _NI_LEVELS_CACHE:
        return _NI_LEVELS_CACHE[_cache_key]

    nP       = np.asarray(nP, dtype=int)
    intList  = np.arange(-n_shells, n_shells + 1)
    energies = []

    for i in itertools.product(intList, intList, intList):
        n   = np.array(i, dtype=float)
        ecm = _free_energy_single(mL, n, nP)
        if ecm is None:
            continue
        ecm = round(ecm, 10)
        if ecm not in energies and ecm < ecm_max:
            energies.append(ecm)

    energies.sort()
    _NI_LEVELS_CACHE[_cache_key] = energies
    return energies

def _nearest_ni(ecm, ni_levels):
    """
    Find the nearest NI level to ecm and its distance.

    Parameters
    ----------
    ecm       : float
    ni_levels : sorted list/array from _generate_ni_levels

    Returns
    -------
    ni_near : float  — nearest NI level
    dist    : float  — signed distance (ecm - ni_near), positive means ecm is above the pole
    ni_below: float or None — nearest NI level strictly below ecm
    ni_above: float or None — nearest NI level strictly above ecm
    """
    ni_arr   = np.asarray(ni_levels, dtype=float)
    if len(ni_arr) == 0:
        return None, np.inf, None, None

    idx      = int(np.argmin(np.abs(ni_arr - ecm)))
    ni_near  = float(ni_arr[idx])
    dist     = float(ecm - ni_near)

    below    = ni_arr[ni_arr < ecm]
    above    = ni_arr[ni_arr > ecm]
    ni_below = float(below[-1]) if len(below) > 0 else None
    ni_above = float(above[0])  if len(above) > 0 else None

    return ni_near, dist, ni_below, ni_above


def isZero(J, Lp, Sp, chanp, L, S, chan):
    qn_channels = _normalize_qn_channels()
    if qn_channels:
        for qn_channel in qn_channels:
            if qn_channel["enabled"] and _match_qn(qn_channel, J, Lp, Sp, chanp, L, S, chan):
                return False
        return True

    return not _match_qn(_ACTIVE_QN, J, Lp, Sp, chanp, L, S, chan)


def _extract_p2(Ecm_over_mref, pSqFuncList):
    if isinstance(pSqFuncList, (list, tuple)) and pSqFuncList:
        first = pSqFuncList[0]
        if callable(first):
            return float(first(Ecm_over_mref))
        return float(first)
    return float((Ecm_over_mref ** 2) / 4.0 - 1.0)

def _k_matrix_polynomial(coeffs, p2):
    value = 0.0
    for idx, coeff in enumerate(coeffs):
        value += coeff * (p2 ** idx)
    return float(value)


def _k_matrix_constant(coeffs, p2):
    _ = p2
    return float(coeffs[0]) if len(coeffs) else 0.0


def _k_matrix_linear(coeffs, p2):
    if len(coeffs) == 0:
        return 0.0
    if len(coeffs) == 1:
        return float(coeffs[0])
    return float(coeffs[0] + coeffs[1] * p2)


def _k_matrix_ere(coeffs, p2):
    if len(coeffs) < 1:
        raise ValueError("k_matrix='ere' requires at least 1 parameter: [a0] or [a0, r0].")
    a0 = float(coeffs[0])
    r0 = float(coeffs[1]) if len(coeffs) > 1 else 0.0
    if a0 == 0.0:
        raise ValueError("k_matrix='ere' received a0=0, causing division by zero in -1/a0.")
    return float((-1.0 / a0) + 0.5 * r0 * p2)

def _k_matrix_poly_s(coeffs, Ecm_over_mref):
    A = float(coeffs[0]) if len(coeffs) > 0 else 0.0
    B = float(coeffs[1]) if len(coeffs) > 1 else 0.0
    C = float(coeffs[2]) if len(coeffs) > 2 else 0.0 
    ecm = float(Ecm_over_mref)
    ere1_val = A + ecm**2 * B + (ecm**2)** 2 * C
    return float(ere1_val)

def _k_matrix_ere1_vs(coeffs, Ecm_over_mref, MN, MK):
    _ = (MN, MK)
    a1m = float(coeffs[0]) if len(coeffs) > 0 else 0.0
    r1m = float(coeffs[1]) if len(coeffs) > 1 else 0.0
    r2m = float(coeffs[2]) if len(coeffs) > 2 else 0.0
    ecm = float(Ecm_over_mref)
    ere1_val = ecm * (a1m + ((ecm**2 - 4.0) ) * r1m + ((ecm**2 - 4.0) ) ** 2 * r2m)
    return float(ere1_val)

def _k_matrix_epsilon(coeffs, p2):
    if len(coeffs) < 1:
        raise ValueError("k_matrix='epsilon' requires at least 1 parameter: [epsilon0].")
    epsilon0 = float(coeffs[0])
    return float(epsilon0 * p2)

def _k_matrix_coupled(coeffs, L, Lp, p2, Ecm_over_mref, MN, MK,
                         param_groups=None, k_matrix_modes=None, L_values=None):
    """Generic coupled two-channel K-matrix (2x2 with mixing angle).

    Works for any pair of angular momenta (S-D, P-F, etc.).
    The two L values are taken from L_values (default [0, 2] for S-D).

    params (in JSON) are declared as 3 groups:
      [ [first-wave params], [second-wave params], [mixing params] ]

    k_matrix (in JSON) is declared as a list of 3 modes:
      ["ere1_vs", "ere", "polynomial"]
    mapping to [first-wave, second-wave, mixing-angle].

    The groups are flattened for the optimizer but split back here using
    param_groups (list of group sizes, e.g. [3, 1, 1]).

    Returns the (L, Lp) element of the rotated K-matrix:
      aa: kcot_a * ct² + kcot_b * st² * p2^{-delta}
      bb: kcot_b * ct² + kcot_a * st² * p2^{delta}
      ab: ct * st * (kcot_a * p2^{delta/2} - kcot_b * p2^{-delta/2})
    where delta = Lb - La.

    Barrier factors are included here — calcFunc_param should NOT
    multiply by p²^L again.
    """
    # Determine the two angular momenta from L_values
    if L_values is not None:
        lvals = sorted(int(v) for v in L_values)
    else:
        lvals = [0, 2]  # default S-D for backward compat
    La, Lb = lvals[0], lvals[1]
    delta = Lb - La  # positive integer (e.g. 2 for S-D, P-F)

    if param_groups is None or len(param_groups) < 3:
        param_groups = [len(coeffs) - 2, 1, 1]

    # Split flat coeffs into groups
    c = list(coeffs)
    off1 = param_groups[0]
    off2 = off1 + param_groups[1]
    g1 = c[:off1]
    g2 = c[off1:off2]
    g3 = c[off2:]

    # Resolve per-group modes (default fallback)
    if k_matrix_modes and len(k_matrix_modes) >= 3:
        mode1, mode2, mode3 = k_matrix_modes[0], k_matrix_modes[1], k_matrix_modes[2]
    else:
        mode1, mode2, mode3 = "ere1_vs", "ere", "epsilon"
    #print(f"Energy={Ecm_over_mref}, p2={p2}, La={La}, Lb={Lb}")
    # First-wave eigenvalue (La)
    kcot_a = _evaluate_k_matrix(mode1, g1, p2,
                                Ecm_over_mref=Ecm_over_mref, MN=MN, MK=MK)
    #print(f"kcot_a (La={La}) = {kcot_a}")

    # Second-wave eigenvalue (Lb)
    kcot_b = _evaluate_k_matrix(mode2, g2, p2,
                                Ecm_over_mref=Ecm_over_mref, MN=MN, MK=MK)
    #print(f"kcot_b (Lb={Lb}) = {kcot_b}")
    # Mixing angle
    theta_val = _evaluate_k_matrix(mode3, g3, p2,
                                   Ecm_over_mref=Ecm_over_mref, MN=MN, MK=MK)
    # print(f"theta (mixing) = {theta_val} radians")
    # gg
    ct = np.cos(theta_val)
    st = np.sin(theta_val)

    L_int, Lp_int = int(L), int(Lp)
    half_delta = delta / 2.0
    if L_int == La and Lp_int == La:          # "aa" diagonal
        return kcot_a * ct ** 2 + kcot_b * st ** 2 / p2 ** (2)#* p2 ** (-delta)
    elif L_int == Lb and Lp_int == Lb:        # "bb" diagonal
        return kcot_b * ct ** 2 + kcot_a * st ** 2 * p2 ** 2#delta
    elif {L_int, Lp_int} == {La, Lb}:         # off-diagonal
        return ct * st * (kcot_a * p2  - kcot_b / p2 )#ct * st * (kcot_a * p2 ** half_delta - kcot_b * p2 ** (-half_delta))
    else:
        return 0.0


def _evaluate_k_matrix(mode, coeffs, p2, Ecm_over_mref=None, MN=None, MK=None,
                       L=None, Lp=None, param_groups=None, **kwargs):
    mode_key = str(mode or "ere").strip().lower()
    # print(mode_key)
    # Special case: zero K-matrix (channel enabled but contributes 0)
    if mode_key == "zero":
        return 0.0
    # Coupled-channel modes
    if mode_key == "coupled":
        return _k_matrix_coupled(coeffs, L, Lp, p2, Ecm_over_mref, MN, MK,
                                 param_groups=param_groups,
                                 k_matrix_modes=kwargs.get("k_matrix_modes"),
                                 L_values=kwargs.get("L_values"))
    if mode_key in {"polynomial", "poly"}:
        return _k_matrix_polynomial(coeffs, p2)
    if mode_key == "constant":
        return _k_matrix_constant(coeffs, p2)
    if mode_key == "linear":
        return _k_matrix_linear(coeffs, p2)
    if mode_key == "ere":
        return _k_matrix_ere(coeffs, p2)
    if mode_key == 'ere_inv':
        ere_val = _k_matrix_ere(coeffs, p2)
        if ere_val == 0.0:
            raise ValueError("k_matrix='ere_inv' received ere_val=0, causing division by zero.")
        return float(1.0 / ere_val)
    if mode_key == "ere1_vs":
        return _k_matrix_ere1_vs(coeffs,  Ecm_over_mref, MN, MK)
    if mode_key == "ere1_vs_inv":
        ere1_val = _k_matrix_ere1_vs(coeffs,  Ecm_over_mref, MN, MK)
        if ere1_val == 0.0:
            raise ValueError("k_matrix='ere1_vs_inv' received ere1_val=0, causing division by zero.")
        return float(1.0 / ere1_val)
    if mode_key == "epsilon":
        return _k_matrix_epsilon(coeffs, p2)
    if mode_key == "poly_s":
        return _k_matrix_poly_s(coeffs, Ecm_over_mref)
    raise ValueError(
        f"Unsupported k_matrix mode '{mode}'. "
        "Use coupled, ere, ere_inv, ere1_vs, ere1_vs_inv, poly_s, or zero."
    )


def _require_bmat():
    if BMat is None:
        raise ImportError(
            f"BMat import failed: {_BMAT_IMPORT_ERROR}. "
            "Ensure BMat is installed and importable in this Python environment."
        )


def _frame_label_from_psq(psq):
    psq = int(psq)
    if psq == 0:
        return "ar"
    if psq in (1, 4):
        return "oa"
    if psq == 2:
        return "pd"
    if psq == 3:
        return "cd"
    raise ValueError(f"Unsupported |P|^2={psq} for frame label mapping")

# ── PSQ extra step-divisors (tune per-channel as needed) ─────────────────────
# These are multipliers on top of the base epsaux = 0.1/n_refine.
_PSQ_STEP_DIV = {0: 1.0, 1: 3, 2: 10, 3: 10, 4: np.pi}

# ── L-wave extra step-divisors (higher L → finer grid required) ──────────────
# Calibrated so n_refine=5 (the "default working" midpoint of the 1-10 scale)
# gives the correct resolution for each L-wave:
#   L=0 : eff_epsaux = (0.1/5)/8      = 2.5e-3   (S-wave: OkAY)
#   L=1 : eff_epsaux = (0.1/5)/80     = 2.5e-4   (P-wave)
#   L=2 : eff_epsaux = (0.1/5)/800    = 2.5e-5   (D-wave: verified)
#   L=3 : eff_epsaux = (0.1/5)/2400   = 8.3e-6   (F-wave)
#   L=4 : eff_epsaux = (0.1/5)/4000   = 5.0e-6   (G-wave)
# n_refine=1 → 5× coarser (fast/rough),  n_refine=10 → 2× finer (precise)
_L_STEP_DIV   = {0: 8.0, 1: 80.0, 2: 800.0, 3: 2400.0, 4: 4000.0}

# ── tracks which (psq, max_L) combos have already had their grid step printed ─
_GRID_PRINTED = set()


def _initial_guesses_for_psq(psq, irrep, nsols, epsaux, data_ecm=None,
                             max_L=0, step_mode='adaptive', n_refine=1):
    """
    Return (guesses, steps) for the Ecm root-finding scan.

    Parameters
    ----------
    psq       : int   – |P|^2 of the frame
    irrep     : str   – irrep name (used for the T1g special case at PSQ=0)
    nsols     : int   – number of solutions expected
    epsaux    : float – base step size from the caller (= 0.1 / n_refine)
    data_ecm  : array-like or None
    max_L     : int   – max angular momentum in active channels
    step_mode : 'adaptive' (default)
                    Divides epsaux by PSQ- and L-dependent factors from
                    _PSQ_STEP_DIV / _L_STEP_DIV.  ``n_refine`` is already
                    baked into epsaux, so this adds only the channel factors.
                'uniform'
                    Uses epsaux as-is (channel factors = 1).
                    ``n_refine`` alone controls the grid density.
    n_refine  : int – same value as the caller's root-finding n_refine; not
                    used directly here (epsaux = 0.1/n_refine already encodes
                    it), kept for debug logging only.
    """
    psq   = int(psq)
    max_L = int(max_L)

    if step_mode == 'uniform':
        eff_eps = epsaux          # n_refine is already in epsaux
    else:  # 'adaptive'
        psq_div = _PSQ_STEP_DIV.get(psq, 1.0)
        l_div   = _L_STEP_DIV.get(max_L, float(max_L) * 3.0 if max_L > 4 else 1.0)
        eff_eps = epsaux / (psq_div * l_div)

    # ── initial Ecm guesses ───────────────────────────────────────────────────
    if irrep == 'T1g' and psq == 0:
        guesses = [1.98] + [-1] * nsols
    else:
        guesses = [2.02] + [-1] * nsols

    steps = [eff_eps] * max(1, nsols)

    min_guess_len = nsols + 1
    if len(guesses) < min_guess_len:
        guesses.extend([-1] * (min_guess_len - len(guesses)))
    if len(steps) < max(1, nsols):
        steps.extend([eff_eps] * (max(1, nsols) - len(steps)))

    return guesses, steps

def _calculate_noninteracting_energies(self, mom_vectors, lattice_size, massN):
        """Calculate non-interacting energy levels for comparison."""
        noninteracting_energies = {}
        two_pi_over_L = 2 * np.pi / lattice_size
        
        for mom_vec in mom_vectors:
            mom_key = tuple(mom_vec.tolist())
            if mom_key not in noninteracting_energies:
                # Calculate single-particle energies
                psq = np.sum(np.array(mom_vec)**2)
                E_single = np.sqrt(massN**2 + psq * (two_pi_over_L)**2)
                # Non-interacting two-particle energy in CM frame
                noninteracting_energies[mom_key] = 2 * E_single / massN
                
        return noninteracting_energies

    
def _findspectrum_irrep(epsaux, nsols, mom, mL, par, irrep, MN, MK,
                        data_ecm=None, debug=True, energy_aware=False,
                        step_mode='adaptive', n_refine=1, mN_err=None):
    from scipy.optimize import brentq

    _require_bmat()
    psq         = int(np.sum(np.asarray(mom) ** 2))
    frame_label = _frame_label_from_psq(psq)
    qn_channels = _normalize_qn_channels()

    max_L = 0
    for ch in qn_channels:
        if ch["enabled"]:
            lv = ch.get("L_values")
            if lv is not None:
                for v in lv: max_L = max(max_L, int(v))
            else:
                max_L = max(max_L, int(ch.get("L", 0)))
    max_L_channels = max_L   # true max L in active channels (before BMAT +1)
    max_L         += 1       # BMAT requires max_L + 1

    # ── build BoxQuantization fresh each call (C++ object holds internal state
    #    that is not safe to reuse across minimiser steps) ─────────────────────
    calc_func = lambda J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF: calcFunc_param(
        J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF, MN, MK, *par
    )
    kinv     = BMat.KMatrix(calc_func, isZero)
    chan_list = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
    box_q    = BMat.BoxQuantization(frame_label, psq, irrep, chan_list, [max_L], kinv, True)
    box_q.setRefMassL(mL)
    box_q.setMassesOverRef(0, MN, MK)

    _MU = -1.0
    def faux(e):
        try:
            return float(box_q.getOmegaFromEcm(_MU, float(e)))
        except Exception:
            return np.nan

    # ── data levels ───────────────────────────────────────────────────────────
    if data_ecm is not None and len(data_ecm) > 0:
        d_sorted = np.sort(np.asarray(data_ecm, dtype=float).ravel())
    else:
        d_sorted = np.array([])

    # ── effective grid step (scan resolution, controlled by n_refine + L/PSQ divisors) ──
    _, eff_steps = _initial_guesses_for_psq(
        psq, irrep, nsols, epsaux,
        data_ecm=d_sorted if len(d_sorted) else None,
        max_L=max_L_channels,
        step_mode=step_mode,
        n_refine=n_refine,
    )
    eff_epsaux = eff_steps[0] if eff_steps else epsaux

    # ── NI levels (generate wide first — needed before ni_buffer) ────────────
    ni_levels_wide = _generate_ni_levels(mL, mom, float(d_sorted[-1]) + 0.3 if len(d_sorted) else 2.5)
    ni_arr_wide    = np.asarray(ni_levels_wide, dtype=float)

    # ── NI buffer: physical distance to keep away from NI walls ──────────────
    # fast + mN_err  → physical buffer from NI sensitivity to mN uncertainty
    # uniform        → 5*epsaux  (original, known-good)
    # adaptive/fast  → 5*eff_epsaux, shrinks as n_refine increases, capped at
    #                  min_ni_gap/4 for finely-spaced NI pairs
    if step_mode == 'fast' and mN_err is not None and float(mN_err) > 0:
        # Fast mode only: use physically-motivated buffer from mN uncertainty.
        # mL = mN*L, so δmL/mL = δmN/MN; compute NI levels at mL±δmL and
        # take the largest resulting NI level shift.
        _frac     = float(mN_err) / float(MN)
        _ecm_ceil = (float(d_sorted[-1]) + 0.5) if len(d_sorted) else 3.0
        _ni_p     = np.asarray(_generate_ni_levels(mL * (1.0 + _frac), mom, _ecm_ceil))
        _ni_m     = np.asarray(_generate_ni_levels(mL * (1.0 - _frac), mom, _ecm_ceil))
        _shifts   = []
        for _i, _en in enumerate(ni_arr_wide):
            if _i < len(_ni_p):
                _shifts.append(abs(float(_ni_p[_i]) - float(_en)))
            if _i < len(_ni_m):
                _shifts.append(abs(float(_ni_m[_i]) - float(_en)))
        _ni_buffer_phys = (max(_shifts) if _shifts else 5 * eff_epsaux) / 10.0
        if len(ni_arr_wide) >= 2:
            _min_ni_gap = float(np.min(np.diff(ni_arr_wide)))
            ni_buffer   = min(_ni_buffer_phys, _min_ni_gap / 4.0)
        else:
            ni_buffer   = _ni_buffer_phys
        ni_buffer = max(ni_buffer, 1e-6)   # floor: must not evaluate on the pole
    elif step_mode == 'uniform':
        ni_buffer = 5 * epsaux
    else:
        # adaptive (and fast without mN_err): scale with eff_epsaux so the
        # buffer shrinks as n_refine increases, capped at min_ni_gap/4.
        _base_buffer = 5 * eff_epsaux
        if len(ni_arr_wide) >= 2:
            _min_ni_gap = float(np.min(np.diff(ni_arr_wide)))
            ni_buffer   = _base_buffer#min(_base_buffer, _min_ni_gap / 4.0)
        else:
            ni_buffer = _base_buffer

    _grid_key = (psq, max_L_channels)
    if _grid_key not in _GRID_PRINTED:
        _GRID_PRINTED.add(_grid_key)
        print(f"  [grid] PSQ={psq}  max_L={max_L_channels}  mode={step_mode!r}  "
              f"epsaux={epsaux:.3e}  eff_epsaux={eff_epsaux:.3e}  ni_buffer={ni_buffer:.3e}")
    if debug:
        _progress_log(
            f"  epsaux={epsaux:.3e}  eff_epsaux={eff_epsaux:.3e}  ni_buffer={ni_buffer:.3e}"
            f"  (PSQ={psq}, max_L={max_L_channels}, mode={step_mode!r})"
        )

    ni_arr = ni_arr_wide

    if len(d_sorted) > 0:
        _, _, ni_first_below, _ = _nearest_ni(float(d_sorted[0]),  ni_levels_wide)
        _, _, _, ni_last_above  = _nearest_ni(float(d_sorted[-1]), ni_levels_wide)

        ecm_lo = (ni_first_below + ni_buffer) if ni_first_below is not None \
                 else float(d_sorted[0]) - 0.03
        ecm_hi = (ni_last_above  - ni_buffer) if ni_last_above  is not None \
                 else float(d_sorted[-1]) + 0.03
    else:
        ecm_lo, ecm_hi = 2.001, 2.30

    # trim NI list to the actual range we care about
    ni_levels = [e for e in ni_levels_wide if ecm_lo - 0.05 <= e <= ecm_hi + 0.05]
    ni_arr    = np.asarray(ni_levels, dtype=float)

    if debug:
        _progress_log(
            f"\n{'='*70}\n"
            f"  irrep={irrep}  P²={psq}  energy_aware={energy_aware}\n"
            f"  data : {[f'{e:.5f}' for e in d_sorted]}\n"
            f"  NI   : {[f'{e:.5f}' for e in ni_levels]}\n"
            f"  range: [{ecm_lo:.5f}, {ecm_hi:.5f}]\n"
            f"{'='*70}"
        )

    # ── window builder ────────────────────────────────────────────────────────
    def _window(i_lev, e_data):
        _, _, ni_below, ni_above = _nearest_ni(e_data, ni_levels)

        # NI walls use ni_buffer (physical, independent of grid step)
        lo = (ni_below + ni_buffer) if ni_below is not None else ecm_lo
        hi = (ni_above - ni_buffer) if ni_above is not None else ecm_hi

        if energy_aware and len(d_sorted) > 1:
            mask_below = d_sorted < e_data
            if ni_below is not None:
                mask_below &= d_sorted > ni_below
            if mask_below.any():
                lo = max(lo, float(d_sorted[mask_below][-1]) + 2 * eff_epsaux)

            mask_above = d_sorted > e_data
            if ni_above is not None:
                mask_above &= d_sorted < ni_above
            if mask_above.any():
                hi = min(hi, float(d_sorted[mask_above][0]) - 2 * eff_epsaux)

        hi = max(hi, lo + 10 * eff_epsaux)   # minimum viable width only
        return float(lo), float(hi)

    # ── residual tolerance for real-zero check ────────────────────────────────
    if step_mode == 'fast':
        # Quick 20-point survey instead of 300 — gives a proper scale-aware
        # tolerance without the full scan cost.  Critical for D-wave (and higher
        # L) where faux() ~ p^(2L+1) and a fixed 1e-9 rejects valid roots near
        # NI poles whose |faux| at brentq convergence can be >> 1e-9.
        _sample_pts = []
        for _e in np.linspace(ecm_lo, ecm_hi, 20):
            if len(ni_arr) == 0 or np.all(np.abs(_e - ni_arr) > ni_buffer):
                _v = abs(faux(_e))
                if np.isfinite(_v) and _v > 0:
                    _sample_pts.append(_v)
        _omega_scale  = float(np.nanmedian(_sample_pts)) if _sample_pts else 1e-4
        _residual_tol = 1e-3 * _omega_scale + 1e-11
        if debug:
            _progress_log(f"  [fast] omega_scale={_omega_scale:.3e}  residual_tol={_residual_tol:.3e}")
    else:
        _sample_pts = []
        if len(ni_arr) > 0:
            for _e in np.linspace(ecm_lo, ecm_hi, 300):
                if np.all(np.abs(_e - ni_arr) > ni_buffer):
                    _sample_pts.append(abs(faux(_e)))
        _omega_scale  = float(np.nanmedian(_sample_pts)) if _sample_pts else 1e-4
        _residual_tol = 1e-3 * _omega_scale + 1e-11

        if debug:
            _progress_log(f"  omega_scale={_omega_scale:.3e}  residual_tol={_residual_tol:.3e}")

    # ── main loop ─────────────────────────────────────────────────────────────
    merge_tol     = 6 * eff_epsaux
    solutions     = []
    used_brackets = set()

    if step_mode == 'uniform':
        # ── UNIFORM path: single-pass np.arange scan (original, known-good) ───
        for i_lev, e_data in enumerate(d_sorted[:nsols]):
            grid_lo, grid_hi = _window(i_lev, e_data)

            e_arr  = np.arange(grid_lo, grid_hi + eff_epsaux * 0.5, eff_epsaux)
            omegas = np.array([faux(e) for e in e_arr])

            if debug:
                finite = np.isfinite(omegas)
                _progress_log(
                    f"\n  [level {i_lev+1}/{nsols}]  data={e_data:.5f}\n"
                    f"    window=[{grid_lo:.5f},{grid_hi:.5f}]  "
                    f"pts={len(e_arr)}  finite={finite.sum()}\n"
                    f"    Ω range=[{np.nanmin(omegas):.3e},{np.nanmax(omegas):.3e}]"
                )

            brackets = []
            for k in range(len(e_arr) - 1):
                ok, oj = omegas[k], omegas[k+1]
                if np.isfinite(ok) and np.isfinite(oj) and ok * oj < 0.0:
                    mid = 0.5 * (e_arr[k] + e_arr[k+1])
                    brackets.append((e_arr[k], e_arr[k+1], mid))

            if debug:
                _progress_log(
                    f"    sign changes: {len(brackets)}  "
                    f"mids={[f'{b[2]:.5f}' for b in brackets]}"
                )

            if not brackets:
                if debug:
                    _progress_log(f"    WARNING: no crossing for data={e_data:.5f}")
                continue

            _, _, ni_below_level, _ = _nearest_ni(e_data, ni_levels)
            prev_root = solutions[-1] if solutions else -np.inf

            if i_lev == 0:
                if ni_below_level is not None:
                    target_first = float(ni_below_level) + 6 * epsaux
                    brackets.sort(key=lambda b: (abs(b[2] - target_first), b[2]))
                else:
                    brackets.sort(key=lambda b: (b[2], abs(b[2] - e_data)))
            else:
                brackets.sort(
                    key=lambda b: (
                        b[2] <= (prev_root + merge_tol),
                        abs(b[2] - e_data),
                        b[2],
                    )
                )

            found = False
            for (ea, eb, mid) in brackets:
                key = (round(ea, 9), round(eb, 9))
                if key in used_brackets:
                    continue

                e_fine = np.linspace(ea, eb, 200)
                f_fine = np.array([faux(e) for e in e_fine])

                tight_a = tight_b = None
                for k in range(len(e_fine) - 1):
                    if (np.isfinite(f_fine[k]) and np.isfinite(f_fine[k+1])
                            and f_fine[k] * f_fine[k+1] < 0.0):
                        tight_a, tight_b = e_fine[k], e_fine[k+1]
                        break

                if tight_a is None:
                    continue

                try:
                    e_root = brentq(faux, tight_a, tight_b,
                                    xtol=1e-12, rtol=1e-12, maxiter=500)
                    f_root = abs(faux(e_root))

                    if f_root > _residual_tol:
                        if debug:
                            _progress_log(
                                f"    [pole rejected]  Ecm={e_root:.8f}  "
                                f"|Ω|={f_root:.3e} > tol={_residual_tol:.3e}"
                            )
                        continue

                    if any(abs(e_root - z) < merge_tol for z in solutions):
                        if debug:
                            _progress_log(
                                f"    [degenerate skip] Ecm={e_root:.9f} is within {merge_tol:.3e} "
                                "of an already selected root"
                            )
                        continue

                    solutions.append(float(e_root))
                    used_brackets.add(key)
                    found = True
                    if debug:
                        _progress_log(
                            f"    [✓] Ecm={e_root:.9f}  "
                            f"|Ω|={f_root:.3e}  "
                            f"Δdata={abs(e_root - e_data):.5f}"
                        )
                    break
                except Exception as exc:
                    if debug:
                        _progress_log(f"    brentq failed: {exc}")
                    continue

            if not found and debug:
                _progress_log(f"    WARNING: no root found for level {i_lev+1}")

    elif step_mode == 'adaptive':
        # ── ADAPTIVE path: two-pass coarse→fine scan ───────────────────────────
        # Pass 1 (coarse): scan at coarse_step = eff_epsaux * COARSE_FACTOR,
        #   dropping to eff_epsaux near NI walls; detects sign-change brackets.
        # Pass 2 (fine): re-scan each coarse bracket at eff_epsaux to get a
        #   tight sub-bracket for brentq.
        COARSE_FACTOR = 8
        ni_zone       = ni_buffer * 0.5

        if len(ni_arr) >= 2:
            min_ni_gap = float(np.min(np.diff(np.sort(ni_arr))))
            max_coarse = min_ni_gap / 4.0
        else:
            max_coarse = 1.0

        for i_lev, e_data in enumerate(d_sorted[:nsols]):
            grid_lo, grid_hi = _window(i_lev, e_data)

            # ── Pass 1: coarse scan ───────────────────────────────────────────
            coarse_step = min(eff_epsaux * COARSE_FACTOR, max_coarse)
            e_pts_c = []
            e = grid_lo
            while e <= grid_hi + coarse_step * 0.5:
                e_pts_c.append(min(e, grid_hi))
                if len(ni_arr) > 0 and np.any(np.abs(e - ni_arr) < ni_zone):
                    e += eff_epsaux
                else:
                    e += coarse_step
            e_arr_c  = np.array(e_pts_c)
            omegas_c = np.array([faux(e) for e in e_arr_c])

            coarse_brackets = []
            for k in range(len(e_arr_c) - 1):
                ok, oj = omegas_c[k], omegas_c[k + 1]
                if np.isfinite(ok) and np.isfinite(oj) and ok * oj < 0.0:
                    coarse_brackets.append((e_arr_c[k], e_arr_c[k + 1]))

            # ── Pass 2: fine re-scan inside each coarse bracket ───────────────
            omegas_f_map = {}
            for e, o in zip(e_arr_c, omegas_c):
                omegas_f_map[round(e, 14)] = o

            brackets = []
            for (ca, cb) in coarse_brackets:
                e_fine = np.arange(ca, cb + eff_epsaux * 0.5, eff_epsaux)
                f_fine_vals = []
                for ef in e_fine:
                    key_f = round(ef, 14)
                    if key_f in omegas_f_map:
                        f_fine_vals.append(omegas_f_map[key_f])
                    else:
                        val = faux(ef)
                        omegas_f_map[key_f] = val
                        f_fine_vals.append(val)
                for k in range(len(e_fine) - 1):
                    ok2, oj2 = f_fine_vals[k], f_fine_vals[k + 1]
                    if np.isfinite(ok2) and np.isfinite(oj2) and ok2 * oj2 < 0.0:
                        mid = 0.5 * (e_fine[k] + e_fine[k + 1])
                        brackets.append((e_fine[k], e_fine[k + 1], mid))

            # fallback: coarse missed everything → full fine scan
            if not coarse_brackets:
                e_arr_f  = np.arange(grid_lo, grid_hi + eff_epsaux * 0.5, eff_epsaux)
                omegas_f = np.array([
                    omegas_f_map.get(round(e, 14), faux(e)) for e in e_arr_f
                ])
                for k in range(len(e_arr_f) - 1):
                    ok2, oj2 = omegas_f[k], omegas_f[k + 1]
                    if np.isfinite(ok2) and np.isfinite(oj2) and ok2 * oj2 < 0.0:
                        mid = 0.5 * (e_arr_f[k] + e_arr_f[k + 1])
                        brackets.append((e_arr_f[k], e_arr_f[k + 1], mid))

            e_arr  = e_arr_c
            omegas = omegas_c

            if debug:
                finite = np.isfinite(omegas)
                _progress_log(
                    f"\n  [level {i_lev+1}/{nsols}]  data={e_data:.5f}\n"
                    f"    window=[{grid_lo:.5f},{grid_hi:.5f}]  "
                    f"pts={len(e_arr)}  finite={finite.sum()}\n"
                    f"    Ω range=[{np.nanmin(omegas):.3e},{np.nanmax(omegas):.3e}]"
                )

            if debug:
                _progress_log(
                    f"    sign changes: {len(brackets)}  "
                    f"mids={[f'{b[2]:.5f}' for b in brackets]}"
                )

            if not brackets:
                if debug:
                    _progress_log(f"    WARNING: no crossing for data={e_data:.5f}")
                continue

            _, _, ni_below_level, _ = _nearest_ni(e_data, ni_levels)
            prev_root = solutions[-1] if solutions else -np.inf

            if i_lev == 0:
                if ni_below_level is not None:
                    target_first = float(ni_below_level) + ni_buffer
                    brackets.sort(key=lambda b: (abs(b[2] - target_first), b[2]))
                else:
                    brackets.sort(key=lambda b: (b[2], abs(b[2] - e_data)))
            else:
                brackets.sort(
                    key=lambda b: (
                        b[2] <= (prev_root + merge_tol),
                        abs(b[2] - e_data),
                        b[2],
                    )
                )

            found = False
            for (ea, eb, mid) in brackets:
                key = (round(ea, 9), round(eb, 9))
                if key in used_brackets:
                    continue

                fa, fb = faux(ea), faux(eb)
                if not (np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0):
                    continue
                tight_a, tight_b = ea, eb

                try:
                    e_root = brentq(faux, tight_a, tight_b,
                                    xtol=1e-12, rtol=1e-12, maxiter=500)
                    f_root = abs(faux(e_root))

                    if f_root > _residual_tol:
                        if debug:
                            _progress_log(
                                f"    [pole rejected]  Ecm={e_root:.8f}  "
                                f"|Ω|={f_root:.3e} > tol={_residual_tol:.3e}"
                            )
                        continue

                    if any(abs(e_root - z) < merge_tol for z in solutions):
                        if debug:
                            _progress_log(
                                f"    [degenerate skip] Ecm={e_root:.9f} is within {merge_tol:.3e} "
                                "of an already selected root"
                            )
                        continue

                    solutions.append(float(e_root))
                    used_brackets.add(key)
                    found = True
                    if debug:
                        _progress_log(
                            f"    [✓] Ecm={e_root:.9f}  "
                            f"|Ω|={f_root:.3e}  "
                            f"Δdata={abs(e_root - e_data):.5f}"
                        )
                    break
                except Exception as exc:
                    if debug:
                        _progress_log(f"    brentq failed: {exc}")
                    continue

            if not found and debug:
                _progress_log(f"    WARNING: no root found for level {i_lev+1}")

    elif step_mode == 'fast':
        # ── FAST path: coarse probe-and-bisect per level ──────────────────────
        # Instead of scanning at eff_epsaux resolution (10–2000+ evals/level),
        # probe with 8→16→32 intervals, caching evaluations across rounds so
        # doubling the probe count only evaluates the new midpoints.
        # Then run brentq directly from the coarse bracket.
        # Typical cost: ~30–80 faux() calls per level vs 100–2000 for adaptive.
        # Works because each NI sub-interval contains exactly one physical level.
        # Probe sizes double each round until a sign change is found or the
        # ceiling (max(32, n_refine)) is reached.  n_refine is honoured so
        # the user can increase resolution via the JSON config.
        _fast_ceil = max(32, int(n_refine))
        FAST_PROBE_SIZES = []
        _fp = 8
        while _fp < _fast_ceil:
            FAST_PROBE_SIZES.append(_fp)
            _fp *= 2
        FAST_PROBE_SIZES.append(_fast_ceil)  # always include the ceiling
        FAST_PROBE_SIZES = sorted(set(FAST_PROBE_SIZES))

        for i_lev, e_data in enumerate(d_sorted[:nsols]):
            grid_lo, grid_hi = _window(i_lev, e_data)

            # Per-level eval cache: linspace(n+1) ⊆ linspace(2n+1) at double
            # size, so later rounds only pay for their NEW midpoints.
            _eval_cache = {}

            def _feval(e):
                k = round(e, 14)
                if k not in _eval_cache:
                    _eval_cache[k] = faux(e)
                return _eval_cache[k]

            brackets = []
            for n_int in FAST_PROBE_SIZES:
                e_probe = np.linspace(grid_lo, grid_hi, n_int + 1)
                f_probe = [_feval(e) for e in e_probe]
                for k in range(len(e_probe) - 1):
                    ok, oj = f_probe[k], f_probe[k + 1]
                    if np.isfinite(ok) and np.isfinite(oj) and ok * oj < 0.0:
                        mid = 0.5 * (e_probe[k] + e_probe[k + 1])
                        brackets.append((e_probe[k], e_probe[k + 1], mid))
                if brackets:
                    break   # stop expanding once a sign-change bracket is found

            if debug:
                _progress_log(
                    f"\n  [level {i_lev+1}/{nsols}]  data={e_data:.5f}\n"
                    f"    window=[{grid_lo:.5f},{grid_hi:.5f}]  "
                    f"probes={len(_eval_cache)}  "
                    f"sign changes: {len(brackets)}  "
                    f"mids={[f'{b[2]:.5f}' for b in brackets]}"
                )

            if not brackets:
                if debug:
                    _progress_log(f"    WARNING: no crossing for data={e_data:.5f}")
                continue

            _, _, ni_below_level, _ = _nearest_ni(e_data, ni_levels)
            prev_root = solutions[-1] if solutions else -np.inf

            if i_lev == 0:
                if ni_below_level is not None:
                    target_first = float(ni_below_level) + ni_buffer
                    brackets.sort(key=lambda b: (abs(b[2] - target_first), b[2]))
                else:
                    brackets.sort(key=lambda b: (b[2], abs(b[2] - e_data)))
            else:
                brackets.sort(
                    key=lambda b: (
                        b[2] <= (prev_root + merge_tol),
                        abs(b[2] - e_data),
                        b[2],
                    )
                )

            found = False
            for (ea, eb, mid) in brackets:
                key = (round(ea, 9), round(eb, 9))
                if key in used_brackets:
                    continue
                try:
                    e_root = brentq(faux, ea, eb, xtol=1e-12, rtol=1e-12, maxiter=500)
                    f_root = abs(faux(e_root))

                    if f_root > _residual_tol:
                        if debug:
                            _progress_log(
                                f"    [pole rejected]  Ecm={e_root:.8f}  "
                                f"|Ω|={f_root:.3e} > tol={_residual_tol:.3e}"
                            )
                        continue

                    if any(abs(e_root - z) < merge_tol for z in solutions):
                        if debug:
                            _progress_log(
                                f"    [degenerate skip] Ecm={e_root:.9f} within "
                                "merge_tol of an already selected root"
                            )
                        continue

                    solutions.append(float(e_root))
                    used_brackets.add(key)
                    found = True
                    if debug:
                        _progress_log(
                            f"    [✓] Ecm={e_root:.9f}  "
                            f"|Ω|={f_root:.3e}  "
                            f"Δdata={abs(e_root - e_data):.5f}"
                        )
                    break
                except Exception as exc:
                    if debug:
                        _progress_log(f"    brentq failed: {exc}")
                    continue

            if not found and debug:
                _progress_log(f"    WARNING: no root found for level {i_lev+1}")

    # ── finalise ──────────────────────────────────────────────────────────────
    solutions = sorted(solutions)
    merged    = []
    final_merge_tol = 3 * epsaux if step_mode == 'uniform' else ni_buffer
    for z in solutions:
        if merged and abs(z - merged[-1]) < final_merge_tol:
            merged[-1] = 0.5 * (merged[-1] + z)
        else:
            merged.append(z)

    if debug:
        _progress_log(
            f"\n  [result] {len(merged)}/{nsols}  "
            f"{[f'{z:.7f}' for z in merged]}"
        )

    # ── soft fallback: fill missing levels instead of raising ─────────────────
    #
    # When params are bad, Omega may not cross zero in the expected window.
    # Instead of raising RuntimeError (which gives a flat 1e12 chi2 the
    # optimizer cannot follow), we fill missing levels with the position of
    # minimum |Omega| in the search window.
    #
    # WHY THIS WORKS:
    #   Omega = K^{-1}(params) - B(Ecm)
    #   The location of min|Omega| shifts as params change → chi2 varies
    #   → Nelder-Mead gets a gradient to climb out of the bad region
    #
    # WHY NOT USE DATA POSITION AS FALLBACK:
    #   That would give residual=0 and chi2=0 for the missing level → wrong
    #
    # WHY NOT RAISE:
    #   RuntimeError → _chi2 catches → returns flat 1e12 → optimizer stuck
    # ─────────────────────────────────────────────────────────────────────────
    if len(merged) < nsols:
        if debug:
            _progress_log(
                f"\n  [soft fallback] {len(merged)}/{nsols} found — "
                f"filling {nsols - len(merged)} missing level(s) with "
                f"min|Ω| positions (gives optimizer a varying signal)"
            )

        already_used = {round(z, 8) for z in merged}

        for i_lev, e_data in enumerate(d_sorted[:nsols]):
            if len(merged) >= nsols:
                break

            # skip levels we already found
            if any(abs(e_data - z) < 10 * epsaux for z in merged):
                continue

            # ── try the normal window first ───────────────────────────────────
            grid_lo, grid_hi = _window(i_lev, e_data)

            # ── if normal window is too narrow, expand symmetrically ──────────
            # This catches sub-threshold levels where the window edge is the
            # boundary itself (e.g. the very first level below the NI pole)
            window_width = grid_hi - grid_lo
            if window_width < 20 * epsaux:
                expand = max(0.05, 5 * epsaux)
                grid_lo = max(ecm_lo, grid_lo - expand)
                grid_hi = min(ecm_hi, grid_hi + expand)
                if debug:
                    _progress_log(
                        f"    [fallback] level {i_lev+1}: window too narrow "
                        f"({window_width:.2e}), expanded to "
                        f"[{grid_lo:.5f}, {grid_hi:.5f}]"
                    )

            # ── also try a wide window spanning the full data range ───────────
            # catches cases where the level moved far from expected position
            wide_lo = ecm_lo
            wide_hi = ecm_hi

            # scan both windows, keep the best (smallest |Omega|)
            best_ecm   = None
            best_omega = np.inf

            for (lo, hi, label) in [
                (grid_lo, grid_hi, "normal"),
                (wide_lo,  wide_hi,  "wide"),
            ]:
                if lo >= hi:
                    continue

                n_scan   = max(300, int((hi - lo) / epsaux * 0.5))
                e_scan   = np.linspace(lo, hi, n_scan)
                o_scan   = np.array([faux(e) for e in e_scan])

                # mask NI poles — large |Omega| spikes are not zeros
                finite   = np.isfinite(o_scan)
                abs_o    = np.where(finite, np.abs(o_scan), np.inf)

                # also reject points very close to NI levels
                if len(ni_arr) > 0:
                    for ni in ni_arr:
                        abs_o[np.abs(e_scan - ni) < 5 * epsaux] = np.inf

                if np.all(np.isinf(abs_o)):
                    continue

                idx_best   = int(np.argmin(abs_o))
                omega_best = float(abs_o[idx_best])

                if omega_best < best_omega:
                    best_omega = omega_best
                    best_ecm   = float(e_scan[idx_best])

                    if debug:
                        _progress_log(
                            f"    [fallback {label}] level {i_lev+1}: "
                            f"min|Ω|={omega_best:.3e} at Ecm={best_ecm:.6f}  "
                            f"(data={e_data:.5f})"
                        )

            # ── last resort: use data position ────────────────────────────────
            # Only when Omega is NaN everywhere (truly degenerate params).
            # This gives chi2=0 contribution for this level which is wrong,
            # but at least prevents a crash — log loudly.
            if best_ecm is None:
                best_ecm = float(e_data)
                if debug:
                    _progress_log(
                        f"    [fallback LAST RESORT] level {i_lev+1}: "
                        f"Omega NaN everywhere, using data position {e_data:.5f}. "
                        f"chi2 contribution will be 0 — params are very unphysical."
                    )

            key = round(best_ecm, 8)
            if key not in already_used:
                merged.append(best_ecm)
                already_used.add(key)
            else:
                # avoid exact duplicate — nudge slightly
                nudged = best_ecm + epsaux
                merged.append(nudged)
                already_used.add(round(nudged, 8))

        merged = sorted(merged)

        if debug:
            _progress_log(
                f"\n  [after fallback] {len(merged)}/{nsols}  "
                f"{[f'{z:.7f}' for z in merged]}"
            )

    return merged[:nsols]

def _findspectrum_sweep(epsaux, nsols, mom, mL, par, irrep, MN, MK,
                         data_ecm=None, debug=True):
    from scipy.optimize import brentq

    _require_bmat()

    psq         = int(np.sum(np.asarray(mom) ** 2))
    frame_label = _frame_label_from_psq(psq)
    qn_channels = _normalize_qn_channels()

    max_L = 0
    for ch in qn_channels:
        if ch["enabled"]:
            lv = ch.get("L_values")
            if lv is not None:
                for v in lv: max_L = max(max_L, int(v))
            else:
                max_L = max(max_L, int(ch.get("L", 0)))
    max_L += 1

    calc_func = lambda J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF: calcFunc_param(
        J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF, MN, MK, *par
    )
    kinv      = BMat.KMatrix(calc_func, isZero)
    chan_list  = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
    box_q     = BMat.BoxQuantization(frame_label, psq, irrep, chan_list, [max_L], kinv, True)
    box_q.setRefMassL(mL)
    box_q.setMassesOverRef(0, MN, MK)

    _MU = -1.0
    def faux(e):
        try:
            return float(box_q.getOmegaFromEcm(_MU, float(e)))
        except Exception:
            return np.nan

    # ── range ─────────────────────────────────────────────────────────────────
    if data_ecm is not None and len(data_ecm) > 0:
        d_sorted  = np.sort(np.asarray(data_ecm, dtype=float).ravel())
        ecm_start = float(d_sorted[0])  - 0.01
        ecm_end   = float(d_sorted[-1]) + 0.01
    else:
        ecm_start, ecm_end = 2.001, 2.40

    ni_levels = _generate_ni_levels(mL, mom, ecm_end + 0.2)
    ni_arr    = np.asarray(ni_levels, dtype=float)
    ni_tol    = 3*epsaux #max(5 * epsaux, 0.002)          # how close = "is a NI level"

    def _is_ni(e):
        return len(ni_arr) > 0 and bool(np.any(np.abs(ni_arr - e) < ni_tol))

    if debug:
        _progress_log(
            f"\n{'='*60}\n"
            f"  irrep={irrep}  P²={psq}\n"
            f" ni tol={ni_tol:.3f}\n"
            f"  data : {[f'{v:.5f}' for v in d_sorted]}\n"
            f"  sweep [{ecm_start:.5f}, {ecm_end:.5f}]  step={epsaux}\n"
            f"  NI   : {[f'{v:.5f}' for v in ni_levels]}\n"
            f"{'='*60}"
        )

    # ── sweep ─────────────────────────────────────────────────────────────────
    solutions = []
    e_pts  = np.arange(ecm_start, ecm_end, epsaux)
    omegas = np.array([faux(e) for e in e_pts])

    for k in range(len(e_pts) - 1):
        if len(solutions) >= nsols:
            break

        fa, fb = omegas[k], omegas[k + 1]
        ea, eb = e_pts[k],  e_pts[k + 1]

        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if fa * fb >= 0.0:
            continue
        # sign change found — refine with brentq
        _progress_log(f"\n  [sign change] Ecm in [{ea:.5f}, {eb:.5f}]  Ω in [{fa:.3e}, {fb:.3e}]")
        try:
            root = brentq(faux, ea, eb, xtol=1e-10, rtol=1e-10, maxiter=200)
        except Exception:
            continue

        # # check residual (pole crossings have huge |Omega|)
        # if abs(faux(root)) > 1e-4:
        #     if debug:
        #         _progress_log(f"  epsaux={epsaux:.7f} ")
        #         _progress_log(f"  [pole skip]  Ecm={root:.7f}  |Ω|={abs(faux(root)):.3e}")
        #     continue

        # check if it's a non-interacting energy
        if _is_ni(root):
            if debug:
                _progress_log(f"  [NI  skip]   Ecm={root:.7f}")
            continue

        # check not a duplicate
        if solutions and abs(root - solutions[-1]) < 3 * epsaux:
            continue

        solutions.append(float(root))
        if debug:
            _progress_log(
                f"  [✓] Ecm={root:.9f}  |Ω|={abs(faux(root)):.3e}  "
                f"({len(solutions)}/{nsols})"
            )

    if debug:
        _progress_log(f"\n  [result] {len(solutions)}/{nsols}  {[f'{z:.7f}' for z in solutions]}")

    return solutions[:nsols]

def _findspectrum_simple(epsaux, nsols, mom, mL, par, irrep, MN, MK,
                         data_ecm=None, debug=True, beta=7.0,
                         save_dir="./omega_debug"):
    from scipy.optimize import fsolve

    _require_bmat()

    psq         = int(np.sum(np.asarray(mom) ** 2))
    frame_label = _frame_label_from_psq(psq)
    qn_channels = _normalize_qn_channels()

    max_L = 0
    for ch in qn_channels:
        if ch["enabled"]:
            lv = ch.get("L_values")
            if lv is not None:
                for v in lv: max_L = max(max_L, int(v))
            else:
                max_L = max(max_L, int(ch.get("L", 0)))
    max_L += 1

    calc_func = lambda J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF: calcFunc_param(
        J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF, MN, MK, *par
    )
    kinv      = BMat.KMatrix(calc_func, isZero)
    chan_list  = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
    box_q     = BMat.BoxQuantization(frame_label, psq, irrep, chan_list, [max_L], kinv, True)
    box_q.setRefMassL(mL)
    box_q.setMassesOverRef(0, MN, MK)

    _MU = -1.0
    def faux(e):
        try:
            return float(box_q.getOmegaFromEcm(_MU, float(e)))
        except Exception:
            return np.nan

    # ── range ─────────────────────────────────────────────────────────────────
    if data_ecm is not None and len(data_ecm) > 0:
        d_sorted  = np.sort(np.asarray(data_ecm, dtype=float).ravel())
        ecm_start = float(d_sorted[0])  - 0.10
        ecm_end   = float(d_sorted[-1]) + 0.10
    else:
        ecm_start, ecm_end = 2.001, 2.40

    ni_levels = _generate_ni_levels(mL, mom, ecm_end + 0.2)
    ni_arr    = np.asarray(ni_levels, dtype=float)
    ni_tol    = max(5 * epsaux, 0.002)

    def _is_ni(e):
        return len(ni_arr) > 0 and bool(np.any(np.abs(ni_arr - e) < ni_tol))

    # ── tanh root finder ──────────────────────────────────────────────────────
    def _tanh_root(ea, eb):
        """
        Map [ea, eb] → t in R via  x(t) = mid + half*tanh(beta*t)
        then fsolve for t* where Omega(x(t*)) = 0.
        Returns (x_root, success).
        """
        mid  = 0.5 * (ea + eb)
        half = 0.5 * (eb - ea)

        def g(t_arr):
            x = mid + half * np.tanh(beta * float(t_arr[0]))
            v = faux(x)
            return [v if np.isfinite(v) else 1e6]

        try:
            t_sol, _, ier, _ = fsolve(g, [0.0], full_output=True)
            if ier != 1:
                return None, False
            x_root = float(mid + half * np.tanh(beta * float(t_sol[0])))
            if not (ea <= x_root <= eb):
                return None, False
            residual = abs(faux(x_root))
            if residual > 1e-4:
                return None, False
            return x_root, True
        except Exception:
            return None, False

    # ── sweep ─────────────────────────────────────────────────────────────────
    e_pts  = np.arange(ecm_start, ecm_end, epsaux)
    omegas = np.array([faux(e) for e in e_pts])

    if debug:
        _progress_log(
            f"\n{'='*60}\n"
            f"  irrep={irrep}  P²={psq}\n"
            f"  sweep [{ecm_start:.5f}, {ecm_end:.5f}]  step={epsaux}  beta={beta}\n"
            f"  NI   : {[f'{v:.5f}' for v in ni_levels]}\n"
            f"{'='*60}"
        )
        # ── omega debug plot ──────────────────────────────────────────────────
        import matplotlib.pyplot as plt
        import os

        os.makedirs(save_dir, exist_ok=True)

        omega_plot = omegas.copy().astype(float)

        # break lines at NI poles before computing scale
        for ni in ni_arr:
            omega_plot[np.abs(e_pts - ni) < ni_tol * 2] = np.nan

        # ── auto y-scale: use inter-pole finite values only ───────────────────
        finite_vals = omega_plot[np.isfinite(omega_plot)]
        if len(finite_vals) > 0:
            p_lo = float(np.percentile(np.abs(finite_vals), 80))   # ignore top 20 %
            _clip = max(p_lo * 2.0, 0.05)                          # at least ±0.05
        else:
            _clip = 1.0

        # now hard-clip
        omega_plot[np.abs(omega_plot) > _clip] = np.nan

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(e_pts, omega_plot, color="steelblue", lw=1.6)
        ax.axhline(0, color="gray", lw=0.8, ls="--")

        # NI vertical lines
        for ni in ni_arr:
            if ecm_start <= ni <= ecm_end:
                ax.axvline(ni, color="gray", lw=1.0, ls=":", alpha=0.7, label="NI")

        # data levels
        if data_ecm is not None:
            for de in d_sorted:
                ax.axvline(de, color="tomato", lw=1.0, ls="--", alpha=0.8, label="data")

        ax.set_xlabel(r"$E_{cm}/m_N$", fontsize=13)
        ax.set_ylabel(r"$\Omega$",      fontsize=13)
        ax.set_title(f"Ω sweep — {irrep}  P²={psq}  (y clipped at ±{_clip:.3f})", fontsize=12)
        ax.set_ylim(-_clip, _clip)
        ax.set_xlim(ecm_start, ecm_end)
        ax.grid(True, alpha=0.25)

        # deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9)

        plt.tight_layout()
        safe  = irrep.replace("/", "_")
        fpath = os.path.join(save_dir, f"omega_{safe}_psq{psq}.pdf")
        plt.savefig(fpath, bbox_inches="tight", dpi=150)
        plt.close(fig)
        _progress_log(f"  [debug plot] saved: {fpath}  (y ±{_clip:.4f})")
    # ── walk forward through sign changes ─────────────────────────────────────
    solutions = []

    for k in range(len(e_pts) - 1):
        if len(solutions) >= nsols:
            break

        fa, fb = omegas[k], omegas[k + 1]
        ea, eb = e_pts[k],  e_pts[k + 1]

        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if fa * fb >= 0.0:
            continue

        # ── tanh refinement ───────────────────────────────────────────────────
        root, ok = _tanh_root(ea, eb)

        if not ok:
            if debug:
                _progress_log(f"  [tanh fail] bracket=[{ea:.6f},{eb:.6f}]")
            continue

        if _is_ni(root):
            if debug:
                _progress_log(f"  [NI  skip]  Ecm={root:.7f}")
            continue

        if solutions and abs(root - solutions[-1]) < 3 * epsaux:
            continue

        solutions.append(float(root))
        if debug:
            _progress_log(
                f"  [✓] Ecm={root:.9f}  |Ω|={abs(faux(root)):.3e}  "
                f"({len(solutions)}/{nsols})"
            )

    if debug:
        _progress_log(
            f"\n  [result] {len(solutions)}/{nsols}  "
            f"{[f'{z:.7f}' for z in solutions]}"
        )

    return solutions[:nsols]


def _findspectrum_lambda(epsaux, nsols, mom, mL, par, irrep, MN, MK,
                         data_ecm=None, debug=True):
    from scipy.optimize import brentq, minimize_scalar
    from scipy.signal import find_peaks

    _require_bmat()

    psq         = int(np.sum(np.asarray(mom) ** 2))
    frame_label = _frame_label_from_psq(psq)
    qn_channels = _normalize_qn_channels()

    max_L = 0
    for ch in qn_channels:
        if ch["enabled"]:
            lv = ch.get("L_values")
            if lv is not None:
                for v in lv: max_L = max(max_L, int(v))
            else:
                max_L = max(max_L, int(ch.get("L", 0)))
    max_L += 1

    calc_func = lambda J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF: calcFunc_param(
        J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF, MN, MK, *par
    )
    kinv      = BMat.KMatrix(calc_func, isZero)
    chan_list  = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
    box_q     = BMat.BoxQuantization(frame_label, psq, irrep, chan_list, [max_L], kinv, True)
    box_q.setRefMassL(mL)
    box_q.setMassesOverRef(0, MN, MK)

    # ── two views of the same quantity ────────────────────────────────────────
    # abs_min : positive, for find_peaks detection
    # sgn_min : signed,   for brentq (sign change = true zero)
    def _get_eigs(e):
        try:
            return box_q.getEigenvaluesFromEcm(float(e))
        except Exception:
            return None

    def abs_min(e):
        eigs = _get_eigs(e)
        if eigs is None: return np.nan
        return float(min(abs(v) for v in eigs))

    def sgn_min(e):
        """Signed eigenvalue with smallest |λ|. Used for brentq."""
        eigs = _get_eigs(e)
        if eigs is None: return np.nan
        return float(min(eigs, key=abs))

    # ── range ─────────────────────────────────────────────────────────────────
    if data_ecm is not None and len(data_ecm) > 0:
        d_sorted  = np.sort(np.asarray(data_ecm, dtype=float).ravel())
        ecm_start = float(d_sorted[0])  - 0.01
        ecm_end   = float(d_sorted[-1]) + 0.01
    else:
        d_sorted  = np.array([])
        ecm_start, ecm_end = 2.001, 2.40

    ni_levels = _generate_ni_levels(mL, mom, ecm_end + 0.2)
    ni_arr    = np.asarray(ni_levels, dtype=float)
    ni_tol    = 1.5 * epsaux

    def _is_ni(e):
        return len(ni_arr) > 0 and bool(np.any(np.abs(ni_arr - e) < ni_tol))

    # ── grid: evaluate BOTH arrays in one pass ────────────────────────────────
    ngrid  = max(int((ecm_end - ecm_start) / epsaux), 300)
    Egrid  = np.linspace(ecm_start, ecm_end, ngrid)

    abs_lam = np.empty(ngrid)
    sgn_lam = np.empty(ngrid)
    for i, e in enumerate(Egrid):
        eigs = _get_eigs(e)
        if eigs is None:
            abs_lam[i] = np.nan
            sgn_lam[i] = np.nan
        else:
            m = min(eigs, key=abs)
            abs_lam[i] = abs(m)
            sgn_lam[i] = float(m)

    finite  = np.isfinite(abs_lam)
    sentinel = np.nanmax(abs_lam[finite]) if finite.any() else 1.0
    abs_safe = np.where(finite, abs_lam, sentinel)  # no NaN for find_peaks

    # ── candidate detection on abs_min (positive signal) ─────────────────────
    # 1) minima of |λ_min|
    minima_idx, _ = find_peaks(-abs_safe, prominence=np.nanstd(abs_safe[finite]) * 0.1)
    # 2) peaks of -log|λ_min| — amplifies near-zeros even without sign change
    logscore      = -np.log(abs_safe + 1e-16)
    peak_idx,  _  = find_peaks(logscore, prominence=2.0)

    cand_idx = np.unique(np.concatenate([minima_idx, peak_idx]))

    if debug:
        _progress_log(
            f"\n{'='*60}\n"
            f"  irrep={irrep}  P²={psq}\n"
            f"  ni tol={ni_tol:.4f}\n"
            f"  data : {[f'{v:.5f}' for v in d_sorted]}\n"
            f"  sweep [{ecm_start:.5f}, {ecm_end:.5f}]  ngrid={ngrid}\n"
            f"  NI   : {[f'{v:.5f}' for v in ni_levels]}\n"
            f"  candidates from find_peaks: {len(cand_idx)}\n"
            f"{'='*60}"
        )

    # ── also sweep sgn_lam directly for sign changes (catches anything
    #    find_peaks might miss if the dip is very narrow on the grid) ──────────
    sign_change_intervals = []
    for k in range(ngrid - 1):
        if finite[k] and finite[k+1] and sgn_lam[k] * sgn_lam[k+1] < 0.0:
            sign_change_intervals.append((k, k + 1))

    if debug:
        _progress_log(f"  sign-change intervals in sgn_min: {len(sign_change_intervals)}")

    # ── refine ────────────────────────────────────────────────────────────────
    abs_lam_tol  = 1e-6          # threshold for genuine near-zero (no sign change)
    # False sign changes from track swaps have large |λ| at the apparent root.
    # Real zeros have |λ| ≈ 0. Reject if above this after brentq:
    fake_cross_tol = 1e-2
    refine_half  = max(3, ngrid // 300)

    solutions = []

    def _try_root(E_left, E_right, kind):
        """Attempt to find a root in [E_left, E_right]. Returns float or None."""
        root = None
        fl, fr = sgn_min(E_left), sgn_min(E_right)

        if np.isfinite(fl) and np.isfinite(fr) and fl * fr < 0.0:
            try:
                root = brentq(sgn_min, E_left, E_right,
                              xtol=1e-10, rtol=1e-10, maxiter=200)
                # KEY: reject fake sign changes from track swaps
                if abs_min(root) > fake_cross_tol:
                    if debug:
                        _progress_log(
                            f"  [fake cross] Ecm={root:.7f}  |λ|={abs_min(root):.3e}  (track swap)"
                        )
                    return None
                if debug:
                    _progress_log(f"  [brentq {kind}] [{E_left:.5f},{E_right:.5f}] → Ecm={root:.7f}")
            except Exception:
                root = None

        if root is None and kind == "peak":
            # no sign change but candidate peak → minimize |λ_min| directly
            try:
                res = minimize_scalar(abs_min,
                                      bounds=(E_left, E_right),
                                      method='bounded')
                if res.fun < abs_lam_tol:
                    root = res.x
                    if debug:
                        _progress_log(
                            f"  [min_scalar ] [{E_left:.5f},{E_right:.5f}] "
                            f"→ Ecm={root:.7f}  |λ|={res.fun:.3e}"
                        )
            except Exception:
                pass

        return root

    def _accept(root):
        if root is None:
            return False
        if _is_ni(root):
            if debug: _progress_log(f"  [NI  skip]   Ecm={root:.7f}")
            return False
        if any(abs(root - s) < 3 * epsaux for s in solutions):
            if debug: _progress_log(f"  [dup skip]   Ecm={root:.7f}")
            return False
        return True

    # --- process sign-change intervals first (highest confidence) -------------
    for k0, k1 in sign_change_intervals:
        if len(solutions) >= nsols: break
        root = _try_root(Egrid[k0], Egrid[k1], "sign_change")
        if _accept(root):
            solutions.append(float(root))
            if debug:
                _progress_log(
                    f"  [✓ sign] Ecm={root:.9f}  |λ|={abs_min(root):.3e}  "
                    f"({len(solutions)}/{nsols})"
                )

    # --- process peak candidates (catches near-zeros with no sign change) -----
    for idx in cand_idx:
        if len(solutions) >= nsols: break
        i0 = max(0, idx - refine_half)
        i1 = min(ngrid - 1, idx + refine_half)
        root = _try_root(Egrid[i0], Egrid[i1], "peak")
        if _accept(root):
            solutions.append(float(root))
            if debug:
                _progress_log(
                    f"  [✓ peak] Ecm={root:.9f}  |λ|={abs_min(root):.3e}  "
                    f"({len(solutions)}/{nsols})"
                )

    solutions = sorted(solutions)

    if debug:
        _progress_log(
            f"\n  [result] {len(solutions)}/{nsols}  "
            f"{[f'{z:.7f}' for z in solutions]}"
        )

    return solutions[:nsols]





def calcFunc_param(J, Lp, Sp, chanp, L, S, chan, Ecm_over_mref, pSqFuncList, MN, MK, *par):
    if isZero(J, Lp, Sp, chanp, L, S, chan):
        return 0.0

    p2 = _extract_p2(Ecm_over_mref, pSqFuncList)
    qn_channels = _normalize_qn_channels()

    if qn_channels:
        coeffs = None
        active_channel = None
        for qn_channel in qn_channels:
            if qn_channel["enabled"] and _match_qn(qn_channel, J, Lp, Sp, chanp, L, S, chan):
                active_channel = qn_channel
                coeffs = np.asarray(par[qn_channel["slice"]], dtype=float)
                break
        if coeffs is None:
            return 0.0

        local_params = active_channel.get("params", active_channel.get("terms", active_channel.get("calc_terms")))
        # For grouped params (nested lists), use the flattened version for validation
        flat_params = active_channel.get("_flat_params")
        if flat_params is not None:
            params_meta = flat_params
        elif isinstance(local_params, list) and local_params:
            params_meta = local_params
        else:
            params_meta = _ACTIVE_QN.get("params", _ACTIVE_QN.get("calc_terms", []))
        k_matrix_mode = active_channel.get("k_matrix", "polynomial")
    else:
        params_meta = _ACTIVE_QN.get("params", _ACTIVE_QN.get("calc_terms", []))
        n_terms = len(params_meta)
        coeffs = np.asarray(par[:n_terms], dtype=float)
        k_matrix_mode = _ACTIVE_QN.get("k_matrix", "polynomial")

    if params_meta and len(coeffs) != len(params_meta):
        raise ValueError(
            f"Parameter count ({len(coeffs)}) does not match params ({len(params_meta)})."
        )

    # Coupled-channel modes handle barrier factors internally
    if isinstance(k_matrix_mode, list) or str(k_matrix_mode).strip().lower().startswith("coupled_"):
        pg = active_channel.get("_param_groups") if active_channel else None
        km = active_channel.get("_k_matrix_modes") if active_channel else None
        lv = active_channel.get("L_values") if active_channel else None
        mode_str = "coupled" if isinstance(k_matrix_mode, list) else k_matrix_mode
        return _evaluate_k_matrix(mode_str, coeffs, p2,
                                  Ecm_over_mref=Ecm_over_mref, MN=MN, MK=MK,
                                  L=L, Lp=Lp, param_groups=pg,
                                  k_matrix_modes=km, L_values=lv)
    if L == Lp and L in (0, 1, 2, 3):
        return _evaluate_k_matrix(k_matrix_mode, coeffs, p2, Ecm_over_mref=Ecm_over_mref, MN=MN, MK=MK) #p2**int(L) * _evaluate_k_matrix(k_matrix_mode, coeffs, p2, Ecm_over_mref=Ecm_over_mref, MN=MN, MK=MK)
       


def _expand_structure(frames, irreps, levels):
    expanded = []
    for i, frame in enumerate(frames):
        for j, irrep in enumerate(irreps[i]):
            n_levels = int(levels[i][j])
            for level_idx in range(n_levels):
                expanded.append((np.asarray(frame, dtype=float), irrep, level_idx))
    return expanded

def _model_datap2(par, frames, irreps, levels, mL, MN, MK, n=400,
                  debug=False, data2cm=None, kept_mom2=None, kept_irreps=None,
                  step_mode='adaptive', n_refine=1, mN_err=None):
    # ── pre-build data-Ecm lookup (avoids O(N) scan per (frame,irrep) per chi2 call) ──
    _ecm_lookup = None
    if data2cm is not None and kept_mom2 is not None and kept_irreps is not None:
        _ecm_lookup = {}
        for _k, (_m2, _irr) in enumerate(zip(kept_mom2, kept_irreps)):
            _key = (tuple(int(round(x)) for x in np.asarray(_m2).ravel()), _irr)
            _ecm_lookup.setdefault(_key, []).append(float(data2cm[_k][0]))

    epred = []
    for i, mom in enumerate(frames):
        irreps_frame = irreps[i]
        levels_frame = levels[i]
        for j, irrep in enumerate(irreps_frame):
            n_levels = int(levels_frame[j])

            # ── collect data Ecm central values for this (mom, irrep) ─────────
            if _ecm_lookup is not None:
                _mom_key = (tuple(int(round(x)) for x in np.asarray(mom).ravel()), irrep)
                data_ecm_this = _ecm_lookup.get(_mom_key, [])
            else:
                data_ecm_this = None

            epred.extend(
                _findspectrum_irrep(
                    0.1 / float(n), n_levels, mom, mL, par, irrep, MN, MK,
                    data_ecm=data_ecm_this,
                    debug=debug,
                    energy_aware=False,
                    step_mode=step_mode,
                    n_refine=n_refine,
                    mN_err=mN_err,
                )
            )

    epred = np.asarray(epred, dtype=float)
    return epred**2 / 4.0 - 1.0



def _chi2(par, data, frames, irreps, levels, mL, cov, MN, MK, n=400,
          data2cm=None, kept_mom2=None, kept_irreps=None,
          step_mode='adaptive', n_refine=1, mN_err=None):
    try:
        model = _model_datap2(par, frames, irreps, levels, mL, MN, MK, n=n,
                              data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
                              step_mode=step_mode, n_refine=n_refine, mN_err=mN_err)
    except Exception:
        return 1e12

    data_arr  = np.asarray(data,  dtype=float)
    model_arr = np.asarray(model, dtype=float)
    # data_Ecm = 2*np.sqrt( data_arr + 1.0)  # convert back to Ecm/mN for mismatch handling
    # model_Ecm = 2*np.sqrt( model_arr + 1.0)  # convert back to Ecm/mN for mismatch handling
    cov_arr   = np.asarray(cov,   dtype=float)

    # ── handle count mismatch gracefully ──────────────────────────────────────
    if model_arr.shape != data_arr.shape:
        missing = len(data_arr) - len(model_arr)
        penalty = 1e6 * abs(missing)           # smooth, proportional to how many are missing
        n_use   = min(len(data_arr), len(model_arr))
        if n_use > 0:
            resid   = data_arr[:n_use] - model_arr[:n_use]
            sub_cov = cov_arr[:n_use, :n_use]
            return float(resid.T @ np.linalg.inv(sub_cov) @ resid) + penalty
        return float(penalty)
    # print("Input data", data_Ecm)
    # print("Roots:", model_Ecm)
    residual = data_arr - model_arr
    cov_inv  = np.linalg.inv(cov_arr)
    return float(residual.T @ cov_inv @ residual)

def minimizechi2(data, frames, irreps, levels, mL, cov, p0, MN, MK,
                 n=400, max_iter=None, n_maxiter=1000, strategy="nelder-mead",
                 n_starts=20, param_bounds=None,
                 data2cm=None, kept_mom2=None, kept_irreps=None,
                 step_mode='adaptive', n_refine=1,
                 xatol=1e-6, fatol=1e-6, mN_err=None):

    from scipy.optimize import differential_evolution

    p0       = np.asarray(p0, dtype=float)
    data_arr = np.asarray(data, dtype=float)
    cov_inv  = np.linalg.inv(np.asarray(cov, dtype=float))
    expected = _expected_param_count(default_len=len(p0))
    if expected != len(p0):
        raise ValueError(
            f"Initial parameter length ({len(p0)}) does not match expected ({expected})."
        )

    _PENALTY = 1e12

    # ── single helper so data2cm is always forwarded ─────────────────────────
    def _model(params):
        return _model_datap2(params, frames, irreps, levels, mL, MN, MK, n=n,
                             data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
                             step_mode=step_mode, n_refine=n_refine, mN_err=mN_err)

    _cov_arr = np.asarray(cov, dtype=float)

    def objective(params):
        """Inline chi2 using the pre-inverted covariance (avoids recomputing inv(cov) every call)."""
        nonlocal eval_count
        eval_count += 1
        try:
            model_arr = np.asarray(
                _model_datap2(params, frames, irreps, levels, mL, MN, MK, n=n,
                              data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
                              step_mode=step_mode, n_refine=n_refine, mN_err=mN_err),
                dtype=float,
            )
            if model_arr.shape != data_arr.shape:
                missing = len(data_arr) - len(model_arr)
                penalty = 1e6 * abs(missing)
                n_use   = min(len(data_arr), len(model_arr))
                if n_use > 0:
                    resid   = data_arr[:n_use] - model_arr[:n_use]
                    sub_inv = np.linalg.inv(_cov_arr[:n_use, :n_use])
                    return float(resid @ sub_inv @ resid) + penalty
                return float(penalty)
            residual = data_arr - model_arr
            return float(residual @ cov_inv @ residual)
        except Exception as e:
            _progress_log(f"  [error] params {np.array2string(np.asarray(params))}: {e}")
            return _PENALTY


    def _run_nelder_mead(start, maxiter=None):
        opts = {"maxiter": maxiter or int(n), "xatol": xatol, "fatol": fatol}
        return minimize(objective, start, method="Nelder-Mead", options=opts)

    eval_count = 0
    iter_count = 0
    truncated  = False

    if strategy == "nelder-mead":
        class _EarlyStop(Exception): pass
        iter_limit = int(max_iter) if max_iter is not None else None

        def callback(current_params):
            nonlocal iter_count, truncated, eval_count
            iter_count += 1
            eval_count += 1
            if iter_limit is not None and iter_count >= iter_limit:
                truncated = True
                raise _EarlyStop()
            if iter_count % 5 == 0:
                params_arr = np.asarray(current_params, dtype=float)
                try:
                    model_now = np.asarray(_model(params_arr), dtype=float)
                    if model_now.shape == data_arr.shape:
                        residual  = data_arr - model_now
                        chi2_now  = float(residual.T @ cov_inv @ residual)
                        _progress_log(f"params {np.array2string(params_arr)}")
                        _progress_log(f"Model prediction {np.array2string(model_now)}")
                        _progress_log(f"Residual {np.array2string(residual)}")
                        _progress_log(f" Chi2******** {chi2_now}")
                    else:
                        _progress_log(
                            f"params {np.array2string(params_arr)}  "
                            f"[model returned {model_now.shape} vs data {data_arr.shape}]"
                        )
                except Exception as e:
                    _progress_log(f"params {np.array2string(params_arr)}  [callback eval failed: {e}]")

        _progress_log(f"Initial guess {np.array2string(p0)}")
        try:
            chi2_init = objective(p0)
            _progress_log(f"Initial chi2: {chi2_init:.6g}")
            try:
                init_pred = _model(p0)
                init_pred = np.asarray(init_pred, dtype=float)
                if init_pred.shape == data_arr.shape:
                    _progress_log(f"Initial residuals: {np.array2string(data_arr - init_pred)}")
                else:
                    _progress_log(
                        f"Initial residuals: [shape mismatch — "
                        f"data={data_arr.shape}, model={init_pred.shape}]"
                    )
            except Exception as e:
                _progress_log(f"Initial residuals: [unavailable: {e}]")
        except Exception as e:
            _progress_log(f"WARNING: Initial chi2 evaluation failed: {e}")
            _progress_log("Continuing with minimization anyway...")

        _progress_log("minimization started [nelder-mead]")
        try:
            result    = minimize(objective, p0, method="Nelder-Mead",
                                 callback=callback,
                                 options={"maxiter": n_maxiter, "xatol": xatol, "fatol": fatol})
            best_x    = result.x
            best_chi2 = float(result.fun)
            converged = bool(result.success)
        except _EarlyStop:
            converged = False
            result    = minimize(objective, p0, method="Nelder-Mead",
                                 options={"maxiter": iter_count, "xatol": 1e-6, "fatol": 1e-6})
            best_x    = result.x
            best_chi2 = float(result.fun)

    elif strategy == "multistart":
        if param_bounds is None:
            param_bounds = [
                (v - max(abs(v) * 5, 10), v + max(abs(v) * 5, 10))
                for v in p0
            ]
        _progress_log(f"multistart: {n_starts} random starts, bounds={param_bounds}")
        _progress_log(f"Initial guess {np.array2string(p0)} chi2={objective(p0):.6g}")

        best_x    = p0.copy()
        best_chi2 = objective(p0)

        rng = np.random.default_rng(seed=42)
        for i in range(n_starts):
            start = np.array([rng.uniform(lo, hi) for (lo, hi) in param_bounds])
            try:
                res = _run_nelder_mead(start)
                _progress_log(
                    f"  start {i+1}/{n_starts}: chi2={res.fun:.6g}  "
                    f"params={np.array2string(res.x)}"
                )
                if res.fun < best_chi2:
                    best_chi2 = float(res.fun)
                    best_x    = res.x
                    _progress_log(f"  ↑ new best!")
            except Exception as e:
                _progress_log(f"  start {i+1} failed: {e}")

        converged = True

    elif strategy == "differential_evolution":
        from scipy.optimize import basinhopping

        if param_bounds is None:
            param_bounds = [
                (v - max(abs(v) * 5, 10), v + max(abs(v) * 5, 10))
                for v in p0
            ]

        _progress_log(f"basinhopping: bounds={param_bounds} n_starts={n_starts}")
        _progress_log(f"Initial chi2: {objective(p0):.6g}")

        def _bh_callback(x, f, accepted):
            _progress_log(
                f"  BH chi2={f:.6g}  accepted={accepted}  "
                f"params={np.array2string(np.asarray(x))}"
            )

        bh_result = basinhopping(
            objective, p0,
            minimizer_kwargs={"method": "Nelder-Mead",
                              "options": {"maxiter": int(n), "xatol": 1e-6, "fatol": 1e-6}},
            niter=n_starts * 10, seed=42, callback=_bh_callback,
        )
        best_x    = bh_result.x
        best_chi2 = float(bh_result.fun)
        converged = True

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Use 'nelder-mead', 'multistart', or 'differential_evolution'."
        )

    status_str = "TRUNCATED" if truncated else ("converged" if converged else "did NOT converge")
    _progress_log(
        f"minimization ended [{status_str}] chi2={best_chi2:.8g} "
        f"params={np.array2string(best_x)}"
    )
    return best_x, best_chi2, converged

def vij(par, dataE2, frames, irreps, levels, mL, cov, MN, MK, n=150, epsilon=1e-5,
        data2cm=None, kept_mom2=None, kept_irreps=None,
        step_mode='adaptive', n_refine=1, mN_err=None):

    par    = np.asarray(par,    dtype=float)
    data   = np.asarray(dataE2, dtype=float)
    n_data = len(data)
    n_par  = len(par)

    def _model(params):
        return np.asarray(
            _model_datap2(params, frames, irreps, levels, mL, MN, MK, n=n,
                          data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
                          step_mode=step_mode, n_refine=n_refine, mN_err=mN_err),
            dtype=float
        )

    cov_inv         = np.linalg.inv(np.asarray(cov, dtype=float))
    model_at_par    = _model(par)
    residual_at_par = data - model_at_par
    chi2_at_par     = float(residual_at_par.T @ cov_inv @ residual_at_par)
    _progress_log(f"vij: chi2 at input params = {chi2_at_par:.8g}")

    dof = n_data - n_par
    if dof > 0:
        red_chi2 = chi2_at_par / float(dof)
    else:
        red_chi2 = np.nan
    _progress_log(f"vij: dof={dof}  reduced_chi2={red_chi2:.8g}")

    jac = np.zeros((n_data, n_par), dtype=float)

    for i in range(n_par):
        _progress_log(f"  vij: derivative for parameter {i+1}/{n_par}...")

        h = epsilon
        deriv = None

        for attempt in range(6):          # try up to 6 step sizes
            plus  = par.copy(); plus[i]  += h
            minus = par.copy(); minus[i] -= h

            try:
                m_plus  = _model(plus)
                m_minus = _model(minus)
            except Exception as e:
                _progress_log(f"    [vij] attempt {attempt+1}: model error at h={h:.2e}: {e}")
                h /= 2.
                continue

            if m_plus.shape[0] == n_data and m_minus.shape[0] == n_data:
                deriv = (m_plus - m_minus) / (2.0 * h)
                break

            # shape mismatch — try one-sided if one side is ok
            _progress_log(
                f"    [vij] attempt {attempt+1}: h={h:.2e}  "
                f"shapes: +{m_plus.shape[0]} -{m_minus.shape[0]} vs {n_data}"
            )

            if m_plus.shape[0] == n_data and m_minus.shape[0] != n_data:
                m_centre = _model(par)
                if m_centre.shape[0] == n_data:
                    _progress_log(f"    [vij] falling back to forward difference")
                    deriv = (m_plus - m_centre) / h
                    break

            if m_minus.shape[0] == n_data and m_plus.shape[0] != n_data:
                m_centre = _model(par)
                if m_centre.shape[0] == n_data:
                    _progress_log(f"    [vij] falling back to backward difference")
                    deriv = (m_centre - m_minus) / h
                    break

            h /= 2.

        if deriv is None:
            _progress_log(
                f"    [vij] WARNING: could not compute derivative for param {i+1} "
                f"— setting column to zero (parameter may be unconstrained)"
            )
            deriv = np.zeros(n_data)

        jac[:, i] = deriv

    fisher = jac.T @ cov_inv @ jac
    cov_par = np.linalg.pinv(fisher)

    # If reduced chi2 > 1, inflate parameter covariance by reduced chi2.
    # This accounts for under-estimated data variance / model mismatch.
    # if np.isfinite(red_chi2) and red_chi2 > 1.0:
    #     cov_par = cov_par * red_chi2
    #     _progress_log(
    #         f"vij: scaling covariance by reduced_chi2={red_chi2:.6g}"
    #     )

    return cov_par


def build_energy_cm_dict(data2cm, mom2, irreps, levels):
    energy_dict = {}
    for energies, mom, irrep, level_idx in zip(data2cm, mom2, irreps, levels):
        psq = int(np.sum(np.asarray(mom) ** 2))
        psq_label = f"PSQ{psq}"
        energy_dict.setdefault(psq_label, {}).setdefault(irrep, {})[int(level_idx)] = list(energies)
    return energy_dict

def plot_bmatrix_preview(
    data2cm,
    kept_mom2,
    kept_irreps,
    kept_levels,
    mL,
    MN,
    MK,
    ecm_min=None,      # ← None = auto from data
    ecm_max=None,      # ← None = auto from data
    n_sweep=500,
    clip=30,
    save_path=None,
    show= False,
    figsize=(8, 6.134),
    y_scale_mode='data_central',  # 'data_central' (QC-style, default) or 'cloud' (bootstrap cloud)
):
    """
    Plot ONLY the Lüscher lines (B matrix) and data points — no fit needed.

    The B matrix is purely kinematic so no parameters are required.
    Use this BEFORE fitting to:
      - See where data points sit on the quantization condition curves
      - Read off approximate K⁻¹ values at the data energies
      - Choose physically motivated initial parameter guesses

    Parameters
    ----------
    data2cm      : array (n_levels, n_bootstrap), CM energies / m_ref
    kept_mom2    : list of momentum vectors
    kept_irreps  : list of irrep label strings
    mL           : float, m_N * L
    MN, MK       : float, mass ratios
    ecm_min/max  : float or None (auto from data)
    n_sweep      : int, number of sweep points
    clip         : float, mask |B| above this (hides poles)
    save_path    : str or None
    show         : bool
    figsize      : tuple
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm_module
    import os

    _require_bmat()

    # ── auto ecm range from data ─────────────────────────────────────────────
    ecm_means = np.array([float(e[0]) for e in data2cm])
    # print("ecm_means:", ecm_means)
    padding   = 0.05 * (ecm_means.max() - ecm_means.min() or 0.1)
    threshold = 2.0 + 1e-3   # just above two-particle threshold

    if ecm_min is None:
        ecm_min =  ecm_means.min() - padding#max(threshold, ecm_means.min() - padding)
    if ecm_max is None:
        ecm_max = ecm_means.max() + padding


    # ── quantum number setup ─────────────────────────────────────────────────
    qn_channels = _normalize_qn_channels()
    max_L_bmat = 0
    first_enabled = None
    for ch in qn_channels:
        if ch["enabled"]:
            if first_enabled is None:
                first_enabled = ch
            lv = ch.get("L_values")
            if lv is not None:
                for v in lv:
                    max_L_bmat = max(max_L_bmat, int(v))
            else:
                max_L_bmat = max(max_L_bmat, int(ch.get("L", 0)))
    max_L_bmat += 1

    # ── unique (frame, irrep) pairs and assign level indices ────────────────
    seen = {}
    level_indices = {}
    for idx, (mom, irrep) in enumerate(zip(kept_mom2, kept_irreps)):
        key = (tuple(np.asarray(mom).tolist()), irrep)
        if key not in seen:
            seen[key] = (np.asarray(mom, dtype=int), irrep)
            level_indices[key] = []
        level_indices[key].append(idx)
    frame_irrep_list = list(seen.values())
    # Color map for levels (same as plot_energies_vs_irreps)
    import matplotlib.cm as cm
    level_colors_list = list(cm.tab20.colors)
    ecm_color_map = {i: level_colors_list[i % len(level_colors_list)] for i in range(20)}

    # Assign a unique marker shape to each irrep(psq) combo, but keep level colors consistent
    import matplotlib.cm as cm
    marker_shapes = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">", "h", "8", "p", "H", "+", "x", "1", "2", "3", "4"]
    # Use a set of visually distinct colors for Lüscher lines
    luscher_colors = [
        "grey","red", "green", "blue", "orange", "purple", "brown", "magenta", "cyan", "gold", "navy", "lime", "crimson", "teal", "darkorange", "orchid", "slateblue", "olive", "deepskyblue", "maroon", "darkgreen"
    ]
    irrep_marker_map = {}
    irrep_luscher_color_map = {}
    for idx, (mom, irrep) in enumerate(frame_irrep_list):
        key = (tuple(mom), irrep)
        irrep_marker_map[key] = marker_shapes[idx % len(marker_shapes)]
        irrep_luscher_color_map[key] = luscher_colors[idx % len(luscher_colors)]

    # ── sweep arrays ─────────────────────────────────────────────────────────
    ecm_sweep = np.linspace(ecm_min-0.5, ecm_max+0.5, n_sweep)
    p2_sweep  = ecm_sweep ** 2 / 4.0 - 1.0
    # print(p2_sweep)

    # ── dummy K=0 BoxQuantization factory ───────────────────────────────────
    # B matrix is kinematic — K choice doesn't affect it
    _bq_cache = {}

    def _make_box_q(psq, irrep):
        key = (psq, irrep)
        if key in _bq_cache:
            return _bq_cache[key]
        frame_label = _frame_label_from_psq(psq)

        def _dummy_calc(J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF):
            return 0.0  # dummy K⁻¹ = 0, only B matrix matters here

        kinv     = BMat.KMatrix(_dummy_calc, isZero)
        chan_list = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
        box_q    = BMat.BoxQuantization(
            frame_label, psq, irrep, chan_list, [max_L_bmat], kinv, True
        )
        box_q.setRefMassL(mL)
        box_q.setMassesOverRef(0, MN, MK)
        _bq_cache[key] = box_q
        return box_q

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    palette = cm_module.tab10(np.linspace(0, 1, max(len(frame_irrep_list), 1)))

    y_vals_lines = []
    y_vals_data = []
    y_for_limits = []
    # ── Lüscher lines ────────────────────────────────────────────────────────
    for idx, (mom, irrep) in enumerate(frame_irrep_list):
        psq   = int(np.sum(mom ** 2))
        box_q = _make_box_q(psq, irrep)
        key   = (tuple(mom), irrep)
        luscher_color = irrep_luscher_color_map[key]
        marker        = irrep_marker_map[key]

        # Probe matrix size at a safe midpoint energy
        ecm_mid = ecm_sweep[len(ecm_sweep) // 2]
        n_waves = _get_bmat_size(box_q, ecm_mid)


        # One curve per diagonal element (partial wave)
        linestyles = ["-", "--", ":", "-."]
        for iw in range(n_waves):
            b_vals = []
            for ecm in ecm_sweep:
                try:
                    val = float(np.real(box_q.getBoxMatrixFromEcm(float(ecm), iw, iw)))
                except Exception:
                    val = np.nan
                if np.abs(val) > clip:
                    val = np.nan
                b_vals.append(val)
                if not np.isnan(val):
                    y_vals_lines.append(val)

            b_arr  = np.array(b_vals, dtype=float)
            b_arr[np.abs(b_arr) > clip] = np.nan
            jumps  = np.where(np.abs(np.diff(b_arr)) > clip * 0.5)[0]
            for j in jumps:
                b_arr[j] = np.nan

            wave_label = rf"{irrep}({psq}) $B_{{{iw}{iw}}}$"
            ax.plot(
                p2_sweep, b_arr,
                color=luscher_color,
                lw=1.8,
                ls= '--',#linestyles[iw % len(linestyles)],  # different dash per wave
                alpha = 0.5,
                label=wave_label,
                marker=marker, markevery=[-1], markersize=8
            )
    # ── data points ──────────────────────────────────────────────────────────
    for idx, (ecm_samples, mom, irrep) in enumerate(zip(data2cm, kept_mom2, kept_irreps)):
        ecm_samples = np.asarray(ecm_samples)
        ecm_central = float(ecm_samples[0])        # index 0 = central value
        ecm_boots   = ecm_samples[1:]              # [1:] = bootstrap samples

        p2_central  = ecm_central ** 2 / 4.0 - 1.0

        psq       = int(np.sum(np.asarray(mom) ** 2))
        box_q     = _make_box_q(psq, irrep)
        key       = (tuple(mom), irrep)
        marker    = irrep_marker_map[key]
        level_idx = int(kept_levels[idx])
        color     = ecm_color_map.get(level_idx, "black")

        n_waves = _get_bmat_size(box_q, ecm_central)
        for iw in range(n_waves):

            # ── central B value ───────────────────────────────────────────
            try:
                b_central = float(np.real(box_q.getBoxMatrixFromEcm(ecm_central, iw, iw)))
            except Exception:
                continue
            if abs(b_central) > clip:
                continue
            print(f"Plotting data point for mom={mom}, irrep={irrep},(p2={p2_central:.3f}), B={b_central:.3f}")
            # ── bootstrap cloud: compute B for every bootstrap sample ─────
            b_boots = []
            p2_boots = []
            for ecm_s in ecm_boots:
                try:
                    val = float(np.real(box_q.getBoxMatrixFromEcm(float(ecm_s), iw, iw)))
                    if abs(val) <= clip:
                        b_boots.append(val)
                        p2_boots.append(float(ecm_s) ** 2 / 4.0 - 1.0)
                    else:
                        b_boots.append(np.nan)
                        p2_boots.append(np.nan)
                except Exception:
                    b_boots.append(np.nan)
                    p2_boots.append(np.nan)

            b_boots  = np.array(b_boots,  dtype=float)
            p2_boots = np.array(p2_boots, dtype=float)

            # ── keep middle 68 %: sort by ecm, slice [16th, 84th] ────────
            finite_mask = np.isfinite(b_boots) & np.isfinite(p2_boots)
            if finite_mask.sum() > 0:
                sort_idx      = np.argsort(ecm_boots[finite_mask])
                b_sorted      = b_boots[finite_mask][sort_idx]
                p2_sorted     = p2_boots[finite_mask][sort_idx]
                N             = len(sort_idx)
                lo, hi        = int(0.16 * N), int(0.84 * N)
                b_cloud       = b_sorted[lo:hi]
                p2_cloud      = p2_sorted[lo:hi]

                # light scatter cloud (same color, same marker, low alpha)
                ax.plot(
                    p2_cloud, b_cloud,
                    linestyle="None",
                    marker=marker,
                    markersize=8,
                    color=color,
                    alpha=0.50,
                    zorder=3,
                )
                # include cloud in y-limits (mirrors QC fit plot's model-curve percentile)
                cloud_arr = b_cloud[np.isfinite(b_cloud)]
                if cloud_arr.size > 2:
                    y_for_limits.append(float(np.percentile(cloud_arr, 2.0)))
                    y_for_limits.append(float(np.percentile(cloud_arr, 98.0)))
                else:
                    y_for_limits.extend(cloud_arr.tolist())

            # ── central value: solid marker with black edge ───────────────
            ax.plot(
                p2_central, b_central,
                linestyle="None",
                marker=marker,
                markersize=10,
                color=color,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.9,
                zorder=10,
            )
            y_vals_data.append(b_central)
            y_for_limits.append(b_central)

            # ── annotate level index on first wave only ───────────────────
            if iw == 0:
                ax.annotate(
                    f"{level_idx}",
                    xy=(p2_central, b_central),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color="black",
                )

    # ── aesthetics ───────────────────────────────────────────────────────────

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    # Automatically set x-axis (p2) limits to cover all data points with padding
    all_p2 = []
    for idx, (ecm_samples, mom, irrep) in enumerate(zip(data2cm, kept_mom2, kept_irreps)):
        ecm_mean = float(np.mean(ecm_samples))
        p2_mean = ecm_mean ** 2 / 4.0 - 1.0
        all_p2.append(p2_mean)
    if all_p2:
        p2_min, p2_max = min(all_p2), max(all_p2)
        p2_pad = 0.1 * (p2_max - p2_min if p2_max > p2_min else 1.0)
        ax.set_xlim(p2_min - p2_pad, p2_max + p2_pad)
    else:
        ax.set_xlim(p2_sweep[0], p2_sweep[-1])
    # ── select y-limits source ────────────────────────────────────────────────
    # 'data_central': scale to central B values at data Ecm only (same logic as
    #                 the QC fit plot — unaffected by bootstrap cloud spread).
    # 'cloud':        also fold in the 68 % bootstrap cloud percentiles.
    if y_scale_mode == 'data_central':
        y_for_limits = list(y_vals_data)
    # else: y_for_limits already contains cloud percentiles from the loop above

    # Dynamically set y-limits from data central values + bootstrap cloud
    if y_for_limits:
        y_arr = np.array(y_for_limits, dtype=float)
        y_arr = y_arr[np.isfinite(y_arr)]
        y_lo = y_arr.min()
        y_hi = y_arr.max()
        y_span = max(y_hi - y_lo, 0.05)
        ypad = 0.12 * y_span
        y_lo_lim = y_lo - ypad
        y_hi_lim = y_hi + ypad
        y_floor = 0.08 * y_span
        y_lo_lim = min(y_lo_lim, -y_floor)
        y_hi_lim = max(y_hi_lim, y_floor)
        ax.set_ylim(y_lo_lim, y_hi_lim)
    else:
        ax.set_ylim(-clip, clip)
    ax.set_xlabel(r"$p^2\,/\,m_N^2$")
    
    wave_label = _wave_label_from_config()
    wave_label_sub = str(wave_label).replace("$", "")
    # Extract orbital angular momentum letter from labels like "1S0", "1P1", "3D2"
    orbital_letter = None
    for char in wave_label:
        if char.isalpha():
            orbital_letter = char.lower()
            break
    
    # Map wave letter to L quantum number and compute exponent
    wave_to_L = {"s": 0, "p": 1, "d": 2, "f": 3}
    L_val = wave_to_L.get(orbital_letter, 0)
    exponent = 2 * L_val + 1
    ax.set_ylabel(rf"$p^{{{exponent}}}\cot\delta_{{{wave_label_sub}}} / m_N^{{{exponent}}}$", fontsize=14)
    
    # ax.set_title(f"Luscher - {wave_label}")
    # Only show the irrep(psq) legend, with uniform grey color (data points use level colors)
    from matplotlib.lines import Line2D
    legend_elements = []
    for idx, (mom, irrep) in enumerate(frame_irrep_list):
        key = (tuple(mom), irrep)
        marker = irrep_marker_map[key]
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='dimgrey', label=rf"{_format_irrep_label(irrep)}({int(np.sum(mom**2))})",
                   markerfacecolor='none', markeredgecolor='dimgrey', markeredgewidth=1.2,
                   markersize=8, linewidth=1.8)
        )
    ax.legend(handles=legend_elements, fontsize=11, loc="upper left", ncol=1, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax

def plot_quantization_condition(
    par,
    vij,
    data2cm,
    kept_mom2,
    kept_irreps,
    kept_levels,
    mL,
    MN,
    MK,
    ecm_min=None,
    ecm_max=None,
    n_sweep=800,
    clip=10.0,
    show_errorbars=False,
    show_model_curve=True,
    param_labels=None,
    chi2_over_dof=None,
    save_path=None,
    show=False,
    figsize=(10, 7),
):
    """
    Plot the Lüscher quantization condition.
    Conventions:
      data2cm[i][0]  = central value
      data2cm[i][1:] = bootstrap samples
      par            = best-fit parameters
      vij            = covariance matrix of par, shape (n_par, n_par)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm_module
    import matplotlib.lines as mlines
    import os

    _require_bmat()
    par = np.asarray(par, dtype=float)

    # ── auto ecm range ───────────────────────────────────────────────────────
    ecm_means = np.array([float(e[0]) for e in data2cm])
    padding   = 0.05 * (ecm_means.max() - ecm_means.min() or 0.5)
    if ecm_min is None:
        ecm_min = ecm_means.min() - padding
    if ecm_max is None:
        ecm_max = ecm_means.max() + padding

    # ── quantum number setup ─────────────────────────────────────────────────
    qn_channels   = _normalize_qn_channels()
    max_L_bmat    = 0
    first_enabled = None
    for ch in qn_channels:
        if ch["enabled"]:
            if first_enabled is None:
                first_enabled = ch
            lv = ch.get("L_values")
            if lv is not None:
                for v in lv:
                    max_L_bmat = max(max_L_bmat, int(v))
            else:
                max_L_bmat = max(max_L_bmat, int(ch.get("L", 0)))
    #max_L_bmat += 1

    # ── unique (frame, irrep) pairs ──────────────────────────────────────────
    seen          = {}
    level_indices = {}
    for idx, (mom, irrep) in enumerate(zip(kept_mom2, kept_irreps)):
        key = (tuple(np.asarray(mom).tolist()), irrep)
        if key not in seen:
            seen[key] = (np.asarray(mom, dtype=int), irrep)
            level_indices[key] = []
        level_indices[key].append(idx)
    frame_irrep_list = list(seen.values())

    import matplotlib.cm as cm

    # ── marker per irrep(psq), no fill ──────────────────────────────────────
    marker_shapes = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">",
                     "h", "8", "p", "H"]
    irrep_marker_map = {}
    for idx, (mom, irrep) in enumerate(frame_irrep_list):
        key = (tuple(mom), irrep)
        irrep_marker_map[key] = marker_shapes[idx % len(marker_shapes)]

    # level colors from tab20
    level_colors_list = list(cm.tab20.colors)
    ecm_color_map     = {i: level_colors_list[i % len(level_colors_list)] for i in range(40)}

    # ── sweep arrays ─────────────────────────────────────────────────────────
    ecm_sweep = np.linspace(ecm_min - 0.3, ecm_max + 0.3, n_sweep)
    p2_sweep  = ecm_sweep ** 2 / 4.0 - 1.0

    # ── model K⁻¹ curve + gvar band ──────────────────────────────────────────
    model_kinv = None
    kinv_lo    = None
    kinv_hi    = None

    if show_model_curve and first_enabled is not None:
        J     = int(first_enabled.get("twoJ",  0))
        L     = int(first_enabled.get("L",     0))
        S     = int(first_enabled.get("twoS",  0))
        Lp    = int(first_enabled.get("Lp",    L))
        Sp    = int(first_enabled.get("twoSp", S))
        chan  = int(first_enabled.get("chan",   0))
        chanp = int(first_enabled.get("chanp",  0))

        def _eval_kinv(p, ecm_arr):
            out = []
            for e in ecm_arr:
                try:
                    out.append(float(np.real(
                        calcFunc_param(J, Lp, Sp, chanp, L, S, chan,
                                       float(e), None, MN, MK, *p)
                    )))
                except Exception:
                    out.append(np.nan)
            return np.array(out, dtype=float)

        # ── central curve ─────────────────────────────────────────────────────
        model_kinv = _eval_kinv(par, ecm_sweep)


        # ── sample from N(par, vij) → one curve per draw ──────────────────────
        kinv_lo = kinv_hi = None
        if vij is not None:
            vij_arr = np.asarray(vij, dtype=float)
            try:
                par_draws = np.random.multivariate_normal(par, vij_arr, size=500)
            except np.linalg.LinAlgError:
                par_draws = np.random.multivariate_normal(
                    par, np.diag(np.diag(vij_arr)), size=300)

            kinv_samples = np.array([_eval_kinv(ps, ecm_sweep) for ps in par_draws])  # (N, n_sweep)

            # Use 16th–84th percentile of the draw ensemble (≈ ±1σ band).
            # This correctly reflects parameter uncertainty propagated through the model,
            # including how it varies across the x-axis.
            kinv_lo = np.nanpercentile(kinv_samples, 16, axis=0)
            kinv_hi = np.nanpercentile(kinv_samples, 84, axis=0)

        # ── mask poles ────────────────────────────────────────────────────────
        pole_mask = np.abs(model_kinv) > clip
        model_kinv[pole_mask] = np.nan
        if kinv_lo is not None:
            kinv_lo[pole_mask] = np.nan
            kinv_hi[pole_mask] = np.nan

        diffs = np.abs(np.diff(np.where(np.isfinite(model_kinv), model_kinv, 0.0)))
        for jj in np.where(diffs > clip * 0.3)[0]:
            model_kinv[jj] = model_kinv[jj + 1] = np.nan
            if kinv_lo is not None:
                kinv_lo[jj] = kinv_lo[jj + 1] = np.nan
                kinv_hi[jj] = kinv_hi[jj + 1] = np.nan


    # ── BoxQuantization factory ───────────────────────────────────────────────
    _bq_cache = {}

    def _make_box_q(psq, irrep):
        key = (psq, irrep)
        if key in _bq_cache:
            return _bq_cache[key]
        frame_label = _frame_label_from_psq(psq)

        def _calc(J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF):
            return calcFunc_param(J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF, MN, MK, *par)

        kinv_bq  = BMat.KMatrix(_calc, isZero)
        chan_list = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
        box_q    = BMat.BoxQuantization(
            frame_label, psq, irrep, chan_list, [max_L_bmat], kinv_bq, True
        )
        box_q.setRefMassL(mL)
        box_q.setMassesOverRef(0, MN, MK)
        _bq_cache[key] = box_q
        return box_q

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax     = plt.subplots(figsize=figsize)
    y_vals_data = []

    # ── data points: bootstrap cloud + bold central marker ───────────────────
    for idx, (ecm_samples, mom, irrep) in enumerate(zip(data2cm, kept_mom2, kept_irreps)):
        ecm_samples = np.asarray(ecm_samples, dtype=float)
        ecm_central = float(ecm_samples[0])
        ecm_boots   = ecm_samples[1:]

        p2_central = ecm_central ** 2 / 4.0 - 1.0
        psq        = int(np.sum(np.asarray(mom) ** 2))
        box_q      = _make_box_q(psq, irrep)
        key        = (tuple(mom), irrep)
        marker     = irrep_marker_map[key]
        level_idx  = int(kept_levels[idx])
        color      = ecm_color_map.get(level_idx, "black")

        n_waves = _get_bmat_size(box_q, ecm_central)
        for iw in range(n_waves):

            try:
                b_central = float(np.real(box_q.getBoxMatrixFromEcm(ecm_central, iw, iw)))
            except Exception:
                continue
            if abs(b_central) > clip:
                continue

            # bootstrap cloud
            b_boots  = []
            p2_boots = []
            for ecm_s in ecm_boots:
                try:
                    val = float(np.real(box_q.getBoxMatrixFromEcm(float(ecm_s), iw, iw)))
                    if abs(val) <= clip:
                        b_boots.append(val)
                        p2_boots.append(float(ecm_s) ** 2 / 4.0 - 1.0)
                    else:
                        b_boots.append(np.nan)
                        p2_boots.append(np.nan)
                except Exception:
                    b_boots.append(np.nan)
                    p2_boots.append(np.nan)

            b_boots  = np.array(b_boots,  dtype=float)
            p2_boots = np.array(p2_boots, dtype=float)

            finite_mask = np.isfinite(b_boots) & np.isfinite(p2_boots)
            if finite_mask.sum() > 0:
                sort_idx  = np.argsort(ecm_boots[finite_mask])
                b_sorted  = b_boots[finite_mask][sort_idx]
                p2_sorted = p2_boots[finite_mask][sort_idx]
                N         = len(sort_idx)
                lo, hi    = int(0.16 * N), int(0.84 * N)
                b_cloud   = b_sorted[lo:hi]
                p2_cloud  = p2_sorted[lo:hi]

                # hollow markers for cloud
                ax.plot(
                    p2_cloud, b_cloud,
                    linestyle="None", marker=marker,
                    markersize=6, color=color,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=0.7,
                    alpha=0.4, zorder=3,
                )
                # y_vals_data.extend(b_cloud.tolist())

            # bold filled central marker with black edge
            ax.plot(
                p2_central, b_central,
                linestyle="None", marker=marker,
                markersize=8, color=color,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=10,
            )
            y_vals_data.append(b_central)

            if iw == 0:
                ax.annotate(
                    f"{level_idx}",
                    xy=(p2_central, b_central),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="dimgray",
                    fontweight="bold",
                )

    # ── K⁻¹ model curve + 1σ band ────────────────────────────────────────────
    if show_model_curve and model_kinv is not None:
        if kinv_lo is not None and kinv_hi is not None:
            ax.fill_between(
                p2_sweep, kinv_lo, kinv_hi,
                color="#F4B6C2", alpha=0.20, zorder=0
            )
        ax.plot(
            p2_sweep, model_kinv,
            color="#F4B6C2", lw=2.0, zorder=0
        )
    # ── aesthetics ───────────────────────────────────────────────────────────
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)

    all_p2 = [float(e[0]) ** 2 / 4.0 - 1.0 for e in data2cm]
    if all_p2:
        p2_min, p2_max = min(all_p2), max(all_p2)
        p2_pad = 0.12 * (p2_max - p2_min if p2_max > p2_min else 1.0)

        # Always keep some bound-state side (q^2 < 0) visible.
        x_lo = min(p2_min - p2_pad, -0.05)
        x_hi = p2_max + p2_pad
        if x_hi <= 0.0:
            x_hi = 0.05
        ax.set_xlim(x_lo, x_hi)
    else:
        ax.set_xlim(min(p2_sweep[0], -0.05), max(p2_sweep[-1], 0.05))

    wave_label = _wave_label_from_config()
    wave_label_sub = str(wave_label).replace("$", "")
    # Extract orbital angular momentum letter from labels like "1S0", "1P1", "3D2"
    orbital_letter = None
    for char in wave_label:
        if char.isalpha():
            orbital_letter = char.lower()
            break

    # Map wave letter to L quantum number and compute exponent
    wave_to_L = {"s": 0, "p": 1, "d": 2, "f": 3}
    L_val = wave_to_L.get(orbital_letter, 0)
    print("L_val:", L_val)
    exponent = 2 * L_val + 1
    exp_power = 2 * L_val
    print("exponent:", exponent)
    # Bound/virtual-state guides: y = ±(-p^2)^((2L+1)/2) for p^2 < 0
    # Scaled to match y-axis: q^*(2L+1) * cot(delta)
    y_vals_guides = []
    x_lo, x_hi = ax.get_xlim()
    p2_neg_max = min(0.0, x_hi)
    if x_lo < p2_neg_max:
        p2_bound = np.linspace(-0.5, p2_neg_max, 500)
        y_bound = np.sqrt(-p2_bound)#np.power(np.clip(-p2_bound, 1e-10, None))**(1/2)
        ax.plot(
            p2_bound, y_bound,
            color="goldenrod", lw=1.6, ls="--", alpha=0.9,
        )
        ax.plot(
            p2_bound, -y_bound,
            color="goldenrod", lw=1.6, ls="--", alpha=0.9,
        )
        # y_vals_guides.extend(y_bound.tolist())
        # y_vals_guides.extend((-y_bound).tolist())

    # ── fix 3: smarter y-range using percentiles + model curve ───────────────
    # if y_vals_data:
    #     y_arr = np.array(y_vals_data, dtype=float)
    #     y_arr = y_arr[np.isfinite(y_arr)]

    #     # use 10th–90th percentile of data to avoid outlier-driven range
    #     if len(y_arr) > 2:
    #         y_lo_data = np.percentile(y_arr, 10)
    #         y_hi_data = np.percentile(y_arr, 90)
    #     else:
    #         y_lo_data, y_hi_data = y_arr.min(), y_arr.max()

    #     # also include model curve values in visible region
    #     if show_model_curve and model_kinv is not None:
    #         visible = np.isfinite(model_kinv)
    #         if visible.any():
    #             mc = model_kinv[visible]
    #             # clip model extremes so one pole doesn't blow the range
    #             mc_lo = np.percentile(mc, 5)
    #             mc_hi = np.percentile(mc, 95)
    #             y_lo_data = min(y_lo_data, mc_lo)
    #             y_hi_data = max(y_hi_data, mc_hi)
    # Use data + fit curve/band to set y-limits (robust against poles/outliers).
    y_for_limits = list(y_vals_data)

    # Restrict model contributions to the visible x-range and clip extreme tails.
    x_lo_lim, x_hi_lim = ax.get_xlim()
    x_vis = np.asarray(p2_sweep, dtype=float)
    vis_mask = np.isfinite(x_vis) & (x_vis >= x_lo_lim) & (x_vis <= x_hi_lim)

    model_vals = []
    if show_model_curve and model_kinv is not None:
        mk = np.asarray(model_kinv, dtype=float)
        if mk.shape == x_vis.shape:
            model_vals.extend(mk[vis_mask].ravel().tolist())
    if show_model_curve and kinv_lo is not None and kinv_hi is not None:
        klo = np.asarray(kinv_lo, dtype=float)
        khi = np.asarray(kinv_hi, dtype=float)
        if klo.shape == x_vis.shape:
            model_vals.extend(klo[vis_mask].ravel().tolist())
        if khi.shape == x_vis.shape:
            model_vals.extend(khi[vis_mask].ravel().tolist())

    if model_vals:
        m_arr = np.asarray(model_vals, dtype=float)
        m_arr = m_arr[np.isfinite(m_arr)]
        if m_arr.size > 8:
            m_lo = float(np.percentile(m_arr, 2.0))
            m_hi = float(np.percentile(m_arr, 98.0))
            y_for_limits.extend([m_lo, m_hi])
        else:
            y_for_limits.extend(m_arr.tolist())

    if y_for_limits:
        y_arr = np.array(y_for_limits, dtype=float)
        y_arr = y_arr[np.isfinite(y_arr)]

        y_lo = y_arr.min()
        y_hi = y_arr.max()
        y_span = max(y_hi - y_lo, 0.05)
        ypad  = 0.12 * y_span

        y_lo_lim = y_lo - ypad
        y_hi_lim = y_hi + ypad

        # Always show both y > 0 and y < 0 with a small margin.
        y_floor = 0.08 * y_span
        y_lo_lim = min(y_lo_lim, -y_floor)
        y_hi_lim = max(y_hi_lim,  y_floor)

        ax.set_ylim(y_lo_lim, y_hi_lim)
    else:
        ax.set_ylim(-clip, clip)

    # Set evenly spaced y-ticks (same approach as BMat preview functions)
    # y_lo_lim, y_hi_lim = ax.get_ylim()
    # y_ticks = np.linspace(y_lo_lim, y_hi_lim, 6)
    # ax.set_yticks(y_ticks)
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))

    # Halve x-axis tick labels symmetrically across the current x-range.
    # x_ticks_current = ax.get_xticks()
    # if len(x_ticks_current) > 2:
    #     n_x_ticks = max(2, int(np.ceil(len(x_ticks_current) / 2.0)))
    #     keep_idx = np.round(np.linspace(0, len(x_ticks_current) - 1, n_x_ticks)).astype(int)
    #     keep_idx = np.unique(keep_idx)
    #     ax.set_xticks(x_ticks_current[keep_idx])

    # ax.tick_params(axis="both")

    #ax.set_ylim(-0.1,0.1)
    ax.set_xlabel(r"$p^{\star 2}\,/\,m_N^2$")
    if exponent == 1:
        ax.set_ylabel(rf"$p^{{\star}}\cot\delta_{{{wave_label_sub}}} / m_N$")
    else:
        ax.set_ylabel(rf"$p^{{\star {exponent}}}\cot\delta_{{{wave_label_sub}}} / m_N^{{{exponent}}}$")
    
    #ax.set_title(f"Lüscher QC — {wave_label}", fontsize=14)
    #ax.grid(True, alpha=0.2, linestyle=":")

    # ── legend: hollow marker per irrep(psq), no color fill ──────────────────
    legend_elements = []
    for idx, (mom, irrep) in enumerate(frame_irrep_list):
        key    = (tuple(mom), irrep)
        marker = irrep_marker_map[key]
        psq    = int(np.sum(mom ** 2))
        legend_elements.append(
            mlines.Line2D(
                [], [],
                marker=marker,
                linestyle="None",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=1.2,
                label=f"{_format_irrep_label(irrep)}({psq})",
            )
        )
    # if show_model_curve and model_kinv is not None:
    #     legend_elements.append(
    #         mlines.Line2D([], [], color="steelblue", lw=2,
    #                       label=r"$K^{-1}$ fit")
    #     )
    #     legend_elements.append(
    #         mlines.Line2D([], [], color="steelblue", lw=6,
    #                       alpha=0.2, label=r"$K^{-1}\ 1\sigma$")
    #     )

    # Shared anchor so irrep legend and fit-summary box keep aligned left edges.
    irrep_legend_anchor = (0.02, 0.98)
    legend_artist = ax.legend(
        handles=legend_elements,
        fontsize=14,
        loc="upper left",
        bbox_to_anchor=irrep_legend_anchor,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.85,
        edgecolor="lightgray",
        ncol=1,
    )

    def _format_param_label_for_latex(name):
        short = str(name).split(".")[-1]
        if "_" in short:
            head, tail = short.split("_", 1)
            return rf"{head}_{{{tail}}}"
        return short

    def _fmt_compact(value, error):
        """Format value ± error as value(err) in parenthetical notation.

        The integer in parentheses represents the 2-significant-figure
        uncertainty in units of the last displayed decimal digit, e.g.:
            -0.54  ± 0.028  →  -0.540(28)
            -0.54  ± 0.0028 →  -0.5400(28)
             1.234 ± 0.45   →  1.23(45)
        """
        import math
        if not (np.isfinite(value) and np.isfinite(error) and error > 0):
            return f"{value:.3g}"
        exp = math.floor(math.log10(abs(error)))   # order of leading error digit
        n_dec = max(0, -(exp - 1))                 # decimal places → 2 sig figs in error
        err_int = int(round(error * 10 ** n_dec))  # error as integer in last-digit units
        val_str = f"{value:.{n_dec}f}"
        return rf"{val_str}({err_int})"

    # Optional fit-summary textbox (params +/- errors and chi2/dof)
    summary_lines = []
    par_arr = np.asarray(par, dtype=float)

    if param_labels is None:
        param_labels = [f"p{i}" for i in range(len(par_arr))]
    else:
        param_labels = list(param_labels)
        if len(param_labels) != len(par_arr):
            param_labels = [f"p{i}" for i in range(len(par_arr))]

    par_err = None
    if vij is not None:
        try:
            vij_arr = np.asarray(vij, dtype=float)
            if vij_arr.ndim == 2 and vij_arr.shape[0] == vij_arr.shape[1] == len(par_arr):
                diag = np.diag(vij_arr)
                diag = np.where(diag >= 0.0, diag, np.nan)
                par_err = np.sqrt(diag)
        except Exception:
            par_err = None

    for i, (name, value) in enumerate(zip(param_labels, par_arr)):
        short_name = _format_param_label_for_latex(name)
        if par_err is not None and i < len(par_err) and np.isfinite(par_err[i]):
            summary_lines.append(rf"${short_name} = {_fmt_compact(value, par_err[i])}$")
        else:
            summary_lines.append(rf"${short_name} = {value:.3g}$")

    if chi2_over_dof is not None and np.isfinite(chi2_over_dof):
        summary_lines.append("")
        summary_lines.append(rf"$\chi^2/n_{{\mathrm{{dof}}}} = {chi2_over_dof:.4g}$")

    if summary_lines:
        # Place summary directly under legend using rendered legend bounding box.
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            leg_bbox_disp = legend_artist.get_window_extent(renderer=renderer)
            leg_bbox_axes = leg_bbox_disp.transformed(ax.transAxes.inverted())
            box_x = float(irrep_legend_anchor[0])
            box_top = float(leg_bbox_axes.y0) - 0.02
            box_top = min(max(box_top, 0.05), 0.95)
        except Exception:
            box_x = float(irrep_legend_anchor[0])
            box_top = 0.70

        ax.text(
            box_x,
            box_top,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9, "edgecolor": "lightgray"},
            zorder=20,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
def plot_phase_shifts_multichannel(
    par,
    vij,
    data2cm,
    kept_mom2,
    kept_irreps,
    kept_levels,
    mL,
    MN,
    MK,
    ecm_min=None,
    ecm_max=None,
    n_sweep=500,
    save_path=None,
    show=False,
    mN_samples=None,   # bootstrap mN array, mN_samples[0]=central, [1:]=boots
    L_lattice=None,    # integer lattice size (for NI band computation)
    n_par_samples=400,
    figsize=(10, 8),
):
    """
    Two-panel phase-shift summary for multiple independent (non-mixing) channels.

    Panel a (top, larger): phase shift δ in degrees vs E*/mN for each enabled,
        single-channel wave — each channel gets its own colour + ±1σ band from vij.
    Panel b (bottom): input data energies as horizontal error bars with NI energy
        bands (grey shaded ± 1σ from bootstrap mN), coloured by irrep(psq) combo.
    Shared x-axis: E*/mN.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm_module
    import os

    _L_LABELS = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}

    def _spec_label(twoS, L, twoJ):
        L_str = _L_LABELS.get(L, f"L{L}")
        return rf"$^{{{twoS + 1}}}{L_str}_{{{twoJ // 2}}}$"

    par = np.asarray(par, dtype=float)

    # ── ecm range ─────────────────────────────────────────────────────────────
    ecm_means = np.array([float(e[0]) for e in data2cm])
    padding   = 0.05 * (ecm_means.max() - ecm_means.min() or 0.1)
    ecm_lo    = (ecm_means.min() - padding) if ecm_min is None else float(ecm_min)
    ecm_hi    = (ecm_means.max() + padding) if ecm_max is None else float(ecm_max)
    ecm_sweep = np.linspace(ecm_lo, ecm_hi, n_sweep)

    # ── channels ──────────────────────────────────────────────────────────────
    qn_channels = _normalize_qn_channels()

    # ── unique (mom, irrep) combos for colouring data & NI ────────────────────
    seen_combos   = {}
    for mom, irrep in zip(kept_mom2, kept_irreps):
        key = (tuple(np.asarray(mom).tolist()), irrep)
        seen_combos.setdefault(key, (np.asarray(mom, dtype=int), irrep))
    combos        = list(seen_combos.values())
    combo_colors  = [f"C{i}" for i in range(len(combos))]
    combo_key_map = {(tuple(m.tolist()), ir): ci for ci, (m, ir) in enumerate(combos)}
    marker_shapes = ["o", "s", "D", "^", "v", "P", "*", "X"]
    combo_markers = {(tuple(m.tolist()), ir): marker_shapes[ci % len(marker_shapes)]
                     for ci, (m, ir) in enumerate(combos)}

    # ── NI bands ──────────────────────────────────────────────────────────────
    ni_vlines = []   # (ecm_c, ni_lo, ni_hi)
    if mN_samples is not None and L_lattice is not None:
        mN_arr      = np.asarray(mN_samples, dtype=float)
        mN_central  = float(mN_arr[0])
        mN_boots    = mN_arr[1:]
        mL_central  = mN_central * L_lattice
        unique_moms = {tuple(m.tolist()): m for m, _ in combos}
        seen_ni     = []
        for mom_key, mom_arr in unique_moms.items():
            ni_c = _generate_ni_levels(mL_central, mom_arr, ecm_hi + 0.15)
            ni_boots_all = [
                np.asarray(_generate_ni_levels(mn * L_lattice, mom_arr, ecm_hi + 0.15))
                for mn in mN_boots
            ]
            for ecm_c in ni_c:
                if ecm_c < ecm_lo or ecm_c > ecm_hi:
                    continue
                if any(abs(ecm_c - s) < 1e-6 for s in seen_ni):
                    continue
                seen_ni.append(ecm_c)
                boot_vals = []
                for ni_b in ni_boots_all:
                    if len(ni_b):
                        boot_vals.append(float(ni_b[np.argmin(np.abs(ni_b - ecm_c))]))
                if boot_vals:
                    ni_lo = float(np.percentile(boot_vals, 16))
                    ni_hi = float(np.percentile(boot_vals, 84))
                else:
                    ni_lo = ni_hi = ecm_c
                ni_vlines.append((ecm_c, ni_lo, ni_hi))

    def _draw_ni(ax):
        for ecm_c, ni_lo, ni_hi in ni_vlines:
            ax.axvline(ecm_c, color="grey", lw=1.0, ls="--", alpha=0.5, zorder=0)
            ax.axvspan(ni_lo, ni_hi, color="grey", alpha=0.12, zorder=0)

    # ── helper: compute δ array for a parameter set ───────────────────────────
    def _delta_for_channel(p_arr, ch):
        L_ch   = int(ch.get("L",      0))
        J_ch   = int(ch.get("twoJ",   0))
        Lp_ch  = int(ch.get("Lp",     L_ch))
        S_ch   = int(ch.get("twoS",   0))
        Sp_ch  = int(ch.get("twoSp",  S_ch))
        chan_   = int(ch.get("chan",   0))
        chanp_ = int(ch.get("chanp",  0))
        out    = []
        for ecm in ecm_sweep:
            p2 = ecm ** 2 / 4.0 - 1.0
            if p2 <= 0.0:
                out.append(np.nan)
                continue
            p = np.sqrt(p2)
            try:
                kinv    = float(np.real(calcFunc_param(
                    J_ch, Lp_ch, Sp_ch, chanp_, L_ch, S_ch, chan_,
                    float(ecm), None, MN, MK, *p_arr
                )))
                barrier = p ** (2 * L_ch + 1)
                out.append(np.degrees(np.arctan2(barrier, kinv)))
            except Exception:
                out.append(np.nan)
        return np.array(out, dtype=float)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # ── TOP: phase shifts ──────────────────────────────────────────────────────
    ax_top.axhline(0,  color="grey", lw=0.7, ls="--")
    ax_top.axhline(90, color="grey", lw=0.5, ls=":", alpha=0.5)
    _draw_ni(ax_top)

    # pre-draw vij band (low z-order) then central line
    channel_colors = [f"C{i}" for i in range(20)]
    ch_idx = 0
    for ch in qn_channels:
        if not ch.get("enabled", True):
            continue
        k_mode = ch.get("k_matrix", "polynomial")
        # skip coupled / multi-wave channels
        if isinstance(k_mode, list) or str(k_mode).strip().lower().startswith("coupled"):
            continue

        L_ch  = int(ch.get("L", 0))
        twoJ  = int(ch.get("twoJ", 0))
        twoS  = int(ch.get("twoS", 0))
        label = ch.get("name", None) or _spec_label(twoS, L_ch, twoJ)
        color = channel_colors[ch_idx % len(channel_colors)]
        ch_idx += 1

        delta_central = _delta_for_channel(par, ch)

        # ±1σ band from vij
        if vij is not None:
            vij_arr = np.asarray(vij, dtype=float)
            try:
                draws = np.random.multivariate_normal(par, vij_arr, size=n_par_samples)
            except np.linalg.LinAlgError:
                draws = np.random.multivariate_normal(
                    par, np.diag(np.diag(vij_arr)), size=n_par_samples)
            samp  = np.array([_delta_for_channel(d, ch) for d in draws])
            d_lo  = np.nanpercentile(samp, 16, axis=0)
            d_hi  = np.nanpercentile(samp, 84, axis=0)
            ax_top.fill_between(ecm_sweep, d_lo, d_hi,
                                color=color, alpha=0.20, zorder=2)

        # mask large jumps (phase-wrap artifacts)
        delta_plot = delta_central.copy()
        jumps = np.where(np.abs(np.diff(np.where(np.isfinite(delta_plot),
                                                  delta_plot, 0.0))) > 90)[0]
        for j in jumps:
            delta_plot[j] = delta_plot[j + 1] = np.nan

        ax_top.plot(ecm_sweep, delta_plot, color=color, lw=2.0,
                    label=label, zorder=3)

    ax_top.set_ylabel(r"$\delta$ (degrees)")
    ax_top.set_ylim(-5, 185)
    ax_top.set_yticks([0, 45, 90, 135, 180])
    ax_top.legend(loc="best", fontsize="small", frameon=True)
    ax_top.set_title("Phase shifts from fit parametrization")

    # ── BOTTOM: data energies + NI ────────────────────────────────────────────
    _draw_ni(ax_bot)
    ax_bot.set_yticks([])
    ax_bot.set_ylabel("levels", fontsize="small")

    for idx, (ecm_samples, mom, irrep) in enumerate(
            zip(data2cm, kept_mom2, kept_irreps)):
        ecm_arr = np.asarray(ecm_samples, dtype=float)
        ecm_c   = float(ecm_arr[0])
        ecm_err = float(np.std(ecm_arr[1:])) if len(ecm_arr) > 1 else 0.0
        key     = (tuple(np.asarray(mom).tolist()), irrep)
        ci      = combo_key_map.get(key, 0)
        color   = combo_colors[ci]
        mkr     = combo_markers.get(key, "o")
        lv      = int(kept_levels[idx])
        ax_bot.errorbar(
            ecm_c, 0.0, xerr=ecm_err,
            fmt=mkr, color=color, ms=7, lw=1.5,
            capsize=4, capthick=1.2,
            markeredgecolor="black", markeredgewidth=0.6,
            zorder=10,
            label=f"{_format_irrep_label(irrep)}(P²={int(np.sum(np.asarray(mom)**2))})"
                  if idx == combo_key_map.get(key, -1) else None,
        )
        ax_bot.annotate(
            str(lv),
            xy=(ecm_c, 0.0), xytext=(0, 7),
            textcoords="offset points",
            ha="center", fontsize=7, color="black",
        )

    # deduplicated legend for combos
    handles, labels_leg = ax_bot.get_legend_handles_labels()
    seen_leg = {}
    h_uniq, l_uniq = [], []
    for h, lb in zip(handles, labels_leg):
        if lb and lb not in seen_leg:
            seen_leg[lb] = True
            h_uniq.append(h); l_uniq.append(lb)
    if h_uniq:
        ax_bot.legend(h_uniq, l_uniq, loc="upper left",
                      fontsize="x-small", frameon=True, ncol=2)

    ax_bot.set_ylim(-0.5, 0.5)
    ax_bot.set_xlabel(r"$E^*/m_N$")
    ax_bot.set_xlim(ecm_lo, ecm_hi)

    fig.align_ylabels([ax_top, ax_bot])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig, (ax_top, ax_bot)


def plot_fit_comparison(
    fits,
    data2cm,
    kept_mom2,
    kept_irreps,
    kept_levels,
    mL,
    MN,
    MK,
    ecm_min=None,
    ecm_max=None,
    n_sweep=800,
    clip=10.0,
    show_errorbars=True,
    save_path=None,
    show=False,
    figsize=(10, 7),
    cov=None,
    data_central=None,
    frames_all=None,
    irreps_all=None,
    levels_all=None,
):
    """
    Plot multiple K^{-1} fits overlaid on the same QC plot for comparison.
    
    Parameters
    ----------
    fits : list of dict
        Each dict contains:
          - 'params': array-like, parameter values
          - 'label': str, legend label
          - 'color': str, line color (optional, default cycles through palette)
          - 'linestyle': str, line style (optional, default '-')
          - 'linewidth': float, line width (optional, default 2.0)
    data2cm : list of arrays
        Data in CM frame
    kept_mom2, kept_irreps, kept_levels : arrays
        Momentum, irrep, level indices
    mL, MN, MK : float
        Box size and reference masses
    ecm_min, ecm_max : float, optional
        Energy range
    n_sweep : int
        Sweep resolution
    clip : float
        Pole clipping threshold
    show_errorbars : bool
        Show data error bars
    save_path : str
        Save path
    show : bool
        Display plot
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.lines as mlines
    import os
    
    _require_bmat()
    
    # auto ecm range
    ecm_means = np.array([float(e[0]) for e in data2cm])
    padding   = 0.05 * (ecm_means.max() - ecm_means.min() or 0.5)
    if ecm_min is None:
        ecm_min = ecm_means.min() - padding
    if ecm_max is None:
        ecm_max = ecm_means.max() + padding

    # quantum number setup
    qn_channels   = _normalize_qn_channels()
    first_enabled = None
    for ch in qn_channels:
        if ch["enabled"]:
            first_enabled = ch
            break
    
    if first_enabled is None:
        raise ValueError("No enabled channels in quantum numbers config")
    
    J     = int(first_enabled.get("twoJ",  0))
    L     = int(first_enabled.get("L",     0))
    S     = int(first_enabled.get("twoS",  0))
    Lp    = int(first_enabled.get("Lp",    L))
    Sp    = int(first_enabled.get("twoSp", S))
    chan  = int(first_enabled.get("chan",   0))
    chanp = int(first_enabled.get("chanp",  0))

    # unique (frame, irrep) pairs
    seen          = {}
    level_indices = {}
    for idx, (mom, irrep) in enumerate(zip(kept_mom2, kept_irreps)):
        key = (tuple(np.asarray(mom).tolist()), irrep)
        if key not in seen:
            seen[key] = (np.asarray(mom, dtype=int), irrep)
            level_indices[key] = []
        level_indices[key].append(idx)
    frame_irrep_list = list(seen.values())

    # marker per irrep(psq)
    marker_shapes = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">", "h", "8", "p", "H"]
    irrep_marker_map = {}
    for idx, (mom, irrep) in enumerate(frame_irrep_list):
        key = (tuple(mom), irrep)
        irrep_marker_map[key] = marker_shapes[idx % len(marker_shapes)]

    level_colors_list = list(cm.tab20.colors)
    
    # sweep arrays
    ecm_sweep = np.linspace(ecm_min - 0.3, ecm_max + 0.3, n_sweep)
    p2_sweep  = ecm_sweep ** 2 / 4.0 - 1.0

    # figure setup
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    
    # helper to evaluate K^{-1}
    def _eval_kinv(p, ecm_arr):
        out = []
        for e in ecm_arr:
            try:
                out.append(float(np.real(
                    calcFunc_param(J, Lp, Sp, chanp, L, S, chan,
                                   float(e), None, MN, MK, *p)
                )))
            except Exception:
                out.append(np.nan)
        return np.array(out, dtype=float)
    
    # default color palette
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # pre-compute chi2/dof for each fit if covariance data is provided
    fit_chi2_dof = []
    if cov is not None and data_central is not None and frames_all is not None:
        cov_arr = np.asarray(cov, dtype=float)
        data_arr = np.asarray(data_central, dtype=float)
        n_data = len(data_arr)
        for fit_spec in fits:
            params = np.asarray(fit_spec['params'], dtype=float)
            n_par = len(params)
            dof = n_data - n_par
            try:
                chi2_val = _chi2(
                    params, data_arr,
                    frames_all, irreps_all, levels_all,
                    mL, cov_arr, MN, MK,
                    n=400,
                )
                fit_chi2_dof.append(chi2_val / dof if dof > 0 else float('nan'))
            except Exception:
                fit_chi2_dof.append(float('nan'))
    else:
        fit_chi2_dof = [None] * len(fits)

    # plot each fit curve
    fit_lines = []
    fit_colors = []
    for idx, fit_spec in enumerate(fits):
        params = np.asarray(fit_spec['params'], dtype=float)
        label = fit_spec.get('label', f'Fit {idx+1}')
        color = fit_spec.get('color', default_colors[idx % len(default_colors)])
        linestyle = fit_spec.get('linestyle', '-')
        linewidth = fit_spec.get('linewidth', 2.0)

        chi2_dof = fit_chi2_dof[idx]
        if chi2_dof is not None and np.isfinite(chi2_dof):
            label = fr"{label}  ($\chi^2$/dof = {chi2_dof:.2f})"
        
        model_kinv = _eval_kinv(params, ecm_sweep)
        
        # mask poles
        pole_mask = np.abs(model_kinv) > clip
        model_kinv[pole_mask] = np.nan
        
        diffs = np.abs(np.diff(np.where(np.isfinite(model_kinv), model_kinv, 0.0)))
        for jj in np.where(diffs > clip * 0.3)[0]:
            model_kinv[jj] = model_kinv[jj + 1] = np.nan
        
        line, = ax.plot(
            p2_sweep, model_kinv,
            color=color, lw=linewidth, ls=linestyle,
            label=label, zorder=5
        )
        fit_lines.append(line)
        fit_colors.append(color)
    
    # BoxQuantization factory
    def _make_box_q_local(psq, irrep):
        frame_label = _frame_label_from_psq(psq)
        rep = str(irrep) if not isinstance(irrep, int) else ""
        return BMat.BoxQuantization(
            L=mL, irrep=rep, psq=psq,
            Mref1=MN, Mref2=MK,
            frame=frame_label
        )
    
    # plot data points
    y_vals_data = []
    for i_dat, (ecm_samples, mom, irrep, level_idx) in enumerate(
        zip(data2cm, kept_mom2, kept_irreps, kept_levels)
    ):
        ecm_samples = np.asarray(ecm_samples, dtype=float)
        ecm_central = float(ecm_samples[0])
        p2_central  = ecm_central**2 / 4.0 - 1.0
        
        key    = (tuple(np.asarray(mom).tolist()), irrep)
        marker = irrep_marker_map[key]
        color  = level_colors_list[int(level_idx) % len(level_colors_list)]
        
        # Convert to q-cot-delta
        try:
            box_q = _make_box_q_local(int(np.sum(mom**2)), irrep)
            b_central = float(np.real(
                box_q.getmatBsubmatrix_fromlatticeE2(ecm_central**2)[0, 0]
            ))
        except Exception:
            continue
        
        # error bars from bootstrap
        if show_errorbars and len(ecm_samples) > 1:
            b_boots = []
            for ecm_b in ecm_samples[1:]:
                try:
                    b_boots.append(float(np.real(
                        box_q.getmatBsubmatrix_fromlatticeE2(ecm_b**2)[0, 0]
                    )))
                except Exception:
                    pass
            if b_boots:
                b_boots = np.array(b_boots)
                b_lo = float(gv.sdev(gv.dataset.avg_data(b_boots, bstrap=True)))
                ax.errorbar(
                    p2_central, b_central,
                    yerr=b_lo,
                    fmt=marker, color=color,
                    markersize=8, capsize=4, capthick=1.0,
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=10
                )
        else:
            ax.plot(
                p2_central, b_central,
                marker=marker, color=color,
                markersize=8,
                markeredgecolor="black", markeredgewidth=0.8,
                linestyle="None", zorder=10
            )
        
        y_vals_data.append(b_central)
        
        # level annotation
        ax.annotate(
            str(int(level_idx)),
            xy=(p2_central, b_central),
            xytext=(4, 0), textcoords="offset points",
            fontsize=8, color="dimgray", fontweight="bold"
        )
    
    # axes setup
    all_p2 = [float(e[0]) ** 2 / 4.0 - 1.0 for e in data2cm]
    if all_p2:
        p2_min, p2_max = min(all_p2), max(all_p2)
        p2_pad = 0.12 * (p2_max - p2_min if p2_max > p2_min else 1.0)
        x_lo = min(p2_min - p2_pad, -0.05)
        x_hi = p2_max + p2_pad
        if x_hi <= 0.0:
            x_hi = 0.05
        ax.set_xlim(x_lo, x_hi)
    else:
        ax.set_xlim(min(p2_sweep[0], -0.05), max(p2_sweep[-1], 0.05))

    wave_label = _wave_label_from_config()
    wave_label_sub = str(wave_label).replace("$", "")
    
    # Extract L for exponent
    orbital_letter = None
    for char in wave_label_sub:
        if char.isalpha():
            orbital_letter = char.lower()
            break
    wave_to_L = {"s": 0, "p": 1, "d": 2, "f": 3}
    L_val = wave_to_L.get(orbital_letter, 0)
    exponent = 2 * L_val + 1

    # y-limits from data
    if y_vals_data:
        y_arr = np.array(y_vals_data, dtype=float)
        y_arr = y_arr[np.isfinite(y_arr)]
        y_lo = y_arr.min()
        y_hi = y_arr.max()
        y_span = max(y_hi - y_lo, 0.05)
        ypad = 0.15 * y_span
        y_lo_lim = y_lo - ypad
        y_hi_lim = y_hi + ypad
        y_floor = 0.08 * y_span
        y_lo_lim = min(y_lo_lim, -y_floor)
        y_hi_lim = max(y_hi_lim, y_floor)
        ax.set_ylim(y_lo_lim, y_hi_lim)
    else:
        ax.set_ylim(-clip, clip)

    # Set evenly spaced y-ticks (same approach as BMat preview functions)
    y_lo_lim, y_hi_lim = ax.get_ylim()
    y_ticks = np.linspace(y_lo_lim, y_hi_lim, 8)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))

    # x-ticks (halve)
    x_ticks_current = ax.get_xticks()
    if len(x_ticks_current) > 2:
        n_x_ticks = max(2, int(np.ceil(len(x_ticks_current) / 2.0)))
        keep_idx = np.round(np.linspace(0, len(x_ticks_current) - 1, n_x_ticks)).astype(int)
        keep_idx = np.unique(keep_idx)
        ax.set_xticks(x_ticks_current[keep_idx])

    ax.tick_params(axis="both", labelsize=12)

    # labels
    ax.set_xlabel(r"$p^{\star 2}\,/\,m_N^2$")
    if exponent == 1:
        ax.set_ylabel(rf"$p^{{\star}}\cot\delta_{{{wave_label_sub}}} / m_N$")
    else:
        ax.set_ylabel(rf"$p^{{\star {exponent}}}\cot\delta_{{{wave_label_sub}}} / m_N^{{{exponent}}}$")
    
    ax.grid(True, alpha=0.2, linestyle=":")
    
    # legend for fits — color each label text to match its fit band
    leg = ax.legend(handles=fit_lines, loc='best', fontsize=11, frameon=True,
                    framealpha=0.9, edgecolor='lightgray')
    for text, color in zip(leg.get_texts(), fit_colors):
        text.set_color(color)
    
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_chi2_landscape(
            par,
            data,
            frames,
            irreps,
            levels,
            mL,
            cov,
            MN,
            MK,
            n=150,
            param_indices=(0, 1),
            n_grid=40,
            param_bounds=None,
            scale=3.0,
            save_path=None,
            show=True,
            figsize=(10, 8),
            log_scale=False,
        ):
        """
        Plot a 3D chi² landscape over two parameters.

        Fixes all other parameters at their best-fit values and scans
        a 2D grid over the two selected parameters.

        Parameters
        ----------
        par           : array-like, best-fit parameters (center of scan)
        data          : 1D array, data p² values
        frames/irreps/levels/mL/cov/MN/MK : as in minimizechi2
        n             : int, spectrum resolution (same as fit)
        param_indices : tuple of 2 ints, which parameters to scan (default (0,1))
        n_grid        : int, number of grid points per axis (n_grid² evaluations)
        param_bounds  : list of 2 (lo, hi) tuples, scan range per axis.
                        If None, auto-computed as par[i] ± scale*max(|par[i]|, 1)
        scale         : float, auto-range half-width in units of |par[i]|
        save_path     : str or None
        show          : bool
        figsize       : tuple
        log_scale     : bool, plot log10(chi²) instead of chi² (helps if range is huge)
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        par = np.asarray(par, dtype=float)
        i0, i1 = int(param_indices[0]), int(param_indices[1])

        if len(par) < 2:
            raise ValueError("plot_chi2_landscape requires at least 2 parameters.")
        if i0 >= len(par) or i1 >= len(par):
            raise ValueError(f"param_indices {param_indices} out of range for {len(par)} parameters.")
        if i0 == i1:
            raise ValueError("param_indices must be two different parameters.")

        # ── axis ranges ──────────────────────────────────────────────────────────
        def _auto_range(idx):
            center = par[idx]
            half   = max(abs(center) * scale, scale)
            return (center - half, center + half)

        lo0, hi0 = param_bounds[0] if param_bounds else _auto_range(i0)
        lo1, hi1 = param_bounds[1] if param_bounds else _auto_range(i1)

        axis0 = np.linspace(lo0, hi0, n_grid)
        axis1 = np.linspace(lo1, hi1, n_grid)
        G0, G1 = np.meshgrid(axis0, axis1)

        _progress_log(
            f"chi2 landscape: scanning params [{i0}, {i1}] "
            f"over {n_grid}×{n_grid} grid ({n_grid**2} evaluations)..."
        )

        # ── evaluate chi² on grid ────────────────────────────────────────────────
        Z = np.zeros_like(G0)
        for gi in range(n_grid):
            for gj in range(n_grid):
                p_scan = par.copy()
                p_scan[i0] = G0[gi, gj]
                p_scan[i1] = G1[gi, gj]
                try:
                    Z[gi, gj] = _chi2(p_scan, data, frames, irreps, levels, mL, cov, MN, MK, n=n)
                except Exception:
                    Z[gi, gj] = np.nan
            _progress_log(f"  row {gi+1}/{n_grid} done")

        # ── resolve parameter labels for axis labels ─────────────────────────────
        qn_channels = _normalize_qn_channels()
        all_labels  = []
        for ch in qn_channels:
            if not ch["enabled"]:
                continue
            params_meta = ch.get("params", ch.get("terms", ch.get("calc_terms", [])))
            if isinstance(params_meta, list) and params_meta:
                for pidx, pm in enumerate(params_meta):
                    all_labels.append(f"{ch['name']}.{pm.get('name', f'p{pidx}')}")
            else:
                for pidx in range(ch.get("n_params", 0)):
                    all_labels.append(f"{ch['name']}.p{pidx}")
        if len(all_labels) != len(par):
            all_labels = [f"p{i}" for i in range(len(par))]

        xlabel = all_labels[i0]
        ylabel = all_labels[i1]

        # ── plot ─────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection="3d")

        Zplot = np.log10(np.clip(Z, 1e-10, None)) if log_scale else Z
        zlabel = r"$\log_{10}(\chi^2)$" if log_scale else r"$\chi^2$"

        surf = ax.plot_surface(
            G0, G1, Zplot,
            cmap=cm.viridis,
            alpha=0.85,
            linewidth=0,
            antialiased=True,
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)

        # mark best-fit point
        chi2_best = _chi2(par, data, frames, irreps, levels, mL, cov, MN, MK, n=n)
        z_best    = np.log10(max(chi2_best, 1e-10)) if log_scale else chi2_best
        ax.scatter(
            [par[i0]], [par[i1]], [z_best],
            color="red", s=80, zorder=10,
            label=f"Best fit\n{xlabel}={par[i0]:.4g}\n{ylabel}={par[i1]:.4g}\nχ²={chi2_best:.4g}",
        )

        ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
        ax.set_zlabel(zlabel, fontsize=11, labelpad=10)
        ax.set_title(r"$\chi^2$ landscape", fontsize=13)
        ax.legend(fontsize=9, loc="upper left")
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            _progress_log(f"chi2 landscape saved: {save_path}")
        if show:
            plt.show()
        plt.close(fig)
        return fig, ax

def estimate_initial_params_from_bmatrix(
        data2cm,
        kept_mom2,
        kept_irreps,
        mL,
        MN,
        MK,
    ):
    """
    Estimate initial parameter guesses by reading Re[B_{iw,iw}(E_data)] at each
    data energy level for each enabled wave (channel), then fitting those B values
    with that channel's K-matrix formula.

    Each channel gets its own diagonal B matrix element:
      channel 0 (e.g. 3S1) → B(ecm, 0, 0)
      channel 1 (e.g. 3D1) → B(ecm, 1, 1)
      etc.

    Returns
    -------
    b_pairs_per_channel : dict  {channel_name: [(p2, B), ...]}
    p0_estimate         : flat np.ndarray matching the full parameter layout
    """
    from scipy.optimize import minimize as _minimize
    _require_bmat()

    qn_channels = _normalize_qn_channels()
    enabled_channels = [ch for ch in qn_channels if ch["enabled"]]

    if not enabled_channels:
        _progress_log("Warning: no enabled channels found — returning zeros")
        return {}, np.zeros(_expected_param_count(default_len=1))

    # ── max_L across all enabled channels ────────────────────────────────────
    max_L_bmat = 0
    for ch in enabled_channels:
        lv = ch.get("L_values")
        if lv is not None:
            for v in lv:
                max_L_bmat = max(max_L_bmat, int(v))
        else:
            max_L_bmat = max(max_L_bmat, int(ch.get("L", 0)))
    max_L_bmat += 1

    # ── BoxQuantization factory (dummy K=0, only B needed) ───────────────────
    _bq_cache = {}

    def _make_box_q(psq, irrep):
        key = (psq, irrep)
        if key in _bq_cache:
            return _bq_cache[key]
        frame_label = _frame_label_from_psq(psq)

        def _calc_zero(J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF):
            return 0.0

        kinv     = BMat.KMatrix(_calc_zero, isZero)
        chan_list = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
        box_q    = BMat.BoxQuantization(
            frame_label, psq, irrep, chan_list, [max_L_bmat], kinv, True
        )
        box_q.setRefMassL(mL)
        box_q.setMassesOverRef(0, MN, MK)
        _bq_cache[key] = box_q
        return box_q

    # ── collect B values per channel (wave index = diagonal index) ───────────
    # Each enabled channel iw reads B(ecm, iw, iw)
    per_channel_p2  = {ch["name"]: [] for ch in enabled_channels}
    per_channel_b   = {ch["name"]: [] for ch in enabled_channels}
    per_channel_ecm = {ch["name"]: [] for ch in enabled_channels}

    for ecm_samples, mom, irrep in zip(data2cm, kept_mom2, kept_irreps):
        ecm_mean = float(np.mean(ecm_samples))
        p2_mean  = ecm_mean ** 2 / 4.0 - 1.0
        if p2_mean <= 0:
            continue
        psq   = int(np.sum(np.asarray(mom) ** 2))

        try:
            box_q = _make_box_q(psq, irrep)
        except Exception as e:
            _progress_log(f"  B estimate: could not build BoxQ ({irrep}, PSq={psq}): {e}")
            continue

        for iw, ch in enumerate(enabled_channels):
            ch_name = ch["name"]
            try:
                b_val = float(np.real(box_q.getBoxMatrixFromEcm(float(ecm_mean), iw, iw)))
                per_channel_p2[ch_name].append(p2_mean)
                per_channel_b[ch_name].append(b_val)
                per_channel_ecm[ch_name].append(ecm_mean)
                _progress_log(
                    f"  B estimate [{ch_name}] iw={iw}: irrep={irrep} PSq={psq} "
                    f"Ecm={ecm_mean:.4f} p2={p2_mean:.4f} B[{iw},{iw}]={b_val:.4f}"
                )
            except Exception as e:
                _progress_log(f"  B estimate [{ch_name}] iw={iw} skipped ({irrep}, PSq={psq}): {e}")

    # ── fit K-matrix parameters per channel independently ────────────────────
    b_pairs_per_channel = {}
    p0_estimate_full    = np.zeros(_expected_param_count(default_len=1))

    for iw, ch in enumerate(enabled_channels):
        ch_name  = ch["name"]
        k_mode   = str(ch.get("k_matrix", "polynomial")).strip().lower()
        n_params = ch.get("n_params", 1)
        sl       = ch.get("slice", slice(0, 0))
        L        = int(ch.get("L", 0))

        p2_arr  = np.array(per_channel_p2[ch_name])
        b_arr   = np.array(per_channel_b[ch_name])
        ecm_arr = np.array(per_channel_ecm[ch_name])

        b_pairs_per_channel[ch_name] = list(zip(p2_arr.tolist(), b_arr.tolist()))
        _progress_log(
            f"  [{ch_name}] iw={iw}  k_matrix={k_mode}  n_params={n_params}  "
            f"n_data_points={len(p2_arr)}"
        )

        if len(p2_arr) == 0:
            _progress_log(f"  [{ch_name}] no data points — leaving params as zero")
            continue

        # ── per-channel K-matrix model ────────────────────────────────────────
        def _b_model(params, _p2=p2_arr, _ecm=ecm_arr, _ch=ch, _L=L, _mode=k_mode):
            vals = []
            for p2, ecm in zip(_p2, _ecm):
                try:
                    v = _evaluate_k_matrix(
                        _mode, params, p2,
                        Ecm_over_mref=ecm, MN=MN, MK=MK
                    )
                    # apply barrier factor for uncoupled channels
                    if _L > 0:
                        v = p2 ** _L * v
                    vals.append(v)
                except Exception:
                    vals.append(1e6)
            return np.array(vals)

        def _residual_sq(params, _b=b_arr, _model=_b_model):
            pred = _model(params)
            return float(np.sum((_b - pred) ** 2))

        # seed first param with mean B value
        p0_guess = np.zeros(n_params)
        if n_params >= 1:
            p0_guess[0] = float(np.mean(b_arr))

        result = _minimize(
            _residual_sq, p0_guess,
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
        )

        ch_estimate = result.x
        _progress_log(
            f"  [{ch_name}] estimated params: {np.array2string(ch_estimate)}"
        )

        # write into the correct slice of the full parameter vector
        if sl.start is not None and sl.stop is not None:
            p0_estimate_full[sl] = ch_estimate

    _progress_log(
        f"Full p0 estimate: {np.array2string(p0_estimate_full)}"
    )
    return b_pairs_per_channel, p0_estimate_full


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION — BoxQuantization Diagnostic Plot
# ═══════════════════════════════════════════════════════════════════════════════
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_box_quantization_diagnostics(
    par,
    kept_mom2,
    kept_irreps,
    mL,
    MN,
    MK,
    ecm_min=None,
    ecm_max=None,
    n_sweep=400,
    save_dir="./boxq_diagnostics",
    show=False,
    data2cm=None
):
    """
    For each unique (psq, irrep), generate a 3-panel diagnostic plot:
      1. Top: Ecm/mN vs det(K^{-1} - B)
      2. Middle: Overlay of K^{-1} and B (diagonal or eigenvalues)
    Plots are saved to save_dir/irrep_psq.pdf
    """
    par = np.asarray(par, dtype=float)
    os.makedirs(save_dir, exist_ok=True)
    combos = set((tuple(np.asarray(mom).tolist()), irrep) for mom, irrep in zip(kept_mom2, kept_irreps))
    for mom, irrep in combos:
        psq = int(np.sum(np.asarray(mom) ** 2))
        frame_label = _frame_label_from_psq(psq)
        qn_channels = _normalize_qn_channels()
        max_L = 0
        for ch in qn_channels:
            if ch["enabled"]:
                lv = ch.get("L_values")
                if lv is not None:
                    for v in lv:
                        max_L = max(max_L, int(v))
                else:
                    max_L = max(max_L, int(ch.get("L", 0)))
        max_L += 1
        def _calc(J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF):
            return calcFunc_param(J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF, MN, MK, *par)
        kinv = BMat.KMatrix(_calc, isZero)
        chan_list = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
        box_q = BMat.BoxQuantization(frame_label, psq, irrep, chan_list, [max_L], kinv, True)
        box_q.setRefMassL(mL)
        box_q.setMassesOverRef(0, MN, MK)
        # Ecm sweep
        if ecm_min is None or ecm_max is None:
            ecm_vals = np.array([float(np.mean(e)) for e in data2cm]) if data2cm is not None else np.array([1.9, 2.5])
            padding = 0.05 * (ecm_vals.max() - ecm_vals.min() or 0.1)
            ecm_min_ = ecm_vals.min() - padding if ecm_min is None else ecm_min
            ecm_max_ = ecm_vals.max() + padding if ecm_max is None else ecm_max
        else:
            ecm_min_, ecm_max_ = ecm_min, ecm_max
        ecm_sweep = np.linspace(ecm_min_, ecm_max_, n_sweep)
        # Panel 1: det(K^{-1} - B)
        det_vals = []
        kinv_vals = []
        bmat_vals = []
        for ecm in ecm_sweep:
            try:
                # Find the quantum channel for this (psq, irrep)
                qn_channel = None
                for ch in qn_channels:
                    if ch["enabled"] and int(np.sum(np.asarray(mom)**2)) == psq:
                        qn_channel = ch
                        break
                if qn_channel is None:
                    raise RuntimeError("No enabled quantum channel found for this (psq, irrep)")
                # Extract quantum numbers
                J     = int(qn_channel.get("twoJ", 0))
                L     = int(qn_channel.get("L", 0))
                S     = int(qn_channel.get("twoS", 0))
                Lp    = int(qn_channel.get("Lp", L))
                Sp    = int(qn_channel.get("twoSp", S))
                chan  = int(qn_channel.get("chan", 0))
                chanp = int(qn_channel.get("chanp", 0))
                # Build Kinv matrix for this channel and energy
                # (Assume uncoupled for simplicity; extend for coupled if needed)
                Kinv = calcFunc_param(J, Lp, Sp, chanp, L, S, chan, ecm, None, MN, MK, *par)#np.array([[calcFunc_param(J, Lp, Sp, chanp, L, S, chan, ecm, None, MN, MK, *par)]])
                Kinv = np.real(Kinv)  # Ensure it's real
                # print(f"Debug: Ecm={ecm:.4f}, Kinv={Kinv}")
                Bmat = float(np.real(box_q.getBoxMatrixFromEcm(ecm)))
                # print(f"Debug: Ecm={ecm:.4f}, Bmat={Bmat}")
                det_vals.append(Kinv - Bmat)
                kinv_vals.append(Kinv)
                bmat_vals.append(Bmat)

            except Exception:
                det_vals.append(np.nan)
                kinv_vals.append([np.nan]*max_L)
                bmat_vals.append([np.nan]*max_L)
        kinv_vals = np.array(kinv_vals)
        bmat_vals = np.array(bmat_vals)
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios':[2,1.5]})
        # Top panel
        axes[0].plot(ecm_sweep, det_vals, color='C0', lw=2)
        axes[0].set_ylabel(r"$\det(K^{-1} - B)$", fontsize=13)
        axes[0].set_title(f"{irrep}({psq})")
        axes[0].axhline(0, color='gray', ls='--', lw=1)
        axes[0].set_ylim(-.3, .3)
        # Middle panel
        # for i in range(kinv_vals.shape[1]):
        axes[1].plot(ecm_sweep, kinv_vals, label=f"Kinv", color=cm.tab10(0/10), ls='-')
        axes[1].plot(ecm_sweep, bmat_vals, label=f"B", color=cm.tab10(1/10), ls='--')
        axes[1].set_ylabel(r"Matrix elements", fontsize=12)
        axes[1].set_ylim(-3, 3)
        axes[1].legend(fontsize=8, ncol=2)
        # # Bottom panel: spectrum summary
        # axes[2].set_xlabel(r"$E_{cm}/m_N$", fontsize=13)
        # axes[2].set_yticks([])
        # axes[2].set_xlim(ecm_min_, ecm_max_)
        # if nonint_energies is not None:
        #     for e in nonint_energies.get((psq, irrep), []):
        #         axes[2].axvline(e, color='gray', ls=':', lw=1, alpha=0.7)
        # if data2cm is not None and kept_levels is not None:
        #     for idx, (mom2, irr, lev) in enumerate(zip(kept_mom2, kept_irreps, kept_levels)):
        #         if tuple(np.asarray(mom2).tolist()) == mom and irr == irrep:
        #             ecm_mean = float(np.mean(data2cm[idx]))
        #             ecm_std = float(np.std(data2cm[idx]))
        #             axes[2].errorbar(ecm_mean, 0, xerr=ecm_std, fmt='o', color='C3', label='Data' if idx==0 else None)
        # axes[2].set_ylim(-1, 1)
        # axes[2].set_yticks([])
        # axes[2].set_ylabel('')
        fname = f"{irrep}_psq{psq}.pdf"
        fpath = os.path.join(save_dir, fname)
        plt.tight_layout()
        plt.savefig(fpath)
        if show:
            plt.show()
        plt.close(fig)
        print(f"Saved: {fpath}")

def chi2_dependence(par, data, frames, irreps, levels, mL, cov, MN, MK,
                    n=150, n_points=20, n_sigma=1.0, vij=None, param_bounds=None,
                    param_labels=None, save_dir=None, show=False, figsize=(8, 5)):
    """
    1D chi² profile scan: vary each parameter independently around its best-fit
    value while holding all others fixed.  Confirms the fit landed at a true
    minimum and gives a visual / logged sense of parameter sensitivity.

    Parameters
    ----------
    par           : array-like, best-fit parameters
    data/frames/irreps/levels/mL/cov/MN/MK : same as minimizechi2
    n             : int, spectrum root-finding resolution
    n_points      : int, scan points per parameter
    n_sigma       : float, sweep ± n_sigma * sigma_i around each best-fit value.
                    If vij is provided sigma_i = sqrt(vij[i,i]).
                    If vij is None falls back to n_sigma * max(|par[i]|, 1).
    vij           : 2D array (n_params, n_params) or None.
                    Parameter covariance matrix from Fisher/vij.
                    Diagonal entries give per-parameter variance.
                    If None, a scale-based fallback is used.
    param_bounds  : list of (lo, hi) per parameter — overrides both vij and
                    fallback for that parameter
    param_labels  : list of str, human-readable names for each parameter
    save_dir      : str or None — if given, saves one PDF per parameter there
    show          : bool, call plt.show() for each plot
    figsize       : tuple

    Returns
    -------
    profiles : dict keyed by param index
        {
          "label"     : str,
          "values"    : np.ndarray,   # scanned parameter values
          "chi2"      : np.ndarray,   # chi2 at each point
          "best_val"  : float,        # best-fit value
          "best_chi2" : float,        # chi2 at best-fit
          "sigma"     : float,        # estimated sigma used for sweep width
        }
    """
    import matplotlib.pyplot as plt
    import os

    par = np.asarray(par, dtype=float)
    n_params = len(par)

    # ── precompute per-parameter sigmas ──────────────────────────────────────
    if vij is not None:
        vij_arr = np.asarray(vij, dtype=float)
        sigmas  = np.sqrt(np.abs(np.diag(vij_arr)))   # abs guards against tiny negatives
    else:
        sigmas  = None

    # ── chi2 at the best-fit point (reference) ───────────────────────────────
    best_chi2 = _chi2(par, data, frames, irreps, levels, mL, cov, MN, MK, n=n)
    _progress_log(f"chi2 dependence: reference chi2 = {best_chi2:.8g}")

    if vij is not None:
        _progress_log(f"chi2 dependence: using vij sigmas = {sigmas}")
    else:
        _progress_log(f"chi2 dependence: vij not provided, using scale fallback (n_sigma={n_sigma})")

    if param_labels is None:
        param_labels = [f"p{i}" for i in range(n_params)]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    profiles = {}

    for i in range(n_params):
        label  = param_labels[i] if i < len(param_labels) else f"p{i}"
        center = float(par[i])

        # ── scan range ────────────────────────────────────────────────────────
        if param_bounds is not None and i < len(param_bounds):
            # manual override takes priority
            lo, hi = float(param_bounds[i][0]), float(param_bounds[i][1])
            sigma_used = (hi - lo) / (2.0 * n_sigma)   # back-compute for logging
        elif sigmas is not None:
            # physics-informed: ± n_sigma × estimated sigma
            sigma_used = float(sigmas[i])
            half       = n_sigma * sigma_used
            lo, hi     = center - half, center + half
        else:
            # fallback when vij not available
            sigma_used = max(abs(center), 1.0)
            half       = n_sigma * sigma_used
            lo, hi     = center - half, center + half

        _progress_log(
            f"  [{label}]  sigma={sigma_used:.4g}  "
            f"sweep=[{lo:.4g}, {hi:.4g}]  (±{n_sigma}σ)"
        )

        scan_vals = np.linspace(lo, hi, n_points)
        chi2_vals = np.full(n_points, np.nan)

        for j, val in enumerate(scan_vals):
            p_scan    = par.copy()
            p_scan[i] = val
            try:
                chi2_vals[j] = _chi2(p_scan, data, frames, irreps, levels,
                                     mL, cov, MN, MK, n=n)
            except Exception:
                chi2_vals[j] = np.nan

        # ── locate 1D minimum ─────────────────────────────────────────────────
        finite_mask = np.isfinite(chi2_vals)
        if finite_mask.any():
            scan_min_idx  = int(np.nanargmin(chi2_vals))
            scan_min_chi2 = float(chi2_vals[scan_min_idx])
            scan_min_val  = float(scan_vals[scan_min_idx])
        else:
            scan_min_chi2 = np.nan
            scan_min_val  = np.nan

        delta = scan_min_chi2 - best_chi2  # ≈ 0 at a true minimum

        profiles[i] = {
            "label"     : label,
            "values"    : scan_vals,
            "chi2"      : chi2_vals,
            "best_val"  : center,
            "best_chi2" : best_chi2,
            "sigma"     : sigma_used,
        }

        _progress_log(
            f"  [{label}]  best_fit={center:.6g}  sigma={sigma_used:.4g}  "
            f"chi2(best_fit)={best_chi2:.6g}  "
            f"scan_min={scan_min_val:.6g}  chi2(scan_min)={scan_min_chi2:.6g}  "
            f"delta_chi2={delta:+.4g}"
        )

        # ── optional plot ─────────────────────────────────────────────────────
        if save_dir is not None:
            fig, ax = plt.subplots(figsize=figsize)

            # mask penalty values so they don't destroy the y-scale
            _PENALTY_THRESHOLD = 1e10
            plot_chi2 = chi2_vals.copy()
            plot_chi2[plot_chi2 > _PENALTY_THRESHOLD] = np.nan

            ax.plot(scan_vals, plot_chi2, "b-", lw=2, label=r"$\chi^2$ profile")
            ax.axvline(center,        color="red",    ls="--", lw=1.5,
                       label=f"Best fit = {center:.6g}")
            ax.axhline(best_chi2,     color="gray",   ls=":",  lw=1.0,
                       label=rf"$\chi^2_{{\min}}$ = {best_chi2:.6g}")
            ax.axhline(best_chi2 + 1, color="orange", ls=":",  lw=1.0,
                       label=r"$\chi^2_{\min}+1$  (1$\sigma$)")

            # shade the ±1σ region from vij
            if sigmas is not None:
                ax.axvspan(center - sigma_used, center + sigma_used,
                           alpha=0.10, color="red",
                           label=rf"$\pm1\sigma$ from $V_{{ij}}$ = ±{sigma_used:.4g}")

            # ── y-limits: based on finite values only, with headroom ──────────
            finite_plot = plot_chi2[np.isfinite(plot_chi2)]
            if len(finite_plot) > 0:
                ymin = min(finite_plot.min(), best_chi2)
                ymax = finite_plot.max()
                ypad = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
                ax.set_ylim(ymin - ypad, ymax + ypad)

            ax.set_xlabel(label,               fontsize=13)
            ax.set_ylabel(r"$\chi^2$",         fontsize=13)
            ax.set_title(rf"$\chi^2$ dependence — {label}  "
                         rf"($\sigma_{{vij}}={sigma_used:.4g}$)", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            safe_label = label.replace("/", "_").replace(".", "_").replace(" ", "_")
            fpath = os.path.join(save_dir, f"chi2_dep_{safe_label}.pdf")
            plt.savefig(fpath, bbox_inches="tight", dpi=150)
            _progress_log(f"    saved: {fpath}")
            if show:
                plt.show()
            plt.close(fig)

    return profiles

def plot_omega_and_eigenvalues(
    par,
    vij,
    kept_mom2,
    kept_irreps,
    kept_levels,
    data2cm,
    frames_g,
    irreps_g,
    levels_g,
    mL,
    MN,
    MK,
    ecm_min=None,
    ecm_max=None,
    n_sweep=400,
    n_refine=300,
    save_dir="./boxq_diagnostics",
    show=False,
    eig_ylim=(-0.5, 0.5),
    mN_samples=None,
    L=None,
    n_par_samples=300,
    step_mode='adaptive',
    mN_err=None,
):
    """
    Single combined figure across all unique (psq, irrep) combos:
      Panel a  : Omega(mu=1, Ecm) — one curve per (psq, irrep), labeled
      Panel b+ : One sub-panel per eigenvalue — one curve per (psq, irrep)
                 labeled as lambda_i  spectroscopic (irrep(psq))
      Panel c  : Spectrum comparison — data + predicted for all combos
    """
    par = np.asarray(par, dtype=float)
    os.makedirs(save_dir, exist_ok=True)

    _Y_DATA = 1.0
    _Y_PRED = 0.0

    # ── spectroscopic label helper ────────────────────────────────────────────
    _L_LABELS = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}
    def _spec_latex(twoS, L, twoJ):
        L_str = _L_LABELS.get(L, f"L{{{L}}}")
        return rf"${{}}^{{{twoS+1}}}{L_str}_{{{int(twoJ/2)}}}$"

    # ── collect wave labels from config ──────────────────────────────────────
    qn_channels = _normalize_qn_channels()
    config_wave_labels = []
    for ch in qn_channels:
        if not ch.get("enabled", True):
            continue
        twoS = int(ch.get("twoS", 0))
        twoJ = int(ch.get("twoJ", 0))
        lv   = ch.get("L_values")
        if lv is not None:
            for v in sorted(set(int(x) for x in lv)):
                config_wave_labels.append(_spec_latex(twoS, v, twoJ))
        else:
            config_wave_labels.append(_spec_latex(twoS, int(ch.get("L", 0)), twoJ))

    # ── max_L ─────────────────────────────────────────────────────────────────
    max_L = 0
    for ch in qn_channels:
        if ch["enabled"]:
            lv = ch.get("L_values")
            if lv is not None:
                for v in lv:
                    max_L = max(max_L, int(v))
            else:
                max_L = max(max_L, int(ch.get("L", 0)))

    # ── ordered unique combos ─────────────────────────────────────────────────
    seen_combos = {}
    for mom, irrep in zip(kept_mom2, kept_irreps):
        key = (tuple(np.asarray(mom).tolist()), irrep)
        if key not in seen_combos:
            seen_combos[key] = (np.asarray(mom, dtype=int), irrep)
    combos = list(seen_combos.values())   # ordered list of (mom, irrep)

    # ── Ecm range ─────────────────────────────────────────────────────────────
    ecm_centrals = np.array([
        float(np.asarray(data2cm[i], dtype=float).ravel()[0])   # [0] = central only
        for i in range(len(data2cm))
    ])
    padding      = 0.05 * (ecm_centrals.max() - ecm_centrals.min() or 0.1)
    ecm_min_     = ecm_centrals.min() - padding if ecm_min is None else ecm_min
    ecm_max_     = ecm_centrals.max() + padding if ecm_max is None else ecm_max
    ecm_sweep    = np.linspace(ecm_min_, ecm_max_, n_sweep)

    # ── per-combo color assignment ────────────────────────────────────────────
    combo_colors = [f"C{i}" for i in range(len(combos))]
    marker_shapes    = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">",
                        "h", "8", "p", "H"]
    combo_marker_map = {}
    combo_key_to_ci   = {}
    for ci, (mom, irrep) in enumerate(combos):
        key = (tuple(mom.tolist()), irrep)
        combo_marker_map[key] = marker_shapes[ci % len(marker_shapes)]
        combo_key_to_ci[key]  = ci
    # ── NI lines using existing _generate_ni_levels ───────────────────────────
    ni_vlines = []   # (ecm_central, ni_lo_band, ni_hi_band)

    if mN_samples is not None and L is not None:
        mN_arr     = np.asarray(mN_samples, dtype=float)
        mN_central = float(mN_arr[0])
        mN_boots   = mN_arr[1:]
        mL_central = mN_central * L

        unique_moms = {tuple(mom.tolist()): mom for mom, _ in combos}
        seen        = []

        for mom_key, mom_arr_ni in unique_moms.items():
            # central NI levels for this mom
            ni_central = _generate_ni_levels(mL_central, mom_arr_ni, ecm_max_ + 0.1)

            # bootstrap NI levels — generate once per boot sample per mom
            ni_boots_all = []
            for mN_b in mN_boots:
                ni_boots_all.append(
                    np.asarray(_generate_ni_levels(mN_b * L, mom_arr_ni, ecm_max_ + 0.1))
                )

            for ecm_c in ni_central:
                if ecm_c < ecm_min_ or ecm_c > ecm_max_:
                    continue
                if any(abs(ecm_c - s) < 1e-6 for s in seen):
                    continue
                seen.append(ecm_c)

                # for each bootstrap, find the nearest NI level to ecm_c
                boot_vals = []
                for ni_b_arr in ni_boots_all:
                    if len(ni_b_arr) == 0:
                        continue
                    nearest = float(ni_b_arr[np.argmin(np.abs(ni_b_arr - ecm_c))])
                    boot_vals.append(nearest)

                if boot_vals:
                    ni_lo = float(np.percentile(boot_vals, 16))
                    ni_hi = float(np.percentile(boot_vals, 84))
                else:
                    ni_lo = ni_hi = ecm_c

                ni_vlines.append((ecm_c, ni_lo, ni_hi))

    def _draw_ni_lines(ax):
        for ecm_c, ni_lo, ni_hi in ni_vlines:
            ax.axvline(ecm_c, color="gray", lw=1.0, ls="--", alpha=0.5, zorder=0)
            ax.axvspan(ni_lo, ni_hi, color="gray", alpha=0.10, zorder=0)
     # ── sweep Omega and eigenvalues for every combo ───────────────────────────
    combo_omega  = {}
    combo_eigs   = {}
    combo_box_q  = {}
    n_eigs_global = 0

    # NI positions for masking — use central mL
    ni_mask_arr = np.array([ecm_c for ecm_c, _, _ in ni_vlines]) if ni_vlines else np.array([])

    def _mask_near_ni(vals):
        """Set a minimal NaN gap only at NI poles to avoid drawing through singular points."""
        out = vals.copy()
        if ni_mask_arr.size == 0:
            return out

        # Leave nearby structure visible: only mask the sweep point closest to each NI pole.
        for ni in ni_mask_arr:
            idx = int(np.argmin(np.abs(ecm_sweep - ni)))
            out[idx] = np.nan
        return out

    for mom, irrep in combos:
        key         = (tuple(mom.tolist()), irrep)
        psq         = int(np.sum(mom ** 2))
        frame_label = _frame_label_from_psq(psq)

        def _calc(J, Lp, Sp, chanp, L_, S, chan, Ecm, pSqF, _p=par):
            return calcFunc_param(J, Lp, Sp, chanp, L_, S, chan, Ecm, pSqF, MN, MK, *_p)

        kinv     = BMat.KMatrix(_calc, isZero)
        chan_list = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]
        box_q    = BMat.BoxQuantization(frame_label, psq, irrep, chan_list, [max_L], kinv, True)
        box_q.setRefMassL(mL)
        box_q.setMassesOverRef(0, MN, MK)
        combo_box_q[key] = box_q

        omega_vals = []
        eig_vals   = []
        for ecm in ecm_sweep:
            try:
                omega_vals.append(box_q.getOmegaFromEcm(1.0, float(ecm)))
            except Exception:
                omega_vals.append(np.nan)
            try:
                eigs = box_q.getEigenvaluesFromEcm(float(ecm))
                eig_vals.append(eigs)
            except Exception:
                eig_vals.append(None)

        omega_vals = np.array(omega_vals, dtype=float)
        # mask NI discontinuities → clean gaps
        omega_vals = _mask_near_ni(omega_vals)
        combo_omega[key] = omega_vals

        # eigenvalue array
        n_eigs = 0
        for ev in eig_vals:
            if ev is not None:
                n_eigs = len(ev)
                break
        n_eigs_global = max(n_eigs_global, n_eigs)

        eig_array = np.full((len(ecm_sweep), n_eigs), np.nan)
        for i, ev in enumerate(eig_vals):
            if ev is not None and len(ev) == n_eigs:
                eig_array[i] = ev

        # mask NI discontinuities per eigenvalue
        for j in range(n_eigs):
            eig_array[:, j] = _mask_near_ni(eig_array[:, j])

        combo_eigs[key] = eig_array
    # ── auto ylim helper (shared by Omega and eigenvalue panels) ─────────────
    def _autolim(all_vals_list, hard_max=30.0):
        """95th percentile of |vals| across all combos, capped at hard_max."""
        combined = np.concatenate([
            v[np.isfinite(v)] for v in all_vals_list if np.any(np.isfinite(v))
        ])
        if len(combined) == 0:
            return hard_max
        lim = float(np.percentile(np.abs(combined), 80)) * 1.3
        return min(hard_max, max(lim, 1e-6))
    omega_lim = _autolim([combo_omega[k] for k in combo_omega])
    eig_lims  = []
    for j in range(n_eigs_global):
        cols = []
        for key in combo_eigs:
            ea = combo_eigs[key]
            if j < ea.shape[1]:
                cols.append(ea[:, j])
        eig_lims.append(_autolim(cols, hard_max=eig_ylim[1]))

    # ── layout ────────────────────────────────────────────────────────────────
    show_lambda_panels = n_eigs_global > 1
    n_lambda           = n_eigs_global if show_lambda_panels else 0
    n_panels           = 1 + n_lambda + 1
    height_ratios      = [2.5] + [1.0] * n_lambda + [0.75]
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(10, 3 + 1.8 * (n_lambda + 1)),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]
# ── Panel a: Omega ────────────────────────────────────────────────────────
    ax_omega = axes[0]
    _draw_ni_lines(ax_omega)
    ax_omega.axhline(0, color="gray", ls="--", lw=1)
    ax_omega.set_ylabel(r"$\Omega$")
    ax_omega.set_ylim(-omega_lim, omega_lim)

    for ci, (mom, irrep) in enumerate(combos):
        key        = (tuple(mom.tolist()), irrep)
        color      = combo_colors[ci]
        omega_vals = combo_omega[key]
        ax_omega.plot(ecm_sweep, omega_vals, color=color, lw=1.8,
                      label=rf"{_format_irrep_label(irrep)}({int(np.sum(mom**2))})")

    _n_combos = len(combos)
    _ncol = min(_n_combos, max(1, (_n_combos + 1) // 2))   # at most 2 rows
    ax_omega.legend(loc="lower center", ncol=_ncol, frameon=True,
                    fontsize="small", columnspacing=0.8, handlelength=1.2)
    # ── Panel b+: eigenvalues ─────────────────────────────────────────────────
    if show_lambda_panels:
        for j in range(n_eigs_global):
            ax  = axes[1 + j]
            lim = eig_lims[j] if j < len(eig_lims) else eig_ylim[1]
            _draw_ni_lines(ax)
            ax.axhline(0, color="gray", ls="--", lw=1)
            ax.set_ylim(-lim, lim)
            ax.set_yticks([0.0])

            wave_str = config_wave_labels[j] if j < len(config_wave_labels) else f"wave {j+1}"
            ax.set_ylabel(rf"$\lambda_{{{j+1}}}$ {wave_str}")

            for ci, (mom, irrep) in enumerate(combos):
                key       = (tuple(mom.tolist()), irrep)
                color     = combo_colors[ci]
                eig_array = combo_eigs[key]
                if j >= eig_array.shape[1]:
                    continue
                ax.plot(ecm_sweep, eig_array[:, j], color=color, lw=1.6,
                        label=rf"{_format_irrep_label(irrep)}({int(np.sum(mom**2))})")

    # ── Panel c: spectrum comparison ─────────────────────────────────────────
    ax_spec = axes[-1]
    _draw_ni_lines(ax_spec)
    # # non-interacting bands
    # if mN_samples is not None and L is not None:
    #     two_pi_over_L  = 2.0 * np.pi / L
    #     mN_arr         = np.asarray(mN_samples, dtype=float)
    #     mN_central     = float(mN_arr[0])
    #     mN_boots       = mN_arr[1:]
    #     seen_ni_means  = []

    #     # collect unique mom vectors across all combos
    #     unique_moms = {tuple(mom.tolist()): mom for mom, _ in combos}

    #     for mom_key, mom_arr_ni in unique_moms.items():
    #         psq     = int(np.sum(mom_arr_ni ** 2))
    #         n_shells = 4
    #         for nx in range(-n_shells, n_shells + 1):
    #             for ny in range(-n_shells, n_shells + 1):
    #                 for nz in range(-n_shells, n_shells + 1):
    #                     n1   = np.array([nx, ny, nz])
    #                     n2   = mom_arr_ni - n1
    #                     n1sq = float(np.sum(n1 ** 2))
    #                     n2sq = float(np.sum(n2 ** 2))

    #                     E1c   = np.sqrt(mN_central**2 + n1sq * two_pi_over_L**2)
    #                     E2c   = np.sqrt(mN_central**2 + n2sq * two_pi_over_L**2)
    #                     Ecm_c = np.sqrt(max((E1c + E2c)**2 - psq * two_pi_over_L**2, 0.0)) / mN_central

    #                     if Ecm_c < ecm_min_ or Ecm_c > ecm_max_:
    #                         continue
    #                     if any(abs(Ecm_c - prev) < 1e-6 for prev in seen_ni_means):
    #                         continue
    #                     seen_ni_means.append(Ecm_c)

    #                     nonint_boots = []
    #                     for m in mN_boots:
    #                         E1 = np.sqrt(m**2 + n1sq * two_pi_over_L**2)
    #                         E2 = np.sqrt(m**2 + n2sq * two_pi_over_L**2)
    #                         nonint_boots.append(
    #                             np.sqrt(max((E1 + E2)**2 - psq * two_pi_over_L**2, 0.0)) / m
    #                         )
    #                     nonint_boots = np.array(nonint_boots)
    #                     ni_lo = float(np.percentile(nonint_boots, 16))
    #                     ni_hi = float(np.percentile(nonint_boots, 84))

    #                     ax_spec.axvspan(ni_lo, ni_hi, color="gray", alpha=0.18, zorder=0)
    #                     ax_spec.axvline(Ecm_c, color="gray", lw=1.2, ls="--", alpha=0.7, zorder=1)

    # data levels — colour matches the omega-panel line for this (mom, irrep) combo
    data_plotted = False
    for idx, (mom2, irr, lev) in enumerate(zip(kept_mom2, kept_irreps, kept_levels)):
        ecm_arr     = np.asarray(data2cm[idx], dtype=float)
        ecm_gv      = gv.dataset.avg_data(ecm_arr, bstrap=True)
        ecm_central = float(gv.mean(ecm_gv))
        ecm_err     = float(gv.sdev(ecm_gv))
        key         = (tuple(np.asarray(mom2).tolist()), irr)
        ci_data     = combo_key_to_ci.get(key, 0)
        pt_color    = combo_colors[ci_data]
        marker      = combo_marker_map.get(key, "o")

        ax_spec.errorbar(
            ecm_central, _Y_DATA,
            xerr=ecm_err,
            fmt=marker, color=pt_color,
            ms=8, lw=1.5, capsize=5, capthick=1.2,
            markeredgecolor="black", markeredgewidth=0.7,
            zorder=10,
            label="Data" if not data_plotted else None,
        )
        ax_spec.annotate(
            str(int(lev)),
            xy=(ecm_central, _Y_DATA),
            xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=8, color="black",
        )
        data_plotted = True

    # predicted levels
    try:
        p2_all  = _model_datap2(par, frames_g, irreps_g, levels_g, mL, MN, MK, n=n_refine,
                                data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
                                step_mode=step_mode, n_refine=n_refine, mN_err=mN_err)
        ecm_all = 2.0 * np.sqrt(np.asarray(p2_all, dtype=float) + 1.0)
        n_pred  = len(ecm_all)                              # ← anchor expected size

        # Predicted-level uncertainty propagation is intentionally disabled
        # when predicted error bars are not shown.
        pred_plotted = False
        for k in range(n_pred):
            pred_e = float(ecm_all[k])
            mom_k  = np.asarray(kept_mom2[k]).ravel()
            irr_k  = kept_irreps[k]
            key_k  = (tuple(mom_k.tolist()), irr_k)
            marker = combo_marker_map.get(key_k, "s")
            xerr = None
            # uncertainty propagation disabled
            # if np.isfinite(pred_sigma[k]):

            ci_pred    = combo_key_to_ci.get(key_k, 0)
            pred_color = combo_colors[ci_pred]
            ax_spec.errorbar(
                pred_e, _Y_PRED,
                xerr=xerr,
                fmt=marker, color=pred_color,
                ms=8, lw=1.5, capsize=5, capthick=1.2,
                zorder=11,
                markeredgecolor="black", markeredgewidth=0.6,
                label="Predicted" if not pred_plotted else None,
            )
            pred_plotted = True
        # pred_plotted = False
        # for k in range(n_pred):
        #     pred_e = float(ecm_all[k])
        #     mom_k  = np.asarray(kept_mom2[k]).ravel()      # ← .ravel() fixes hashability
        #     irr_k  = kept_irreps[k]
        #     key_k  = (tuple(mom_k.tolist()), irr_k)
        #     marker = combo_marker_map.get(key_k, "s")

        #     xerr = None
        #     if pred_boots:
        #         col  = np.array([draw[k] for draw in pred_boots], dtype=float)
        #         col  = col[np.isfinite(col)]
        #         if len(col) > 1:
        #             xerr = float(np.std(col))
        #         _progress_log(
        #             f"  predicted level {k}: Ecm={pred_e:.4f}  "
        #             f"std={xerr if xerr is not None else 0.0:.6f}  "
        #             f"n_draws={len(col)}"
        #         )
        #     if len(col) > 1:
        #         p16 = float(np.percentile(col, 16))
        #         p84 = float(np.percentile(col, 84))
        #         xerr_lo = pred_e - p16
        #         xerr_hi = p84 - pred_e
        #         print(f" xerr for level {k}: -{xerr_lo:.6f} +{xerr_hi:.6f}")
        #     ax_spec.errorbar(
        #         pred_e, _Y_PRED,
        #         xerr= xerr,
        #         fmt=marker, color="C3",
        #         ms=8, lw=1.5, capsize=5, capthick=1.2,
        #         zorder=11,
        #         markeredgecolor="black", markeredgewidth=0.6,
        #         label="Predicted" if not pred_plotted else None,
        #     )
        #     pred_plotted = True

    except Exception as e:
        _progress_log(f"  predicted levels failed: {e}")
        import traceback
        _progress_log(traceback.format_exc())

    # spectrum panel aesthetics
    ax_spec.set_yticks([_Y_PRED, _Y_DATA])
    ax_spec.set_yticklabels(["Pred.", "Data"])
    ax_spec.set_ylim(-0.5, 1.5)
    for y_ref in [_Y_PRED, _Y_DATA]:
        ax_spec.axhline(y_ref, color="gray", ls=":", lw=0.5, alpha=0.3)
    # ax_spec.legend(fontsize=8, loc="upper right", ncol=3, framealpha=0.7)

    axes[-1].set_xlabel(r"$E_{cm}/m_N$", fontsize=13)

    fname = "omega_eigs_combined.pdf"
    fpath = os.path.join(save_dir, fname)
    plt.tight_layout()
    plt.savefig(fpath)
    if show:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# QC NULL-EIGENVECTOR (WAVE DECOMPOSITION) ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
#
# Physics background
# ------------------
# The Lüscher quantization condition det M(E) = 0 is satisfied at the finite-
# volume energy levels E*.  Near each E*, the QC matrix factorises as
#
#     M(E) ≈ R · (E - E*)
#
# where R = dM/dE|_{E*}.  The null eigenvector v* of M(E*) — i.e.
# M(E*) v* = 0 — lives in the partial-wave space and reveals which wave(s)
# couple most strongly to that level.
#
# Method (rank-1 perturbation)
# ----------------------------
# We cannot read M(E*) directly from the BMat Python API (only eigenvalues
# and the scalar Omega are exposed).  However, if we shift the diagonal
# element K^{-1}_{kk} → K^{-1}_{kk} + ε, we obtain a perturbed matrix
#
#     M_k(E) = M(E) + ε · e_k ⊗ e_k
#
# By first-order perturbation theory, the formerly-zero eigenvalue of M_k(E*)
# is
#     λ_0(M_k) ≈ ε · |v* · e_k|²
#
# so   |v* · e_k|²  =  λ_0(M_k(E*)) / ε.
#
# A separate BoxQuantization with the perturbed K-matrix is built for each
# wave probe k, evaluated at E*, and the minimum-|λ| eigenvalue is extracted.
# Normalising these squared components to sum to 1 gives the fractional wave
# content of each level.
# ══════════════════════════════════════════════════════════════════════════════


def _get_active_wave_info():
    """
    Returns a list of active partial waves — one per QC matrix row/column.

    Each element is a dict with keys:
        'L'      : int  — orbital angular momentum
        'twoS'   : int  — 2*S
        'twoJ'   : int  — 2*J
        'label'  : str  — spectroscopic LaTeX string, e.g. '${}^3S_1$'
    """
    _L_LABELS = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}
    qn_channels = _normalize_qn_channels()
    waves = []
    for ch in qn_channels:
        if not ch["enabled"]:
            continue
        twoJ = int(ch.get("twoJ", 0))
        twoS = int(ch.get("twoS", 0))
        lv = ch.get("L_values")
        if lv is not None:
            # Coupled channel: one wave per distinct L value
            for L in sorted(set(int(v) for v in lv)):
                L_str = _L_LABELS.get(L, f"L{L}")
                label = rf"${{}}^{{{twoS + 1}}}{L_str}_{{{twoJ // 2}}}$"
                waves.append({"L": L, "twoS": twoS, "twoJ": twoJ, "label": label})
        else:
            L = int(ch.get("L", 0))
            L_str = _L_LABELS.get(L, f"L{L}")
            label = rf"${{}}^{{{twoS + 1}}}{L_str}_{{{twoJ // 2}}}$"
            waves.append({"L": L, "twoS": twoS, "twoJ": twoJ, "label": label})
    return waves


def eigenvector_decomposition(
    par,
    data2cm,
    kept_mom2,
    kept_irreps,
    mL,
    MN,
    MK,
    eps=1e-3,
    n_refine=100,
    step_mode="adaptive",
):
    """
    Compute the QC null-eigenvector composition for every fitted energy level.

    At each QC solution E*, the Luscher matrix is Omega = B(E*) - K^-1(E*).
    det(Omega) = 0 means the smallest singular value is ~0 and the corresponding
    right singular vector (Vh[-1] from SVD) is the null eigenvector.
    |v_i|^2 gives the fractional coupling of the level to wave i.

    Returns list of dict per QC solution with keys:
        mom, irrep, level_idx, Ecm, wave_labels, weights,
        eigenvector, singular_values, dominant_wave, dominant_idx
    """
    _require_bmat()
    par = np.asarray(par, dtype=float)

    wave_info = _get_active_wave_info()
    n_waves_cfg = len(wave_info)
    if n_waves_cfg == 0:
        _progress_log("[eigenvec] No active waves found in config.")
        return []

    results = []

    # group levels by unique (frame, irrep)
    seen_frame_irreps = {}
    for idx, (mom, irrep) in enumerate(zip(kept_mom2, kept_irreps)):
        key = (tuple(int(x) for x in np.asarray(mom).ravel()), irrep)
        seen_frame_irreps.setdefault(key, []).append(idx)

    # max_L from active channels
    qn_channels = _normalize_qn_channels()
    max_L_ch = 0
    for ch in qn_channels:
        if ch["enabled"]:
            lv = ch.get("L_values")
            if lv is not None:
                for v in lv:
                    max_L_ch = max(max_L_ch, int(v))
            else:
                max_L_ch = max(max_L_ch, int(ch.get("L", 0)))
    bmat_max_L = max_L_ch + 1

    chan_list = [BMat.DecayChannelInfo("n", "n", 1, 1, True, True)]

    for (mom_tuple, irrep), level_indices in seen_frame_irreps.items():
        mom    = np.array(mom_tuple, dtype=int)
        psq    = int(np.sum(mom ** 2))
        frame_label = _frame_label_from_psq(psq)

        data_ecm_this = [float(data2cm[i][0]) for i in level_indices]
        nsols  = len(data_ecm_this)
        epsaux = 0.1 / float(n_refine)

        _progress_log(
            f"[eigenvec] {irrep} PSq={psq}  {nsols} level(s): "
            f"{[f'{e:.5f}' for e in data_ecm_this]}"
        )

        solutions = _findspectrum_irrep(
            epsaux, nsols, mom, mL, par, irrep, MN, MK,
            data_ecm=data_ecm_this, debug=False,
            step_mode=step_mode, n_refine=n_refine,
        )
        if not solutions:
            _progress_log(f"[eigenvec]   No QC solutions found.")
            continue

        # Build a recording BoxQuantization: intercepts every K^-1 call so
        # we have the actual values without needing to pass pSqFuncList manually.
        _kinv_store = {}   # (J, Lp, Sp, chanp, L, S, chan) -> float

        def _recording_cf(J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF,
                           _par=par, _store=_kinv_store):
            val = float(calcFunc_param(
                J, Lp, Sp, chanp, L, S, chan, Ecm, pSqF, MN, MK, *_par
            ))
            _store[(int(J), int(Lp), int(Sp), int(chanp),
                    int(L),  int(S),  int(chan))] = val
            return val

        kinv_rec = BMat.KMatrix(_recording_cf, isZero)
        box_q = BMat.BoxQuantization(
            frame_label, psq, irrep, chan_list, [bmat_max_L], kinv_rec, True
        )
        box_q.setRefMassL(mL)
        box_q.setMassesOverRef(0, MN, MK)

        for sol_idx, ecm_star in enumerate(solutions):
            # Trigger K^-1 evaluation so _kinv_store is populated at this E*
            _kinv_store.clear()
            try:
                _ = box_q.getEigenvaluesFromEcm(float(ecm_star))
            except Exception as exc:
                _progress_log(f"[eigenvec]   getEigenvaluesFromEcm failed: {exc}")
                continue

            n = _get_bmat_size(box_q, float(ecm_star))

            # Build Omega = B - K^-1 as an n x n real matrix
            Omega = np.zeros((n, n), dtype=float)
            for iw in range(n):
                for jw in range(n):
                    B_ij = float(np.real(
                        box_q.getBoxMatrixFromEcm(float(ecm_star), iw, jw)
                    ))
                    kinv_ij = 0.0
                    if iw < n_waves_cfg and jw < n_waves_cfg:
                        wi = wave_info[iw]
                        wj = wave_info[jw]
                        J_i  = wi["twoJ"] // 2
                        J_j  = wj["twoJ"] // 2
                        L_i, S_i = int(wi["L"]), wi["twoS"] // 2
                        L_j, S_j = int(wj["L"]), wj["twoS"] // 2
                        if J_i == J_j:
                            key_ij = (J_i, L_i, S_i, 0, L_j, S_j, 0)
                            kinv_ij = _kinv_store.get(key_ij, 0.0)
                    Omega[iw, jw] = B_ij - kinv_ij

            _progress_log(
                f"[eigenvec]   E*={ecm_star:.7f}  Omega({n}x{n}):\n"
                + "\n".join(
                    "    " + "  ".join(f"{Omega[i,j]:+.4e}" for j in range(n))
                    for i in range(n)
                )
            )

            # SVD: s[-1] ~ 0 is the null singular value; Vh[-1] is null vector
            U, s, Vh = np.linalg.svd(Omega)
            eigenvec = Vh[-1]

            _progress_log(
                f"[eigenvec]   singular values: {np.array2string(s, precision=4)}"
            )
            _progress_log(
                f"[eigenvec]   null vector Vh[-1]: "
                f"{np.array2string(eigenvec, precision=4)}"
            )

            # |v_i|^2 normalised to sum = 1
            raw_w = eigenvec[:n_waves_cfg] ** 2
            total = raw_w.sum()
            weights = raw_w / total if total > 1e-30 else np.ones(n_waves_cfg) / n_waves_cfg

            dom_idx = int(np.argmax(weights))
            level_data_idx = (
                level_indices[sol_idx] if sol_idx < len(level_indices)
                else level_indices[-1]
            )

            for k, wv in enumerate(wave_info):
                _progress_log(
                    f"[eigenvec]     wave {k} ({wv['label']:8s})  "
                    f"|v|^2={weights[k]:.4f}  ({weights[k]*100:.1f}%)"
                )
            _progress_log(
                f"[eigenvec]   -> dominant wave: {wave_info[dom_idx]['label']}  "
                f"({weights[dom_idx]*100:.1f}%)"
            )

            results.append({
                "mom":             mom_tuple,
                "irrep":           irrep,
                "level_idx":       level_data_idx,
                "Ecm":             float(ecm_star),
                "wave_labels":     [w["label"] for w in wave_info],
                "weights":         weights,
                "eigenvector":     eigenvec[:n_waves_cfg],
                "singular_values": s,
                "dominant_wave":   wave_info[dom_idx]["label"],
                "dominant_idx":    dom_idx,
            })

    return results

def print_eigenvector_decomposition(results, log_fn=None):
    """
    Pretty-print the eigenvector decomposition table.

    log_fn : callable(str) or None.  If provided, each line is sent there
             (e.g. lambda msg: runner._log(logging.INFO, msg)) so output
             appears in the run log.  Falls back to print() if None.
    """
    _out = log_fn if log_fn is not None else print
    if not results:
        _out("[eigenvec] No results to display.")
        return

    wave_labels = results[0]["wave_labels"]
    n_waves = len(wave_labels)
    header_waves = "  ".join(f"{lbl:>12s}" for lbl in wave_labels)
    sep = "-" * (50 + 14 * n_waves)
    _out("")
    _out("=" * 60)
    _out("  QC Null-Eigenvector Wave Decomposition")
    _out("=" * 60)
    _out(f"  {'irrep':>10s}  {'PSq':>4s}  {'E*/mN':>10s}  {header_waves}  dominant")
    _out(sep)
    for r in results:
        psq = int(sum(x ** 2 for x in r["mom"]))
        pct_strs = "  ".join(f"{w * 100:>11.1f}%" for w in r["weights"])
        _out(
            f"  {r['irrep']:>10s}  {psq:>4d}  {r['Ecm']:>10.6f}  {pct_strs}  "
            f"{r['dominant_wave']}"
        )
    _out(sep)
    _out("")


def plot_eigenvector_decomposition(
    results,
    save_path=None,
    show=False,
    figsize=None,
    title=None,
):
    """
    Stacked bar chart of the QC null-eigenvector wave composition per level.

    Each bar represents one QC solution E*.  Bar segments show |v* · e_k|²
    (the squared component of the null eigenvector in each partial-wave
    direction), normalised to sum to 1.  A coloured star above each bar
    marks the dominant wave.

    Parameters
    ----------
    results   : list of dict from eigenvector_decomposition()
    save_path : str or None
    show      : bool
    figsize   : tuple or None  (auto-scaled from n_levels if None)
    title     : str or None
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm

    if not results:
        _progress_log("[eigenvec plot] No decomposition results to plot.")
        return

    wave_labels = results[0]["wave_labels"]
    n_waves = len(wave_labels)
    n_levels = len(results)

    # One colour per wave
    if n_waves <= 10:
        colors = list(mcm.tab10.colors[:n_waves])
    else:
        colors = [mcm.tab20.colors[i % 20] for i in range(n_waves)]

    # x-axis labels: irrep + E*
    x_labels = [
        f"{r['irrep']}\n$E^*\\!={r['Ecm']:.4f}$" for r in results
    ]
    xs = np.arange(n_levels)
    bar_width = 0.65

    if figsize is None:
        figsize = (max(6, n_levels * 1.3 + 1), 5)

    fig, ax = plt.subplots(figsize=figsize)

    bottoms = np.zeros(n_levels)
    for k in range(n_waves):
        vals = np.array([r["weights"][k] for r in results])
        ax.bar(
            xs, vals,
            bottom=bottoms,
            width=bar_width,
            label=wave_labels[k],
            color=colors[k],
            edgecolor="white",
            linewidth=0.5,
        )
        # Annotate segments that are large enough to label
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.08:
                ax.text(
                    xs[i], b + v / 2.0,
                    f"{v * 100:.0f}%",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
        bottoms += vals

    # Star above each bar indicating the dominant wave
    for i, r in enumerate(results):
        ax.plot(
            xs[i], 1.07, marker="*", markersize=12,
            color=colors[r["dominant_idx"]], linestyle="none",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel(r"$|v_0 \cdot \hat{e}_k|^2$  (fractional wave content)", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.6)
    ax.legend(title="Partial wave", loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xlabel("Energy level", fontsize=11)

    if title is None:
        title = "QC Null-Eigenvector Wave Decomposition"
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        _progress_log(f"[eigenvec plot] Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)