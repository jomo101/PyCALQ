"""
fit_par_worker.py — process-pool worker entry points for my_fit_model_par.py.

This module must be importable by name in spawned child processes (it lives in
the repo root, which is on sys.path), because multiprocessing pickles the
initializer/task functions by module-qualified reference.

Each worker builds its OWN spectrum engine (its own BMat objects and B-matrix
tables) from the spec sent at pool startup, then serves fast evaluations:

  eval_epred(par)                  -> Ecm predictions for one parameter vector
  eval_epred_blocks((par, idxs))   -> predictions for a subset of blocks
  run_local_fit(job)               -> a full local least-squares fit (multistart)
"""

import os

# Keep BLAS single-threaded inside workers: the engine batches many small
# matrices, where BLAS threading only adds overhead. (Set before numpy spins
# up its thread pools in this process, when possible.)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import importlib.util
import sys
import time

import numpy as np

_MOD = None       # the my_fit_model_par module, loaded from spec["module_path"]
_ENGINE = None
_FIT_CTX = None   # dict(data, cov, observable, ni_sum) or None
_WHITEN = None

_DEBUG_PATH = os.environ.get("HPW_PAR_DEBUG_DIR")


def _dbg(msg):
    if _DEBUG_PATH:
        try:
            with open(os.path.join(_DEBUG_PATH, f"parlog_{os.getpid()}.txt"), "a") as f:
                f.write(f"{time.time():.3f} {msg}\n")
        except Exception:
            pass


def init_worker(spec):
    global _MOD, _ENGINE, _FIT_CTX, _WHITEN
    _dbg("init_worker: start")
    module_path = spec["module_path"]
    name = "fit_par_engine_worker"
    if name in sys.modules:
        _MOD = sys.modules[name]
    else:
        mspec = importlib.util.spec_from_file_location(name, module_path)
        _MOD = importlib.util.module_from_spec(mspec)
        sys.modules[name] = _MOD
        mspec.loader.exec_module(_MOD)

    _MOD.set_quantum_numbers(spec["quantum_numbers"])
    # Workers verify too: basis-ordering candidates must be resolved the same
    # way as in the parent, and verification is what selects the candidate.
    _MOD.set_progress_logger(lambda *_a, **_k: None)
    _ENGINE = _MOD.build_engine_from_spec(spec)

    fc = spec.get("fit_ctx")
    if fc is not None:
        _FIT_CTX = {
            "data": np.asarray(fc["data"], dtype=float),
            "cov": np.asarray(fc["cov"], dtype=float),
            "observable": fc["observable"],
            "ni_sum": (np.asarray(fc["ni_sum"], dtype=float)
                       if fc.get("ni_sum") is not None else None),
        }
        _WHITEN = _MOD._whitener(_FIT_CTX["cov"])
    _dbg("init_worker: done")


def eval_epred(par):
    _dbg(f"eval_epred: start {par}")
    out = _ENGINE.predict_ecm(np.asarray(par, dtype=float)).tolist()
    _dbg("eval_epred: done")
    return out


def eval_epred_blocks(args):
    par, idxs = args
    return _ENGINE.predict_ecm(np.asarray(par, dtype=float),
                               block_idx=list(idxs)).tolist()


def run_local_fit(job):
    """One serial least-squares fit inside this worker (multistart/bootstrap).

    job["data"] (optional) overrides the fit-context data vector — used by
    the bootstrap driver to refit each bootstrap sample. The covariance
    (whitener) stays the central one, as in the old bootstrap code.
    """
    from scipy.optimize import least_squares
    if _FIT_CTX is None:
        return {"ok": False, "error": "worker has no fit context"}
    data = _FIT_CTX["data"]
    if job.get("data") is not None:
        data = np.asarray(job["data"], dtype=float)
    obs = _FIT_CTX["observable"]
    ni = _FIT_CTX["ni_sum"]
    MN = _ENGINE.MN

    def resid(p):
        return _MOD._resid_whitened(_ENGINE.predict_ecm(p), data, _WHITEN,
                                    obs, MN, ni)

    p0 = np.asarray(job["p0"], dtype=float)
    kwargs = dict(method="trf", jac="2-point",
                  x_scale="jac",
                  ftol=job.get("ftol", 1e-10),
                  xtol=job.get("xtol", 1e-12),
                  gtol=1e-12,
                  max_nfev=int(job.get("n_maxiter", 1000)))
    bounds = job.get("bounds")
    if bounds:
        kwargs["bounds"] = (np.array([b[0] for b in bounds], dtype=float),
                            np.array([b[1] for b in bounds], dtype=float))
    try:
        res = least_squares(resid, p0, **kwargs)
        return {"ok": True, "x": res.x.tolist(),
                "chi2": float(np.sum(res.fun ** 2)),
                "status": int(res.status)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
