"""PyCALQ task adapter for the QC2 Luscher quantization-condition fitter.

This module is the *only* PyCALQ-aware file in QC2. The engine modules
(``run_HPW_fit.py``, ``my_fit_model.py``, ``my_fit_model_par.py``,
``fit_par_worker.py``, ``plotting.py``) are carried over verbatim from the
upstream NN-tools research repo, so re-syncing them is a plain copy with no
merge conflict. Everything PyCALQ-specific -- the task class, the YAML->JSON
config bridge, project-directory wiring -- lives here.
"""

import glob
import json
import logging
import os

import QC2  # noqa: F401  -- puts the QC2 dir on sys.path (see QC2/__init__.py)
from QC2.run_HPW_fit import GenericFitRunner

doc = '''
single_channel_fit - fits a K-matrix parametrization to a finite-volume
    spectrum through the Luscher quantization condition (B-matrix / BMAT),
    then reports phase shifts, mixing angles and fit diagnostics.

    Requires the TwoHadronsInBox python bindings (pythib). If BMat is not on
    PYTHONPATH, point QC2 at the build directory with:
        export BMAT_PATH=/path/to/pythib

    Two usage modes. Mode 1 (auto-config) covers the common single-wave case
    from YAML alone; mode 2 (config_file) exposes every fitter flag.

task input (mode 1 - auto-config from YAML, recommended)
---
single_channel_fit:                 #required
  data_file: /path/spectrum.hdf5    #not required #default: auto-discovered
  lattice_size: 48                  #not required #default: from ensemble_id
  continuum: true                   #not required #default: true
  cutoff: 10.0                      #not required #default: 10.0

  scattering:                       #not required #default: fit all levels found
    'N(0)_ref,N(0)_ref':                #channel name
      PSQ0:                             #momentum frame
        - T1g: [0, 1, 2, 3]             #irrep: level indices
      PSQ1:
        - A2: [0, 1]
        - E:  [0, 1]

  # Quantum numbers. twoS is derived from general/particles.py where it is
  # unambiguous; NN admits both S=0 and S=1, so set twoS explicitly there.
  L: 0                              #not required #default: 0 (S-wave); accepts 0/1/2 or S/P/D
  twoS: 2                           #not required #default: minimum allowed, with a warning
  twoJ: 2                           #not required #default: |2L - twoS|
  k_matrix: ere1_vs                 #not required #default: polynomial
  params:                           #required in mode 2
    - {name: a0, initial: -0.2}
    - {name: r0, initial: 0.7}

  # A fully explicit channel list overrides all inference above, and is the
  # only way to express coupled blocks (e.g. 3S1-3D1 with a mixing angle):
  quantum_numbers:                  #not required
    channels:
      - name: 3S1_3D1
        twoJ: 2
        L_values: [0, 2]
        twoS: 2
        enabled: true
        k_matrix: [poly_s, poly_s, epsilon]
        params:
          - [{name: A_S, initial: -0.2}, {name: B_S, initial: 0.7}]
          - [{name: A_D, initial: -0.01}]
          - [{name: eps0, initial: 0.05}]

  # fit settings (all optional - defaults shown)
  strategy: least_squares           #not required #default: least_squares
  n_starts: 20                      #not required #default: 20
  n_refine: 400                     #not required #default: 400
  n_maxiter: 10000                  #not required #default: 10000
  xatol: 1.0e-6                     #not required #default: 1.0e-6
  fatol: 1.0e-6                     #not required #default: 1.0e-6
  step_mode: adaptive               #not required #default: adaptive
  fit_mode: omega                   #not required #default: omega
  fit_observable: shift             #not required #default: shift
  fit_cov_from: shift               #not required #default: matches fit_observable
  param_bounds: []                  #not required #default: none
  compute_vij: true                 #not required #default: true
  auto_p0: false                    #not required #default: false
  skip_minimization: false          #not required #default: false
  MN: 1.0                           #not required #default: 1.0
  MK: 1.0                           #not required #default: 1.0
  bootstrap:                        #not required #default: no bootstrap
    enabled: true                       #required within bootstrap
    n_samples: 1000                     #not required #default: 1000
  n_workers: 8                      #not required #default: all cores (capped 8)

task input (mode 2 - pre-made JSON, full control)
---
single_channel_fit:                 #required
  config_file: path/to/fit.json     #required for mode 2
  study_module: label               #not required #overrides the JSON value
  bmat_preview_only: false          #not required #default: false
'''

# Partial-wave label <-> integer L
_WAVE_LABEL_TO_L = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6}
_L_TO_WAVE_LABEL = {v: k for k, v in _WAVE_LABEL_TO_L.items()}

# Momentum vector for each PSQ label / integer -- mirrors GenericFitRunner
_PSQ_TO_MOMENTUM = {
    'PSQ0': [0, 0, 0], 0: [0, 0, 0],
    'PSQ1': [0, 0, 1], 1: [0, 0, 1],
    'PSQ2': [1, 1, 0], 2: [1, 1, 0],
    'PSQ3': [1, 1, 1], 3: [1, 1, 1],
    'PSQ4': [0, 0, 2], 4: [0, 0, 2],
}

_QC2_DIR = os.path.dirname(os.path.abspath(__file__))

# Parallel engine by default: it tabulates B(E) once and solves roots
# vectorized, which is what makes bootstraps and near-free waves tractable.
_DEFAULT_ENGINE = 'my_fit_model_par.py'


def _resolve_model_file(model_file):
    """Resolve the engine module path, allowing a bare filename inside QC2."""
    if not model_file:
        model_file = _DEFAULT_ENGINE
    if os.path.isabs(model_file):
        return model_file
    local = os.path.join(_QC2_DIR, os.path.basename(model_file))
    if os.path.isfile(local):
        return local
    return os.path.abspath(model_file)


def _parse_particle_base(part_str):
    """Extract base particle name from strings like 'N(0)_ref' -> 'N'."""
    name = part_str.strip()
    for sep in ('(', '_'):
        if sep in name:
            name = name.split(sep)[0]
            break
    return name


def _infer_channel_qn(channel_str, task_params):
    """Infer a quantum_numbers channel dict from a scattering channel string.

    Particle spins come from ``general.particles.particles``. ``L`` comes from
    ``task_params['L']`` (default 0 = S-wave) and ``twoJ`` defaults to the
    minimum ``|2L - twoS|``. Returns ``None`` if any particle is unknown, in
    which case the caller should fall back to an explicit channel list.
    """
    task_params = task_params or {}
    try:
        from general.particles import particles as PARTICLES
    except ImportError:
        logging.warning("HPWFitTask: could not import general.particles - quantum numbers not auto-inferred.")
        return None

    particle_bases = [_parse_particle_base(p) for p in channel_str.split(',')]

    spins = []
    for pname in particle_bases:
        if pname not in PARTICLES:
            logging.warning(f"HPWFitTask: unknown particle '{pname}' - quantum numbers not auto-inferred.")
            return None
        spins.append(float(PARTICLES[pname]['spin']))

    if len(spins) != 2:
        logging.warning(
            f"HPWFitTask: channel '{channel_str}' does not name exactly two particles "
            "- quantum numbers not auto-inferred."
        )
        return None

    twoS_max = int(round(2 * (spins[0] + spins[1])))
    twoS_min = int(round(2 * abs(spins[0] - spins[1])))

    twoS = int(task_params['twoS']) if 'twoS' in task_params else None
    if twoS is None:
        twoS = twoS_min
        if twoS_min != twoS_max:
            logging.warning(
                f"HPWFitTask: channel '{channel_str}' admits twoS = {twoS_min}..{twoS_max} "
                f"(e.g. NN spin singlet and triplet). Defaulting to twoS={twoS}. "
                "Set 'twoS' in the task params to choose the other."
            )

    # L: accept integer or spectroscopic letter ('S','P','D',...)
    L_raw = task_params.get('L', 0)
    L = _WAVE_LABEL_TO_L.get(L_raw.upper(), 0) if isinstance(L_raw, str) else int(L_raw)

    twoJ = int(task_params['twoJ']) if 'twoJ' in task_params else abs(2 * L - twoS)

    # Spectroscopic label (2S+1) L_J -- J, not 2J. Half-integer J (which cannot
    # occur for two identical-spin baryons but can in general) keeps the "/2".
    _J = f"{twoJ // 2}" if twoJ % 2 == 0 else f"{twoJ}/2"
    wave_str = f"{twoS + 1}{_L_TO_WAVE_LABEL.get(L, str(L))}{_J}"
    channel_name = '_'.join(particle_bases) + f'_{wave_str}'

    params = task_params.get('params', [])
    if not params:
        logging.warning(
            f"HPWFitTask: no 'params' (K-matrix initial guesses) given for channel "
            f"'{channel_str}'. Add them to the task params, e.g.:\n"
            "  params:\n    - {name: c0, initial: -0.2}"
        )

    qn_channel = {
        "name":     channel_name,
        "twoJ":     twoJ,
        "L":        L,
        "Lp":       L,
        "twoS":     twoS,
        "twoSp":    twoS,
        "chan":     0,
        "chanp":    0,
        "enabled":  True,
        "k_matrix": task_params.get('k_matrix', 'polynomial'),
    }
    if params:
        qn_channel["params"] = params

    logging.info(
        f"HPWFitTask: inferred quantum numbers for '{channel_str}': "
        f"twoS={twoS}, L={L} ({_L_TO_WAVE_LABEL.get(L, '?')}-wave), twoJ={twoJ} -> {wave_str}"
    )
    return qn_channel


def _find_hpw_hdf5(task_params, proj_handler):
    """Return the HDF5 input path.

    Priority:
      1. Explicit ``data_file`` in task_params.
      2. Most-recently-modified .hdf5/.h5 in an upstream task's data directory.
      3. Most-recently-modified .hdf5/.h5 anywhere under the project root.
    """
    if task_params and task_params.get('data_file'):
        return task_params['data_file']

    for task_name in ('fit_spectrum', 'single_channel_fit'):
        handler = proj_handler.all_tasks.get(task_name)
        if not handler:
            continue
        search_root = handler.data_dir()
        candidates = (
            glob.glob(os.path.join(search_root, '**', '*.hdf5'), recursive=True)
            + glob.glob(os.path.join(search_root, '**', '*.h5'), recursive=True)
        )
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            logging.info(f"HPWFitTask: auto-discovered HDF5 from '{task_name}': {candidates[0]}")
            return candidates[0]

    candidates = (
        glob.glob(os.path.join(proj_handler.root, '**', '*.hdf5'), recursive=True)
        + glob.glob(os.path.join(proj_handler.root, '**', '*.h5'), recursive=True)
    )
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        logging.info(f"HPWFitTask: auto-discovered HDF5 from project root: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        "HPWFitTask: could not find an HDF5 input file. Add 'data_file' to the task "
        "params, or ensure an upstream task (fit_spectrum) wrote an HDF5 file into "
        "the project data directory."
    )


def _build_data_block(scattering):
    """Convert the YAML 'scattering' dict into a GenericFitRunner 'data' list.

    Returns ``[{"momentum": [0,0,0], "irreps": {"T1g": [0,1]}}, ...]``, one
    entry per PSQ with irreps merged across channels, or ``None`` when no
    ``scattering`` block was given (which tells the runner to auto-detect
    every dataset in the HDF5 file).
    """
    if not scattering:
        return None

    psq_merged = {}  # PSQ label -> {irrep: [levels]}
    for _channel, psq_dict in scattering.items():
        for psq_label, irrep_list in psq_dict.items():
            merged = psq_merged.setdefault(psq_label, {})
            # irrep_list is a list of single-key dicts: [{IrrepName: [levels]}]
            for irrep_entry in irrep_list:
                if not isinstance(irrep_entry, dict):
                    logging.warning(
                        f"HPWFitTask: malformed irrep entry under {psq_label}: {irrep_entry!r} "
                        "(expected 'IrrepName: [levels]'), skipping."
                    )
                    continue
                for irrep_name, levels in irrep_entry.items():
                    seen = merged.setdefault(irrep_name, [])
                    for lvl in levels:
                        if lvl not in seen:
                            seen.append(lvl)

    data_block = []
    for psq_label in sorted(psq_merged.keys(), key=str):
        momentum = _PSQ_TO_MOMENTUM.get(psq_label)
        if momentum is None:
            logging.warning(f"HPWFitTask: unknown PSQ label '{psq_label}', skipping.")
            continue
        data_block.append({
            "momentum": momentum,
            "irreps": {k: sorted(v) for k, v in psq_merged[psq_label].items()},
        })

    return data_block


def _build_hpw_config(task_params, proj_handler, general_configs):
    """Build a complete GenericFitRunner config dict from PyCALQ task params."""
    task_params = task_params or {}

    h5_file = _find_hpw_hdf5(task_params, proj_handler)

    # Lattice spatial extent: prefer the sigmond ensemble record, fall back to
    # an explicit task param.
    lattice_size = int(task_params.get('lattice_size', 48))
    try:
        import fvspectrum.sigmond_util as sigmond_util
        ensemble_info = sigmond_util.get_ensemble_info(general_configs)
        lattice_size = int(ensemble_info.getLatticeXExtent())
    except Exception:
        logging.info(f"HPWFitTask: using lattice_size={lattice_size} (no ensemble record).")

    scattering = task_params.get('scattering')
    data_block = _build_data_block(scattering)

    # Quantum numbers: an explicit block always wins; otherwise infer one
    # channel per scattering channel from general/particles.py.
    qn_block = task_params.get('quantum_numbers')
    if qn_block is None:
        inferred = []
        for channel_str in (scattering or {}):
            ch = _infer_channel_qn(channel_str, task_params)
            if ch:
                inferred.append(ch)
        qn_block = {"channels": inferred} if inferred else {}
        if not inferred:
            logging.warning(
                "HPWFitTask: no quantum numbers given or inferred. Provide a "
                "'quantum_numbers' block, or a 'scattering' block whose particles "
                "are known to general/particles.py."
            )

    plot_dir = proj_handler.plot_dir()
    log_dir = proj_handler.log_dir()

    fit_observable = task_params.get('fit_observable', 'shift')
    fit_block = {
        "MN":                float(task_params.get('MN', 1.0)),
        "MK":                float(task_params.get('MK', 1.0)),
        "strategy":          task_params.get('strategy', 'least_squares'),
        "n_starts":          int(task_params.get('n_starts', 20)),
        "n_refine":          int(task_params.get('n_refine', 400)),
        "n_maxiter":         int(task_params.get('n_maxiter', 10000)),
        "xatol":             float(task_params.get('xatol', 1e-6)),
        "fatol":             float(task_params.get('fatol', 1e-6)),
        "step_mode":         task_params.get('step_mode', 'adaptive'),
        "fit_mode":          task_params.get('fit_mode', 'omega'),
        "fit_observable":    fit_observable,
        # Residuals and covariance must use the same observable convention;
        # mixing them (p2 residuals against shift covariance) drifts results.
        "fit_cov_from":      task_params.get('fit_cov_from', fit_observable),
        "compute_vij":       bool(task_params.get('compute_vij', True)),
        "auto_p0":           bool(task_params.get('auto_p0', False)),
        "skip_minimization": bool(task_params.get('skip_minimization', False)),
    }
    for optional in ('param_bounds', 'max_iter', 'bootstrap', 'mN_err',
                     'initial_params', 'channel_initial_params'):
        if optional in task_params:
            fit_block[optional] = task_params[optional]

    # 'study_module' is only a label here (model_file decides the engine), but
    # it names the per-fit plot subdirectory -- so prefer the channel name.
    study_label = task_params.get('study_module')
    if not study_label:
        channels = qn_block.get('channels') or []
        study_label = channels[0].get('name') if channels else 'qc2_fit'
        study_label = str(study_label).replace('/', '_')

    config = {
        "study_module": study_label,
        "model_file":   _resolve_model_file(task_params.get('model_file')),
        "input": {
            "file":         h5_file,
            "lattice_size": lattice_size,
            "continuum":    bool(task_params.get('continuum', True)),
            "cutoff":       float(task_params.get('cutoff', 10.0)),
        },
        "quantum_numbers": qn_block,
        "fit": fit_block,
        "plot": {
            "enabled":           True,
            "save_path":         os.path.join(plot_dir, "hpw_fit_spectrum.pdf"),
            "save_timestamp":    True,
            "com_frame":         True,
            "show":              False,
            "ylabel":            r"$E^\star / m_N$",
            "show_ni":           True,
            "show_level_labels": True,
            "figures_dir":       plot_dir,
            "bmatrix_preview": {
                "enabled":      bool(task_params.get('bmat_preview_only', False)),
                "save_path":    os.path.join(plot_dir, "bmatrix_preview.pdf"),
                "n_sweep":      int(task_params.get('bmat_n_sweep', 800)),
                "clip":         float(task_params.get('bmat_clip', 30.0)),
                "y_scale_mode": task_params.get('bmat_y_scale_mode', 'data_central'),
                "show":         False,
            },
        },
        "logging": {
            "enabled": True,
            "dir":     log_dir,
            "prefix":  "hpw_fit",
            "level":   "INFO",
        },
    }

    if data_block:
        config["data"] = data_block

    return config


class HPWFitTask:
    """Adapts GenericFitRunner to the PyCALQ task interface.

    Mode 1 (``config_file``): the JSON is used as-is, giving access to every
    fitter flag documented in the upstream FITTING_GUIDE.

    Mode 2 (auto-config): a JSON config is generated from the task YAML plus
    the project directory, and written to the task log directory so the exact
    inputs of a run stay reproducible.
    """

    @property
    def info(self):
        return doc

    def __init__(self, task_name, proj_handler, general_configs, task_params):
        self.task_name = task_name
        self.proj_handler = proj_handler
        task_params = task_params or {}

        if 'config_file' in task_params:
            config_path = task_params['config_file']
            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"HPWFitTask: config_file not found: {config_path}")
            logging.info(f"HPWFitTask: using provided config_file: {config_path}")
        else:
            logging.info("HPWFitTask: no config_file provided - auto-generating JSON config.")
            config_dict = _build_hpw_config(task_params, proj_handler, general_configs)
            config_path = os.path.join(proj_handler.log_dir(), 'hpw_fit_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            logging.info(f"HPWFitTask: auto-generated config written to {config_path}")

        # Number of worker processes for the parallel engine, read from the
        # environment by my_fit_model_par at pool construction time.
        if 'n_workers' in task_params:
            os.environ['HPW_FIT_WORKERS'] = str(int(task_params['n_workers']))

        self._runner = GenericFitRunner(
            config_path,
            study_module_override=task_params.get('study_module'),
            bmat_preview_only=bool(task_params.get('bmat_preview_only', False)),
        )

    def run(self):
        """Run the fit. GenericFitRunner writes its own plots and logs."""
        self._runner.run()

    def plot(self):
        """No-op: plotting happens inside run(), driven by the config's plot block."""
        pass
