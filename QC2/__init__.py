"""QC2 - Luscher quantization-condition fits of finite-volume spectra.

Relates lattice finite-volume energy levels to infinite-volume scattering
observables (phase shifts, mixing angles, poles) via the Luscher quantization
condition, using the TwoHadronsInBox B-matrix through the ``BMat`` bindings.

Packaging note
--------------
The engine modules in this package import one another by bare module name
(``import plotting``, ``import my_fit_model``, ``import fit_par_worker``).
That is deliberate and load-bearing, not an oversight:

* ``fit_par_worker`` must be importable *as a top-level module* in spawned
  child processes, because ``multiprocessing`` pickles pool initializer and
  task callables by module-qualified reference.
* ``run_HPW_fit`` loads alternative model modules ("plugins") by file path via
  ``importlib.util.spec_from_file_location``, so those modules resolve their
  own imports against ``sys.path`` rather than against this package.

So this package prepends its own directory to ``sys.path`` on import. That
keeps ``import QC2.run_HPW_fit`` (the PyCALQ task path) and ``python
QC2/run_HPW_fit.py config.json`` (the standalone CLI path) both working, with
no edits to the engine's internal import structure.
"""

import os
import sys

_QC2_DIR = os.path.dirname(os.path.abspath(__file__))

if _QC2_DIR not in sys.path:
    sys.path.insert(0, _QC2_DIR)
