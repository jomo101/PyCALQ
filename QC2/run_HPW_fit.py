import argparse
import importlib
import importlib.util
import json
import logging
import math
import os
import sys
from datetime import datetime

import h5py
import matplotlib
import numpy as np

import plotting as pt
import plotting as _pt

# Default to headless plotting for CLI fits so repeated minimizations do not
# spawn/accumulate GUI-backed Python app processes on macOS.
if os.environ.get("HPW_HEADLESS", "1").strip().lower() not in ("0", "false", "no", "off"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Skip TeX plot styling when this module is re-imported inside a spawned fit
# worker process (matplotlib LaTeX probing is slow and races across workers).
if os.environ.get("HPW_FIT_WORKER") != "1":
    pt.apply_plot_style(use_tex=True)


class GenericFitRunner:
    def __init__(self, config_path, study_module_override=None, bmat_preview_only=False):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.study_module_name = study_module_override or self.config.get("study_module","fit")
        # if not self.study_module_name:
        #     raise KeyError(
        #         "Missing required study module name. Provide 'study_module' in config or pass --module."
        #     )
        self.study = self._load_study_module(self.study_module_name)
        self.log_path = None
        self.run_timestamp = None
        self.logger = logging.getLogger(f"run_hpw_fit.{self.study_module_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logging_enabled = False
        self.bmat_preview_only = bmat_preview_only
        # Populated by _export_results() at the end of a fit.
        self.results = None

    def _export_results(self, par, labels, chi2_val, dof, converged,
                        num_data_points, num_parameters, parametrization_name,
                        vijmat=None):
        """Record the fit results on the runner, and write them as JSON if asked.

        Downstream tooling (the campaign report builders, the PyCALQ task
        wrapper) reads this instead of re-parsing the run log. Set the output
        path with config["output"]["results_json"].
        """
        results = {
            "study_module":    self.study_module_name,
            "timestamp":       self.run_timestamp,
            "parametrization": parametrization_name,
            "converged":       bool(converged),
            "chi2":            float(chi2_val),
            "dof":             int(dof),
            "chi2_per_dof":    (float(chi2_val) / dof) if dof else None,
            "aic":             float(chi2_val - 2 * dof),
            "num_data_points": int(num_data_points),
            "num_parameters":  int(num_parameters),
            "parameters":      [{"name": str(n), "value": float(v)}
                                for n, v in zip(labels, par)],
        }

        if vijmat is not None:
            vij = np.asarray(vijmat, dtype=float)
            errors = np.sqrt(np.diag(vij))
            for entry, err in zip(results["parameters"], errors):
                entry["error"] = float(err)
            results["covariance"] = [[float(x) for x in row] for row in vij]

        self.results = results

        out_path = self.config.get("output", {}).get("results_json")
        if not out_path:
            return results

        try:
            out_path = self._format_path_template(out_path)
            out_dir = os.path.dirname(os.path.abspath(out_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)
            self._log(logging.INFO, "Wrote fit results: %s", out_path)
        except Exception as exc:
            self._log(logging.WARNING, "Could not write results JSON: %s", exc)

        return results

    def _format_path_template(self, template):
        """Expand the {study_module} style placeholders used by plot save paths."""
        try:
            return template.format(
                study_module=self.study_module_name,
                study_model=self.study_module_name,
                module=self.study_module_name,
            )
        except (KeyError, IndexError, ValueError):
            return template

    def _load_study_module(self, module_name):
        model_file = self.config.get("model_file")
        if model_file is None or not str(model_file).strip():
            return importlib.import_module(module_name)

        model_path = os.path.abspath(model_file)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"model_file not found: {model_path}")

        module_basename = os.path.splitext(os.path.basename(model_path))[0]
        dynamic_name = f"hpw_model_{module_basename}"
        spec = importlib.util.spec_from_file_location(dynamic_name, model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for model_file: {model_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _validate_study_interface(self):
        required_symbols = ["minimizechi2", "vij", "build_energy_cm_dict"]
        missing = [name for name in required_symbols if not hasattr(self.study, name)]
        if missing:
            raise AttributeError(
                f"Loaded study module is missing required functions: {', '.join(missing)}"
            )

    def _log(self, level, message, *args):
        if self.logging_enabled:
            self.logger.log(level, message, *args)
            return
        text = message % args if args else message
        print(text, flush=True)

    def _start_logging_if_enabled(self):
        if not self.run_timestamp:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logging_cfg = self.config.get("logging", {})
        log_enabled = logging_cfg.get("enabled", False)
        if not log_enabled:
            self.logging_enabled = False
            return

        log_dir = logging_cfg.get("dir", "./logs/1p1")
        prefix = logging_cfg.get("prefix", self.study_module_name.replace("Lüscher_", ""))
        os.makedirs(log_dir, exist_ok=True)

        timestamp = self.run_timestamp
        self.log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
        log_level_name = str(logging_cfg.get("level", "INFO")).upper()
        log_level = getattr(logging, log_level_name, logging.INFO)

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.setLevel(log_level)
        self.logging_enabled = True

        self.logger.info("%s", "=" * 80)
        self.logger.info("Run started")
        self.logger.info("Config file: %s", self.config_path)
        self.logger.info("Log file: %s", self.log_path)
        self.logger.info("Study module: %s", self.study_module_name)
        self.logger.info("%s", "=" * 80)

    def _assign_momentum_to_nP(self, nP):
        if nP == 0:
            mom = np.array([0, 0, 0])
        elif nP == 1:
            mom = np.array([0, 0, 1])
        elif nP == 2:
            mom = np.array([1, 1, 0])    
        elif nP == 3:
            mom = np.array([1, 1, 1])
        elif nP == 4:
            mom = np.array([0, 0, 2])
        return mom
    
    def _flatten_data(self):
        entries = []

        # Lazy-load HDF5 keys once if any irrep uses the "all" codeword.
        _h5_keys = None
        def _get_h5_keys():
            nonlocal _h5_keys
            if _h5_keys is None:
                fileall = self.config["input"].get("file", "./Data/nn_iso_singlet_fullspectrum.h5")
                with h5py.File(fileall, "r") as _f:
                    _h5_keys = set(_f.keys())
            return _h5_keys

        for block in self.config["data"]:
            momentum = np.array(block["momentum"], dtype=int)
            psq = int(np.sum(momentum**2))
            for irrep, level_list in block["irreps"].items():
                # Accept "all" as either the whole value or as an element inside [].
                use_all = (level_list == "all") or (
                    isinstance(level_list, list) and "all" in level_list
                )
                if use_all:
                    prefix = f"{irrep}_Psq{psq}_level"
                    all_levels = sorted(
                        int(k[len(prefix):])
                        for k in _get_h5_keys()
                        if k.startswith(prefix) and k[len(prefix):].isdigit()
                    )
                    level_list = all_levels
                for lvl in level_list:
                    entries.append(
                        {
                            "level": int(lvl),
                            "nP": psq,
                            "irrep": irrep,
                            "momentum": momentum.copy(),
                        }
                    )

        return entries

    def _apply_quantum_numbers(self):
        quantum_numbers = self.config.get("quantum_numbers", {})
        if not quantum_numbers:
            self._log(logging.INFO, "Quantum numbers: not provided in config")
            return

        self._log(logging.INFO, "Quantum numbers from config:")
        channels = quantum_numbers.get("channels", quantum_numbers.get("sets", []))
        if isinstance(channels, list) and channels:
            # Count enabled channels only
            enabled_channels = [ch for ch in channels if bool(ch.get("enabled", True))]
            self._log(logging.INFO, "  channels (%d):", len(enabled_channels))
            
            enabled_idx = 0
            for idx, channel in enumerate(channels):
                if not isinstance(channel, dict):
                    continue

                channel_name = str(channel.get("name", f"channel_{idx}"))
                enabled = bool(channel.get("enabled", True))
                
                # Only log enabled channels
                if not enabled:
                    continue
                
                k_matrix = channel.get("k_matrix", "<not provided>")
                params = channel.get("params", channel.get("terms", channel.get("calc_terms", [])))

                self._log(logging.INFO, "    - channel[%d]: %s (enabled=%s)", enabled_idx, channel_name, enabled)
                self._log(logging.INFO, "      k_matrix: %s", k_matrix)
                self._log(logging.INFO, "      params:")

                if isinstance(params, list) and params:
                    for param_idx, param in enumerate(params):
                        if isinstance(param, dict):
                            pname = str(param.get("name", f"p{param_idx}"))
                            if "initial" in param:
                                self._log(logging.INFO, "        - %s = %s", pname, param["initial"])
                            else:
                                self._log(logging.INFO, "        - %s", pname)
                        else:
                            self._log(logging.INFO, "        - p%d = %s", param_idx, param)
                else:
                    self._log(logging.INFO, "        - <none>")
                
                enabled_idx += 1
        else:
            self._log(logging.INFO, "  channels: <not provided>")

        for key, value in quantum_numbers.items():
            if key in {"channels", "sets"}:
                continue
            self._log(logging.INFO, "  %s: %s", key, value)

        if hasattr(self.study, "set_quantum_numbers"):
            self.study.set_quantum_numbers(quantum_numbers)
            self._log(logging.INFO, "Applied quantum numbers via %s.set_quantum_numbers(...)", self.study_module_name)
        else:
            self._log(
                logging.WARNING,
                "Note: %s has no set_quantum_numbers(...) hook. "
                "If you want runtime control of 2J/L/L'/S/etc (or branch control for coupled channels), "
                "add that function in your study module.",
                self.study_module_name,
            )

    def _setup_study_progress_logging(self):
        if hasattr(self.study, "set_progress_logger"):
            self.study.set_progress_logger(lambda message: self._log(logging.INFO, "%s", message))
            self._log(logging.INFO, "Enabled study progress logging hook")

    def _resolve_initial_params(self, fit_cfg):
        channel_params = fit_cfg.get("channel_initial_params")
        if channel_params is None:
            quantum_numbers = self.config.get("quantum_numbers", {})
            channels = quantum_numbers.get("channels", quantum_numbers.get("sets", []))
            if channels:
                flat_params = []
                all_found = True
                for idx, channel in enumerate(channels):
                    if not bool(channel.get("enabled", True)):
                        continue
                    # Skip zero k_matrix channels (no params to fit)
                    k_mode = str(channel.get("k_matrix", "polynomial")).strip().lower()
                    if k_mode == "zero":
                        continue
                    params = channel.get("params", channel.get("terms", channel.get("calc_terms", [])))
                    if not isinstance(params, list) or not params:
                        all_found = False
                        break

                    channel_name = str(channel.get("name", f"channel_{idx}"))
                    # Handle nested lists (grouped params for coupled channels)
                    if isinstance(params[0], list):
                        for group in params:
                            for param_idx, param in enumerate(group):
                                if "initial" not in param:
                                    raise ValueError(
                                        f"Missing initial value for {channel_name}.params group"
                                    )
                                flat_params.append(float(param["initial"]))
                    else:
                        for param_idx, param in enumerate(params):
                            if "initial" not in param:
                                raise ValueError(
                                    f"Missing initial value for {channel_name}.params[{param_idx}]"
                                )
                            flat_params.append(float(param["initial"]))

                if all_found and flat_params:
                    return flat_params

            return fit_cfg.get("initial_params", [-0.2, 0.7])

        quantum_numbers = self.config.get("quantum_numbers", {})
        channels = quantum_numbers.get("channels", quantum_numbers.get("sets", []))
        if not channels:
            raise ValueError(
                "fit.channel_initial_params was provided, but quantum_numbers.channels is missing or empty."
            )

        params_by_name = {}
        if isinstance(channel_params, dict):
            params_by_name = channel_params
        elif isinstance(channel_params, list):
            for idx, entry in enumerate(channel_params):
                if not isinstance(entry, dict) or "name" not in entry or "params" not in entry:
                    raise ValueError(
                        f"Invalid fit.channel_initial_params[{idx}]. Expected object with name and params."
                    )
                params_by_name[str(entry["name"])] = entry["params"]
        else:
            raise TypeError("fit.channel_initial_params must be a dict or list of {name, params}.")

        flat_params = []
        for idx, channel in enumerate(channels):
            if not bool(channel.get("enabled", True)):
                continue
            name = str(channel.get("name", f"channel_{idx}"))
            params = channel.get("params", channel.get("terms", channel.get("calc_terms", [])))
            if isinstance(params, list) and params:
                n_params = len(params)
            else:
                n_params = int(channel.get("n_params", 0))
            if name not in params_by_name:
                raise ValueError(f"Missing initial params for enabled channel '{name}'.")

            channel_p0 = list(params_by_name[name])
            if len(channel_p0) != n_params:
                raise ValueError(
                    f"Channel '{name}' expects n_params={n_params}, but got {len(channel_p0)} initial params."
                )
            flat_params.extend(channel_p0)

        return flat_params

    def _resolve_param_labels(self, fit_cfg, n_params):
        quantum_numbers = self.config.get("quantum_numbers", {})
        channels = quantum_numbers.get("channels", quantum_numbers.get("sets", []))
        global_params = quantum_numbers.get("params", quantum_numbers.get("calc_terms", []))

        labels = []
        if channels:
            for idx, channel in enumerate(channels):
                if not bool(channel.get("enabled", True)):
                    continue

                channel_name = str(channel.get("name", f"channel_{idx}"))
                channel_params = channel.get("params", channel.get("terms", channel.get("calc_terms")))
                # Skip zero k_matrix channels
                k_mode = str(channel.get("k_matrix", "polynomial")).strip().lower()
                if k_mode == "zero":
                    continue
                # Handle nested lists (grouped params for coupled channels)
                if isinstance(channel_params, list) and channel_params and isinstance(channel_params[0], list):
                    flat_cp = []
                    for group in channel_params:
                        flat_cp.extend(group)
                    n_channel_params = len(flat_cp)
                    params = flat_cp
                elif isinstance(channel_params, list) and channel_params:
                    n_channel_params = len(channel_params)
                    params = channel_params
                else:
                    n_channel_params = int(channel.get("n_params", 0))
                    params = global_params

                if isinstance(params, list) and len(params) == n_channel_params:
                    for param_idx, param in enumerate(params):
                        param_name = str(param.get("name", f"p{param_idx}"))
                        labels.append(f"{channel_name}.{param_name}")
                else:
                    for param_idx in range(n_channel_params):
                        labels.append(f"{channel_name}.p{param_idx}")

        if not labels:
            if isinstance(global_params, list) and len(global_params) == n_params:
                labels = [str(param.get("name", f"p{i}")) for i, param in enumerate(global_params)]
            else:
                labels = [f"p{i}" for i in range(n_params)]

        if len(labels) != n_params:
            labels = [f"p{i}" for i in range(n_params)]

        return labels

    def _resolve_parametrization_name(self):
        # First, try to build from JSON config channel k_matrix modes
        quantum_numbers = self.config.get("quantum_numbers", {})
        channels = quantum_numbers.get("channels", quantum_numbers.get("sets", []))
        if channels:
            parts = []
            for ch in channels:
                if not bool(ch.get("enabled", True)):
                    continue
                name = ch.get("name", "channel")
                mode = ch.get("k_matrix", "polynomial")
                if isinstance(mode, list):
                    mode = "coupled[" + ",".join(str(m) for m in mode) + "]"
                parts.append(f"{name}:{mode}")
            if parts:
                return " | ".join(parts)

        # Fallback: check module-level attributes
        for attr_name in ("k_matrix", "param_string", "PARAMETRIZATION", "PARAMETRIZATION_NAME", "model_name"):
            if hasattr(self.study, attr_name):
                value = getattr(self.study, attr_name)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        if hasattr(self.study, "MODEL_10"):
            model_obj = getattr(self.study, "MODEL_10")
            model_name = getattr(model_obj, "__name__", str(model_obj))
            if hasattr(self.study, "calcFunc_param"):
                return f"{model_name} (via calcFunc_param)"
            return str(model_name)

        if hasattr(self.study, "calcFunc_param"):
            return "calcFunc_param"

        return "<not provided by study module>"

    def _log_data_entries(self, title, entries):
        self._log(logging.INFO, "%s (%d):", title, len(entries))
        if not entries:
            self._log(logging.INFO, "  - none")
            return

        for idx, entry in enumerate(entries, start=1):
            self._log(
                logging.INFO,
                "  %02d) %s (PSq_%d): level_%d",
                idx,
                entry["irrep"],
                entry["nP"],
                entry["level"],
            )

    def _resolve_plot_save_path(self, plot_cfg):
        if not self.run_timestamp:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        template = plot_cfg.get("save_path")
        if not template:
            template = f"./Images/Spectrum/fit_{self.study_module_name}.pdf"

        if not isinstance(template, str):
            return template

        try:
            resolved_path = template.format(
                study_module=self.study_module_name,
                study_model=self.study_module_name,
                module=self.study_module_name,
                timestamp=self.run_timestamp,
            )

            root, ext = os.path.splitext(resolved_path)
            if self.run_timestamp not in os.path.basename(root):
                return f"{root}_{self.run_timestamp}{ext}"
            return resolved_path
        except (KeyError, IndexError, ValueError):
            self._log(
                logging.WARNING,
                "Could not format plot.save_path template '%s'; using literal path.",
                template,
            )
            return template
            
    def _apply_figures_dir(self, resolved_path, as_dir=False):
        """If figures_dir is configured, redirect the save path into the
        per-fit subfolder figures_dir/<study_module>/ (keeping the filename).
        Set "plot": {"figures_flat": true} to keep the old flat layout.
        With as_dir=True, return the per-fit directory itself."""
        figures_dir = getattr(self, '_figures_dir', None)
        if not figures_dir:
            return None if as_dir else resolved_path
        if getattr(self, '_figures_flat', False):
            sub = figures_dir
        else:
            sub = os.path.join(figures_dir, self.study_module_name)
        if as_dir:
            os.makedirs(sub, exist_ok=True)
            return sub
        if not resolved_path:
            return resolved_path
        fname = os.path.basename(resolved_path)
        if not fname:
            return resolved_path
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, fname)

    def _resolve_subplot_save_path(self, plot_cfg):
        """Resolve save path for level subplot comparison plots."""
        template = plot_cfg.get("subplot_save_path")
        if not template:
            template = f"./Images/Spectrum/level_subplots_{self.study_module_name}.pdf"

        if not isinstance(template, str):
            return template

        try:
            resolved_path = template.format(
                study_module=self.study_module_name,
                study_model=self.study_module_name,
                module=self.study_module_name,
            )
            return resolved_path
        except (KeyError, IndexError, ValueError):
            self._log(
                logging.WARNING,
                "Could not format plot.subplot_save_path template '%s'; using literal path.",
                template,
            )
            return template
            
    def _save_results_as_image(self, results_text, save_path):
        """
        Save the given results text as an image.

        Parameters:
            results_text (str): The text to save as an image.
            save_path (str): The path to save the image.
        """
        # Render WITHOUT TeX: the results text contains raw underscores
        # (e.g. 3S1_3D1_coupled.A_S) that make the latex subprocess hang or
        # fail, which blocked the whole run at the very end.
        try:
            with matplotlib.rc_context({"text.usetex": False}):
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.axis('off')  # Turn off the axes

                # Add the text to the figure with left alignment
                ax.text(0.05, 0.95, results_text, fontsize=11, ha='left', va='top',
                        transform=ax.transAxes, fontfamily='monospace')

                # Save the figure as an image
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
        except Exception as exc:
            print(f"WARNING: could not save results image ({exc})", flush=True)
            plt.close("all")

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

    @staticmethod #Python decorator that defines a method inside a class that doesn’t use self (instance) or cls (class).
    def _build_grouped_inputs(mom2, irreps):
        framesall = []
        irrepsall = []
        levelsall = []
        frame_to_idx = {}

        for mom, irrep_label in zip(mom2, irreps):
            key = tuple(np.asarray(mom).tolist())
            if key not in frame_to_idx:
                frame_to_idx[key] = len(framesall)
                framesall.append(np.array(mom))
                irrepsall.append([])
                levelsall.append([])

            frame_idx = frame_to_idx[key]
            if irrep_label in irrepsall[frame_idx]:
                irrep_idx = irrepsall[frame_idx].index(irrep_label)
                levelsall[frame_idx][irrep_idx] += 1
            else:
                irrepsall[frame_idx].append(irrep_label)
                levelsall[frame_idx].append(1)

        return framesall, irrepsall, levelsall

    def load_data(self):
        """
        Load and preprocess spectrum data from the HDF5 file described in the config.

        Returns
        -------
        dict with keys:
            data2, data2cm, datap2, covp2,
            kept_mom2, kept_irreps, kept_levels, kept_nPvec,
            massN, lattice_size, mass_n, mass_k,
            framesall, irrepsall, levelsall,
            energy_cm_data
        """
        input_cfg = self.config["input"]
        fileall   = input_cfg.get("file", "./Data/nn_iso_singlet_fullspectrum.h5")
        continuum  = input_cfg.get("continuum", True)
        lattice_size = int(input_cfg.get("lattice_size", 48))
        cutoff2    = float(input_cfg.get("cutoff", 10))

        def _auto_detect_data_entries(h5_file):
            entries = []
            for key in h5_file.keys():
                if key != 'mN':
                    parts = key.split("_")
                    irrep = parts[0]
                    mom2  = int(parts[1].replace("Psq", ""))
                    level = int(parts[-1].replace("level", ""))
                    entries.append({
                        "level": level,
                        "nP": mom2,
                        "irrep": irrep,
                        "momentum": self._assign_momentum_to_nP(mom2),
                    })
            return sorted(entries, key=lambda x: x["nP"])

        if not self.config.get("data"):
            self._log(logging.INFO, "No data in config; auto-detecting HDF5 datasets.")
            with h5py.File(fileall, "r") as h5_file:
                selection_entries = _auto_detect_data_entries(h5_file)
        else:
            selection_entries = self._flatten_data()

        self._log(logging.INFO, "Input file: %s", fileall)
        self._log(logging.INFO, "Input settings: continuum=%s, lattice_size=%d, cutoff=%s",
                  continuum, lattice_size, cutoff2)

        names = [
            f"{e['irrep']}_Psq{e['nP']}_level{e['level']}"
            for e in selection_entries
        ]

        with h5py.File(fileall, "r") as f2:
            massN = np.array(f2["mN/E0"])
            two_pi_over_L = 2 * np.pi / lattice_size

            if not names:
                raise ValueError(
                    "Config field 'data' is empty. Add at least one momentum/irrep/level entry."
                )

            sample_dataset = next(
                (f"{n}/E_N1" for n in names if f"{n}/E_N1" in f2), None
            )
            if sample_dataset is None:
                raise KeyError(
                    "None of the selected entries contain an E_N1 dataset in the input file."
                )

            n_bootstrap = len(np.array(f2[sample_dataset]))
            data2 = np.zeros((0, n_bootstrap))
            kept_mom2, kept_irreps, kept_levels, kept_nPvec = [], [], [], []
            rejected_entries = []

            for i, name in enumerate(names):
                required_keys = [f"{name}/dE_NN", f"{name}/Psq_N1", f"{name}/Psq_N2"]
                if not continuum:
                    required_keys.extend([f"{name}/E_N1", f"{name}/E_N2"])
                missing_key = next((k for k in required_keys if k not in f2), None)
                if missing_key is not None:
                    raise KeyError(
                        f"Missing HDF5 dataset key: {missing_key}. "
                        "Check config.data momentum/irrep/level entries against available datasets."
                    )

                if continuum:
                    shift = f2[name + "/dE_NN"]
                    E_N1  = np.sqrt(massN**2 + f2[name + "/Psq_N1"][0] * two_pi_over_L**2)
                    E_N2  = np.sqrt(massN**2 + f2[name + "/Psq_N2"][0] * two_pi_over_L**2)
                    auxread = (shift + E_N1 + E_N2) / massN
                else:
                    auxread = (
                        np.array(f2[name + "/E_N1"])
                        + np.array(f2[name + "/E_N2"])
                        + np.array(f2[name + "/dE_NN"])
                    ) / massN

                if np.mean(auxread) < cutoff2:
                    data2 = np.vstack((data2, auxread))
                    kept_mom2.append(selection_entries[i]["momentum"])
                    kept_irreps.append(selection_entries[i]["irrep"])
                    kept_levels.append(selection_entries[i]["level"])
                    kept_nPvec.append(selection_entries[i]["nP"])
                else:
                    rejected_entries.append(selection_entries[i])

        if rejected_entries:
            self._log_data_entries("Levels rejected by cutoff", rejected_entries)
        if not kept_nPvec:
            raise ValueError(
                "No selected levels passed the cutoff; adjust selection or cutoff in config."
            )

        data2cm = np.sqrt(
            data2**2 - np.outer(
                np.array(kept_nPvec),
                (2 * math.pi / lattice_size / massN) ** 2
            )
        )
        datap2 = data2cm**2 / 4 - 1.0
        covp2  = np.cov(datap2[:, 1:])   # col 0 is the mean; covariance from bootstrap samples only
        framesall, irrepsall, levelsall = self._build_grouped_inputs(kept_mom2, kept_irreps)
        energy_cm_data = self.study.build_energy_cm_dict(
            data2cm, kept_mom2, kept_irreps, kept_levels
        )

        fit_cfg = self.config.get("fit", {})
        mass_n = float(fit_cfg.get("MN", 1.0))
        mass_k = float(fit_cfg.get("MK", 1.0))

        return {
            "data2":          data2,
            "data2cm":        data2cm,
            "datap2":         datap2,
            "covp2":          covp2,
            "kept_mom2":      kept_mom2,
            "kept_irreps":    kept_irreps,
            "kept_levels":    kept_levels,
            "kept_nPvec":     kept_nPvec,
            "massN":          massN,
            "lattice_size":   lattice_size,
            "mass_n":         mass_n,
            "mass_k":         mass_k,
            "framesall":      framesall,
            "irrepsall":      irrepsall,
            "levelsall":      levelsall,
            "energy_cm_data": energy_cm_data,
        }

    def run(self):
        _run_start = datetime.now()
        print(f"\n{'='*60}")
        print(f"Run started : {_run_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Config      : {self.config_path}")
        print(f"{'='*60}\n")
        self._start_logging_if_enabled()
        self._validate_study_interface()
        self._log(logging.INFO, "Study module: %s", self.study_module_name)
        self._apply_quantum_numbers()
        self._setup_study_progress_logging()

        input_cfg = self.config["input"]
        fileall = input_cfg.get("file", "./Data/nn_iso_singlet_fullspectrum.h5")
        continuum = input_cfg.get("continuum", True)
        lattice_size = int(input_cfg.get("lattice_size", 48))
        cutoff2 = float(input_cfg.get("cutoff", 10))
        # Automatically detect all available datasets in the HDF5 file
        def _auto_detect_data_entries(h5_file):
            entries = []
            for key in h5_file.keys():
                if key != 'mN':
                    parts = key.split("_")
                    irrep = parts[0]
                    mom2  = int(parts[1].replace("Psq", ""))
                    level = int(parts[-1].replace("level", ""))
                    entries.append({
                        "level": level,
                        "nP": mom2,
                        "irrep": irrep,
                        "momentum": self._assign_momentum_to_nP(mom2),
                    })
            entries = sorted(entries, key=lambda x: x["nP"])
            return entries

        # Use the auto-detected entries if no data is provided in the config
        if not self.config.get("data"):
            self._log(logging.INFO, "No data provided in config; auto-detecting datasets in HDF5 file.")
            with h5py.File(fileall, "r") as h5_file:
                selection_entries = _auto_detect_data_entries(h5_file)
                self._log_data_entries("Auto-detected data entries", selection_entries)
                # self._log(logging.INFO, "Auto-detected data entries", selection_entries)
        else:
            selection_entries = self._flatten_data()

        self._log(logging.INFO, "Input file: %s", fileall)
        self._log(logging.INFO, "Input settings: continuum=%s, lattice_size=%d, cutoff=%s", continuum, lattice_size, cutoff2)
        self._log_data_entries("Requested fit levels from config", selection_entries)

        names = [
            f"{entry['irrep']}_Psq{entry['nP']}_level{entry['level']}"
            for entry in selection_entries
        ]
        
        with h5py.File(fileall, "r") as f2:
            massN = np.array(f2["mN/E0"])
            two_pi_over_L = 2 * np.pi / lattice_size

            if not names:
                self._log(logging.CRITICAL, "No data entries provided in config['data']; cannot continue fit.")
                raise ValueError("Config field 'data' is empty. Add at least one momentum/irrep/level entry.")

            sample_dataset = None
            for name in names:
                candidate = f"{name}/E_N1"
                if candidate in f2:
                    sample_dataset = candidate
                    break

            if sample_dataset is None:
                self._log(logging.CRITICAL, "Could not find any E_N1 dataset for selected data entries; cannot continue fit.")
                raise KeyError("None of the selected entries contain an E_N1 dataset in the input file.")


            n_bootstrap = len(np.array(f2[sample_dataset]))
            print(f"[run_HPW_fit] Number of bootstrap samples: {n_bootstrap}")

            data2 = np.zeros((0, n_bootstrap))
            datashifts = 0.0 * data2
            kept_mom2 = []
            kept_irreps = []
            kept_levels = []
            kept_nPvec = []
            kept_entries = []
            rejected_entries = []
            data_shifts_raw = np.zeros((0, n_bootstrap))  # raw dE_NN in lattice units
            ni_sum_central  = []                           # E_N1_c + E_N2_c per level

            for i, name in enumerate(names):
                #self._log(logging.INFO, "read now %s", name)
                required_keys = [f"{name}/dE_NN", f"{name}/Psq_N1", f"{name}/Psq_N2"]
                if not continuum:
                    required_keys.extend([f"{name}/E_N1", f"{name}/E_N2"])

                missing_key = next((key for key in required_keys if key not in f2), None)
                if missing_key is not None:
                    self._log(logging.CRITICAL, "Missing HDF5 dataset key: %s", missing_key)
                    raise KeyError(
                        f"Missing HDF5 dataset key: {missing_key}. "
                        "Check config.data momentum/irrep/level entries against available datasets."
                    )

                if continuum:
                    shift = np.array(f2[name + "/dE_NN"])
                    E_N1 = np.sqrt(massN**2 + f2[name + "/Psq_N1"][0] * (two_pi_over_L) ** 2)
                    E_N2 = np.sqrt(massN**2 + f2[name + "/Psq_N2"][0] * (two_pi_over_L) ** 2)
                    auxread = (shift + E_N1 + E_N2) / massN
                    _ni_c = float(E_N1[0] + E_N2[0])   # central single-particle sum (lattice units)
                else:
                    shift   = np.array(f2[name + "/dE_NN"])
                    E_N1_arr = np.array(f2[name + "/E_N1"])
                    E_N2_arr = np.array(f2[name + "/E_N2"])
                    auxread  = (E_N1_arr + E_N2_arr + shift) / massN
                    _ni_c    = float(E_N1_arr[0] + E_N2_arr[0])

                auxread2 = shift / massN
                if np.mean(auxread) < cutoff2:
                    data2 = np.vstack((data2, auxread))
                    datashifts = np.vstack((datashifts, auxread2))
                    data_shifts_raw = np.vstack((data_shifts_raw, shift))
                    ni_sum_central.append(_ni_c)
                    kept_mom2.append(selection_entries[i]["momentum"])
                    kept_irreps.append(selection_entries[i]["irrep"])
                    kept_levels.append(selection_entries[i]["level"])
                    kept_nPvec.append(selection_entries[i]["nP"])
                    kept_entries.append(selection_entries[i])
                else:
                    rejected_entries.append(selection_entries[i])
            ni_sum_central = np.asarray(ni_sum_central, dtype=float)   # shape (n_levels,)

        if rejected_entries:
            self._log_data_entries("Levels rejected by cutoff", rejected_entries)

        if not kept_nPvec:
            self._log(logging.CRITICAL, "No selected levels passed the cutoff; cannot continue fit.")
            raise ValueError("No selected levels passed the cutoff; adjust selection or cutoff in config.")
        data_Elab_mref = data2
        data2cm = np.sqrt(data2**2 - np.outer(np.array(kept_nPvec), (2 * math.pi / lattice_size / massN) ** 2))
        datap2  = data2cm**2 / 4 - 1.0
        ni_sum_central = np.asarray(ni_sum_central, dtype=float)

        # ── Observable & covariance selection ─────────────────────────────────
        fit_cfg      = self.config["fit"]
        _fit_obs     = str(fit_cfg.get("fit_observable", "shift"))
        _fit_cov_src = str(fit_cfg.get("fit_cov_from",   "shift"))

        # Shift normalised by mN: dimensionless dE_NN/mN — same units as the
        # model output (epred = Ecm/mN, ni_sum/mN).  datashifts is already
        # computed as dE_NN/massN during data loading.
        _mN0 = float(massN[0])
        _Cov_shift_raw = np.cov(data_shifts_raw[:, 1:])            # raw lattice units²
        _Cov_shift_norm = _Cov_shift_raw / (_mN0 ** 2)             # normalised (dimensionless)

        if _fit_obs == "p2":
            _data_obs = datap2
        elif _fit_obs == "ecm":
            _data_obs = data2cm
        elif _fit_obs == "elab":
            # True lab-frame energy, normalised: E_lab/mN — exactly how the
            # levels are stored in the h5 (data2 = E_lab/mN per sample).
            # The model converts its E_cm/mN roots via
            # E_lab/mN = sqrt(ecm^2 + (P/mN)^2); the per-level lab boost
            # (P/mN)^2 rides in the ni_sum slot (see _to_observable).
            _data_obs = data2
            ni_sum_central = (np.asarray(kept_nPvec, dtype=float)
                              * (2 * math.pi / lattice_size / _mN0) ** 2)
        elif _fit_obs == "shift":
            # Normalised lab-frame shift dE_NN/mN. dE_NN and the NI sum are
            # lab-frame quantities, so the model must boost its cm root:
            #   shift/mN = sqrt((Ecm/mN)^2 + (P/mN)^2) - ni_sum_lab/mN.
            # ni_sum carries both rows: [ni_sum_lab/mN, (P/mN)^2]
            # (see _to_observable; a 1-D ni_sum falls back to the old
            # rest-frame-only formula Ecm/mN - ni_sum/mN).
            _data_obs = datashifts
            _psq_norm = (np.asarray(kept_nPvec, dtype=float)
                         * (2 * math.pi / lattice_size / _mN0) ** 2)
            ni_sum_central = np.stack([ni_sum_central / _mN0, _psq_norm])
        else:
            raise ValueError(f"Unknown fit_observable '{_fit_obs}'. Use: p2 | ecm | elab | shift")

        if _fit_cov_src == "shift":
            _Ecm_c = data2cm[:, 0]
            if _fit_obs == "shift":
                _J = np.ones(len(_Ecm_c))                          # dX/d(shift/mN) = 1
                covp2 = _Cov_shift_norm
            elif _fit_obs == "elab":
                # shift is a lab-frame energy difference, so
                # d(E_lab/mN)/d(shift/mN) = 1: normalised shift covariance.
                covp2 = _Cov_shift_norm
            elif _fit_obs == "ecm":
                _J = np.ones(len(_Ecm_c)) / _mN0
                covp2 = np.outer(_J, _J) * _Cov_shift_raw
            else:  # p2
                _J = _Ecm_c / (2.0 * _mN0)
                covp2 = np.outer(_J, _J) * _Cov_shift_raw
        else:
            covp2 = np.cov(_data_obs[:, 1:])

        self._log(logging.INFO, "fit_observable=%s  fit_cov_from=%s", _fit_obs, _fit_cov_src)

        framesall, irrepsall, levelsall = self._build_grouped_inputs(kept_mom2, kept_irreps)

        energy_cm_data = self.study.build_energy_cm_dict(data2cm, kept_mom2, kept_irreps, kept_levels)
        # print("Energy CM data:", energy_cm_data)
        # ref masses
        mass_n = float(fit_cfg.get("MN", 1.0))
        mass_k = float(fit_cfg.get("MK", 1.0))
        plot_cfg = self.config.get("plot", {})
        # ── figures_dir: single directory for all output figures ──────────────
        # Defaults to ./figures, so every run's figures land together in
        # ./figures/<study_module>/ (set figures_dir explicitly to group
        # campaigns, e.g. ./figures/SD_par; figures_dir: false disables).
        _fd = plot_cfg.get("figures_dir", "./figures")
        if isinstance(_fd, str) and _fd:
            try:
                _fd = _fd.format(
                    study_module=self.study_module_name,
                    study_model=self.study_module_name,
                    module=self.study_module_name,
                )
            except (KeyError, ValueError):
                _fd = _fd.replace("{study_module}", self.study_module_name)
            self._figures_dir = _fd.rstrip("/\\")
            os.makedirs(self._figures_dir, exist_ok=True)
        else:
            self._figures_dir = None
        self._figures_flat = bool(plot_cfg.get("figures_flat", False))
        if plot_cfg.get("enabled", True):
            if plot_cfg.get("save_timestamp", True):
                plot_save_path = self._resolve_plot_save_path(plot_cfg)
            else:
                plot_save_path = plot_cfg.get("save_path", f"./Images/Spectrum/fit_{self.study_module_name}.pdf")
            if "{study_module}" in plot_save_path:
                plot_save_path = plot_save_path.format(study_module=self.study_module_name)
            plot_save_path = self._apply_figures_dir(plot_save_path)
            if isinstance(plot_save_path, str):
                plot_dir = os.path.dirname(plot_save_path)
                if plot_dir:
                    os.makedirs(plot_dir, exist_ok=True)
            self._log(logging.INFO, "Plot save path: %s", plot_save_path)
            _figsize = plot_cfg.get("spectrum_figsize", None)
            if isinstance(_figsize, list) and len(_figsize) == 2:
                _figsize = tuple(_figsize)
            # y-axis framing: by default frame the DATA range so predicted/NI
            # levels above the top data point don't inflate the top margin.
            # Override with plot.spectrum_ylim [lo, hi], or tune the headroom
            # with plot.y_from_data / plot.y_top_pad / plot.y_bottom_pad.
            _spectrum_ylim = plot_cfg.get("spectrum_ylim", None)
            if isinstance(_spectrum_ylim, list) and len(_spectrum_ylim) == 2:
                _spectrum_ylim = tuple(_spectrum_ylim)
            else:
                _spectrum_ylim = None
            _pt.plot_energies_vs_irreps(
                energy_cm_data,
                massN,
                lattice_size,
                save_path=plot_save_path,
                C_o_M=plot_cfg.get("com_frame", True),
                ylabel=plot_cfg.get("ylabel", r"$E^\star / m_N $"),
                show=bool(plot_cfg.get("show", False)),
                show_ni=bool(plot_cfg.get("show_ni", True)),
                show_level_labels=bool(plot_cfg.get("show_level_labels", False)),
                figsize=_figsize,
                title=plot_cfg.get("title", None),
                ylim=_spectrum_ylim,
                y_from_data=bool(plot_cfg.get("y_from_data", True)),
                y_top_pad=float(plot_cfg.get("y_top_pad", 0.08)),
                y_bottom_pad=float(plot_cfg.get("y_bottom_pad", 0.05)),
            )
            # _ps.make_spectrum_plot(
            #     energy_cm_data,
            #     massN,
            #     lattice_size,
            #     save_path=plot_save_path,
            #     C_o_M=plot_cfg.get("com_frame", True),
            #     ylabel=plot_cfg.get("ylabel", r"$E^\star / m_N $"),
            #     show=bool(plot_cfg.get("show", False)),
            #     show_ni=bool(plot_cfg.get("show_ni", True)),
            #     show_level_labels=bool(plot_cfg.get("show_level_labels", False)),
            #     figsize=_figsize,
            #     title=plot_cfg.get("title", None),
            # )
            
            # Create level-by-level subplot comparison with non-interacting energies
            if plot_cfg.get("level_subplots", False):
                subplot_save_path = self._apply_figures_dir(self._resolve_subplot_save_path(plot_cfg))
                self._log(logging.INFO, "Level subplot save path: %s", subplot_save_path)
                noninteracting_energies = self._calculate_noninteracting_energies(
                    kept_mom2, lattice_size, massN
                )
                pt.plot_level_subplots_with_noninteracting(
                    energy_cm_data,
                    noninteracting_energies,
                    kept_irreps,
                    kept_levels,
                    kept_nPvec,
                    massN,
                    lattice_size,
                    save_path=subplot_save_path,
                    ylabel=plot_cfg.get("ylabel", r"$E^\star / m_N $"),
                )
            if self.bmat_preview_only:
                # Clean B-matrix preview filename: replace .pdf with _bmatrix_preview_{self.study_module_name}.pdf
                if plot_save_path.lower().endswith('.pdf'):
                    preview_save = plot_save_path[:-4] + f"_bmatrix_preview.pdf"
                else:
                    preview_save = plot_save_path + f"_bmatrix_preview.pdf"
                self._log(logging.INFO, "Plotting B matrix preview (pre-fit)...")
                self.study.plot_bmatrix_preview(
                    data2cm,
                    kept_mom2,
                    kept_irreps,
                    kept_levels,
                    mL      = massN[0] * lattice_size,
                    MN      = mass_n,
                    MK      = mass_k,
                    clip    = 30,
                    n_sweep = 3000,
                    save_path = preview_save,
                    show    = False,
                    y_scale_mode = plot_cfg.get("bmatrix_preview", {}).get("y_scale_mode", "data_central"),
                    show_level_labels = bool(plot_cfg.get("bmatrix_preview", {})
                                             .get("show_level_labels", False)),
                )
                self._log(logging.INFO, "B matrix preview saved: %s", preview_save)
                return
        # ── fit ──────────────────────────────────────────────────────────────
        # fit_cfg = self.config["fit"]
        p0 = self._resolve_initial_params(fit_cfg)
        n_refine   = int(fit_cfg.get("n_refine", 400))      # root-finding grid resolution
        step_mode  = str(fit_cfg.get("step_mode", "adaptive"))   # 'adaptive' or 'uniform' or "fast"
        # mN_err: uncertainty on MN used to set the NI-level buffer automatically.
        # If "mN_err" is not in the JSON, it is auto-derived from the standard
        # deviation of the massN bootstrap samples (massN[0] = central value,
        # massN[1:] = bootstrap samples).
        _mN_err_cfg = fit_cfg.get("mN_err", None)
        if _mN_err_cfg is not None:
            mN_err = float(_mN_err_cfg)
        elif hasattr(massN, "__len__") and len(massN) > 1:
            mN_err = float(np.std(massN[1:]))   # col 0 is the mean; std from bootstrap samples only
        else:
            mN_err = None
        n_maxiter = int(fit_cfg.get("n_maxiter", 10000))  # optimizer max iterations
        max_iter = fit_cfg.get("max_iter", None)
        if max_iter is not None:
            max_iter = int(max_iter)

        skip_minimization = bool(fit_cfg.get("skip_minimization", False))

        parametrization_name = self._resolve_parametrization_name()
        n_fit_params = len(p0)
        param_labels = self._resolve_param_labels(fit_cfg, n_fit_params)

        # ── B matrix preview (no fit needed) ────────────────────────────────
        preview_cfg = plot_cfg.get("bmatrix_preview", {})
        print(kept_levels)
        if preview_cfg.get("enabled", False):
            preview_save = self._apply_figures_dir(preview_cfg.get(
                "save_path", f"./Images/Spectrum/bmatrix_preview_{self.study_module_name}.pdf"
            ))
            self._log(logging.INFO, "Plotting B matrix preview (pre-fit)...")
            self.study.plot_bmatrix_preview(
                data2cm,
                kept_mom2,
                kept_irreps,
                kept_levels,
                mL      = massN[0] * lattice_size,
                MN      = mass_n,
                MK      = mass_k,
                clip    = float(preview_cfg.get("clip", 30.0)),
                n_sweep = int(preview_cfg.get("n_sweep", 800)),
                save_path = preview_save,
                show    = bool(preview_cfg.get("show", False)),
                y_scale_mode = str(preview_cfg.get("y_scale_mode", "data_central")),
            )
            self._log(logging.INFO, "B matrix preview saved: %s", preview_save)

        # ── auto-estimate p0 from B matrix if requested ──────────────────────────
        # ── auto-estimate p0 — SKIPPED if skip_minimization=True ─────────────
        if skip_minimization:
            self._log(logging.INFO,
                "skip_minimization=True: using config initial values as final params, "
                "auto_p0 ignored"
            )
        elif fit_cfg.get("auto_p0", False):
            self._log(logging.INFO, "Auto-estimating p0 from B matrix values...")
            _, p0_auto = self.study.estimate_initial_params_from_bmatrix(
                data2cm, kept_mom2, kept_irreps,
                mL = massN[0] * lattice_size,
                MN = mass_n,
                MK = mass_k,
            )
            if len(p0_auto) == len(p0):
                self._log(logging.INFO, "Auto p0: %s  (overrides config p0: %s)", p0_auto, p0)
                p0 = p0_auto
            else:
                self._log(logging.WARNING, "Auto p0 length mismatch — keeping config p0")
        self._log(logging.INFO, "Fit setup:")
        self._log(logging.INFO, "Parametrization: %s", parametrization_name)
        self._log(logging.INFO, "Initial parameter guesses (p0): %s", p0)
        self._log(logging.INFO, "Number of fit parameters: %d", n_fit_params)
        self._log(logging.INFO, "Model masses MN, MK: %s, %s", mass_n, mass_k)
        self._log(logging.INFO, "Refinement n: %d", n_refine)
        if max_iter is not None:
            self._log(logging.INFO, "Truncation after max_iter: %d callback iterations", max_iter)
        for name, value in zip(param_labels, p0):
            self._log(logging.INFO, "Initial guess %-24s = % .10g", name, value)
        # Central data values in the chosen observable space
        _data_central = _data_obs[:, 0]
        self._log(logging.INFO, "Number of data points: %d  (observable=%s)", len(_data_central), _fit_obs)

        # ── pretty-print data covariance ─────────────────────────────────────
        _errs = np.sqrt(np.diag(covp2))
        def _psq_lbl(m):
            try:
                return int(round(sum(int(x)**2 for x in m)))
            except (TypeError, ValueError):
                return int(round(float(m)))
        _lbs = [f"{ir}·P{_psq_lbl(mv)}·L{lv}"
                for ir, mv, lv in zip(kept_irreps, kept_mom2, kept_levels)]
        _lw  = max((len(lb) for lb in _lbs), default=8)
        self._log(logging.INFO, "Data covariance matrix:\n%s", covp2)
        # ─────────────────────────────────────────────────────────────────────

        fit_start = datetime.now()
        self._log(logging.INFO, "Starting chi2 minimization...")
        self._log(logging.INFO, "%s","~ " * 20)
        # in GenericFitRunner.run(), in the fit_cfg block:
        fit_strategy  = fit_cfg.get("strategy", "nelder-mead")
        n_starts      = int(fit_cfg.get("n_starts", 20))
        param_bounds  = fit_cfg.get("param_bounds", None)  # [[lo,hi], ...]
        xatol         = float(fit_cfg.get("xatol", 1e-6))
        fatol         = float(fit_cfg.get("fatol", 1e-6))
        # fit_mode: "omega" (default) uses getOmegaFromEcm; "eigenvalue" uses
        # getEigenvaluesFromEcm with λ/√(μ²+λ²) regularisation.
        # Passed only if the study module's minimizechi2 accepts it (e.g. fitting_code.py).
        fit_mode = str(fit_cfg.get("fit_mode", "omega"))
        if skip_minimization:
            self._log(logging.INFO, "=" * 50)
            self._log(logging.INFO, "SKIP MINIMIZATION: running chi2 with config params only")
            self._log(logging.INFO, "=" * 50)
            par       = np.asarray(p0, dtype=float)
            chi2_val  = self.study._chi2(
                par, _data_central,
                framesall, irrepsall, levelsall,
                massN[0] * lattice_size,
                covp2, mass_n, mass_k, n=n_refine,
                data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
                step_mode=step_mode, n_refine=n_refine, mN_err=mN_err,
                fit_mode=fit_mode, observable=_fit_obs, ni_sum=ni_sum_central,
            )
            converged = True
        else:
            fit_start = datetime.now()
            self._log(logging.INFO, "Starting chi2 minimization...")
            self._log(logging.INFO, "%s", "~ " * 20)
            import inspect as _inspect
            _mc_params = _inspect.signature(self.study.minimizechi2).parameters
            _extra = {}
            if "fit_mode"   in _mc_params: _extra["fit_mode"]   = fit_mode
            if "observable" in _mc_params: _extra["observable"] = _fit_obs
            if "ni_sum"     in _mc_params: _extra["ni_sum"]     = ni_sum_central
            _bs_cfg = fit_cfg.get("bootstrap")
            if "bootstrap" in _mc_params and _bs_cfg:
                _extra["bootstrap"] = _bs_cfg
            # multistart basin-mapping knobs (par engine)
            if "multistart_seed" in _mc_params and "multistart_seed" in fit_cfg:
                _extra["multistart_seed"] = int(fit_cfg["multistart_seed"])
            if "multistart_dump" in _mc_params and "multistart_dump" in fit_cfg:
                _extra["multistart_dump"] = str(fit_cfg["multistart_dump"])

            par, chi2_val, converged = self.study.minimizechi2(
                _data_central,
                framesall, irrepsall, levelsall,
                massN[0] * lattice_size,
                covp2, p0, mass_n, mass_k, n_refine,
                max_iter     = max_iter,
                n_maxiter    = n_maxiter,
                strategy     = fit_strategy,
                n_starts     = n_starts,
                param_bounds = [tuple(b) for b in param_bounds] if param_bounds else None,
                data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
                step_mode=step_mode, n_refine=n_refine, mN_err=mN_err,
                xatol=xatol, fatol=fatol,
                **_extra,
            )

        fit_end = datetime.now()
        self._log(
            logging.INFO,
            "Finished chi2 minimization in %.2f s",
            (fit_end - fit_start).total_seconds(),
        )

        num_data_points = len(_data_central)
        num_parameters = len(par)
        dof = num_data_points - num_parameters

        self._log(logging.INFO, "Final fit results:")
        self._log(logging.INFO, "Number of data points: %d", num_data_points)
        self._log(logging.INFO, "Number of fit parameters: %d", num_parameters)
        self._log(logging.INFO, "Parametrization: %s", parametrization_name)
        if not converged:
            self._log(logging.WARNING, "**FIT DID NOT CONVERGE** — results below are partial/truncated")
        self._log(logging.INFO, "Converged: %s", converged)
        self._log(logging.INFO, "Chi2: %s", chi2_val)
        self._log(logging.INFO, "DOF: %d", dof)
        self._log(logging.INFO, "Chi2/DOF: %s", chi2_val / dof)
        self._log(logging.INFO, "AIC: %s", chi2_val - 2 * dof)
        self._log(logging.INFO, "Parameters: %s", par)
        final_labels = param_labels if len(param_labels) == num_parameters else [f"p{i}" for i in range(num_parameters)]
        for name, value in zip(final_labels, par):
            self._log(logging.INFO, "Final fit %-28s = % .10g", name, value)

        # ── residuals ─────────────────────────────────────────────────────────
        _residuals_text = ""
        try:
            _mL_r    = massN[0] * lattice_size
            _model_v = self.study._model_datap2(
                par, framesall, irrepsall, levelsall, _mL_r, mass_n, mass_k,
                n=n_refine, data2cm=data2cm, kept_mom2=kept_mom2,
                kept_irreps=kept_irreps, step_mode=step_mode,
                n_refine=n_refine, mN_err=mN_err,
            )
            _data_v  = datap2[:, 0]
            _err_v   = np.sqrt(np.diag(covp2))
            _resid_v = _data_v - _model_v
            _pull_v  = np.where(_err_v > 0, _resid_v / _err_v, np.nan)
            def _psq_r(m):
                try:
                    return int(round(sum(int(x)**2 for x in m)))
                except (TypeError, ValueError):
                    return int(round(float(m)))

            # ── group indices by (irrep, psq) in order of first appearance ──
            _groups = {}   # (irrep, psq) → [idx, ...]
            for _i, (ir, mv) in enumerate(zip(kept_irreps, kept_mom2)):
                _key = (ir, _psq_r(mv))
                _groups.setdefault(_key, []).append(_i)

            _col_hdr = (f"    {'n':>3}  {'data':>12}  "
                        f"{'model':>12}  {'residual':>12}  {'pull':>7}")
            _r_sep   = "    " + "-" * (len(_col_hdr) - 4)
            _r_lines = ["", "  Fit residuals  (data - model):"]

            for (_ir, _psq), _idxs in _groups.items():
                _r_lines += [_r_sep,
                              f"    {_ir}  (Psq={_psq}):",
                              _r_sep, _col_hdr, _r_sep]
                for _i in _idxs:
                    _lv   = kept_levels[_i]
                    _dv   = _data_v[_i]
                    _mv   = _model_v[_i]
                    _rv   = _resid_v[_i]
                    _pv   = _pull_v[_i]
                    _flag = "  < large pull" if np.isfinite(_pv) and abs(_pv) > 2 else ""
                    _r_lines.append(
                        f"    {int(_lv):>3}  {_dv:>12.7f}  {_mv:>12.7f}"
                        f"  {_rv:>12.4e}  {_pv:>7.3f}{_flag}"
                    )
            _r_lines.append(_r_sep)

            _vp = _pull_v[np.isfinite(_pull_v)]
            if len(_vp):
                _r_lines.append(
                    f"    max|pull|={np.max(np.abs(_vp)):.3f}  "
                    f"rms pull={np.sqrt(np.mean(_vp**2)):.3f}  "
                    f"mean resid={np.mean(_resid_v[np.isfinite(_resid_v)]):.4e}"
                )
            _r_lines.append("")
            _residuals_text = "\n".join(_r_lines)
            self._log(logging.INFO, _residuals_text)
        except Exception as _exc:
            self._log(logging.WARNING, "Could not compute residuals: %s", _exc)
        # ─────────────────────────────────────────────────────────────────────

        vijmat = None  # ← default so plot_omega_and_eigenvalues never gets NameError
        if fit_cfg.get("compute_vij", True):
            vijmat = self.study.vij(
                par,
                _data_central,
                framesall, irrepsall, levelsall,
                massN[0] * lattice_size,
                covp2, mass_n, mass_k,
                n=n_refine,
                data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
                step_mode=step_mode, n_refine=n_refine, mN_err=mN_err,
                fit_mode=fit_mode, observable=_fit_obs, ni_sum=ni_sum_central,
            )
            self._log(logging.INFO, "Cov matrix of parameters")
            self._log(logging.INFO, "%s", vijmat)
            self._log(logging.INFO, "Errors on parameters")
            self._log(logging.INFO, "%s", np.sqrt(np.diag(vijmat)))
            
            # Save results as image
            results_text = f"""Final fit results:
                Number of data points: {num_data_points}
                Number of fit parameters: {num_parameters}
                Parametrization: {parametrization_name}
                Converged: {converged}
                Chi2: {chi2_val}
                DOF: {dof}
                Chi2/DOF: {chi2_val / dof:.8f}
                AIC: {chi2_val - 2 * dof:.8f}
                Parameters: {par}
                """
            
            # Add individual parameter values
            for name, value in zip(final_labels, par):
                results_text += f"Final fit {name:<28s} = {value:12.10g}\n"
            
            # Add covariance matrix and errors
            results_text += f"\nCov matrix of parameters\n{vijmat}\n"
            results_text += f"Errors on parameters\n{np.sqrt(np.diag(vijmat))}"
            results_text += f"\n{_residuals_text}"

            self._export_results(
                par, final_labels, chi2_val, dof, converged,
                num_data_points, num_parameters, parametrization_name,
                vijmat=vijmat,
            )
            
            # Determine save path for results image (without timestamp)
            plot_cfg = self.config.get("plot", {})
            if plot_cfg.get("enabled", True):
                # Get template without timestamp
                template = plot_cfg.get("save_path")
                if not template:
                    template = f"./Images/Spectrum/fit_{self.study_module_name}.pdf"
                # Format template without timestamp
                try:
                    clean_path = template.format(
                        study_module=self.study_module_name,
                        study_model=self.study_module_name,
                        module=self.study_module_name
                    )
                except (KeyError, IndexError, ValueError):
                    clean_path = template
                # Replace .pdf extension with _results.png
                base_path = os.path.splitext(clean_path)[0]
                results_image_path = f"{base_path}_results.png"
            else:
                # Default path if plotting is disabled
                results_image_path = f"./Images/fit_results_{self.study_module_name}.png"
            results_image_path = self._apply_figures_dir(results_image_path)

            # Ensure the directory exists
            _ri_dir = os.path.dirname(results_image_path)
            if _ri_dir:
                os.makedirs(_ri_dir, exist_ok=True)
            
            # Save the results as an image
            self._save_results_as_image(results_text, results_image_path)
            self._log(logging.INFO, "Saved results summary as image: %s", results_image_path)
        else:
            # Save results as image even when vij computation is disabled
            results_text = f"""Final fit results:
                Number of data points: {num_data_points}
                Number of fit parameters: {num_parameters}
                Parametrization: {parametrization_name}
                Converged: {converged}
                Chi2: {chi2_val}
                DOF: {dof}
                Chi2/DOF: {chi2_val / dof:.8f}
                AIC: {chi2_val - 2 * dof:.8f}
                Parameters: {par}
                """
            
            # Add individual parameter values
            for name, value in zip(final_labels, par):
                results_text += f"Final fit {name:<28s} = {value:12.10g}\n"
            results_text += f"\n{_residuals_text}"

            self._export_results(
                par, final_labels, chi2_val, dof, converged,
                num_data_points, num_parameters, parametrization_name,
            )

            # Determine save path for results image (without timestamp)
            plot_cfg = self.config.get("plot", {})
            if plot_cfg.get("enabled", True):
                # Get template without timestamp
                template = plot_cfg.get("save_path")
                if not template:
                    template = f"./Images/Spectrum/fit_{self.study_module_name}.pdf"
                # Format template without timestamp
                try:
                    clean_path = template.format(
                        study_module=self.study_module_name,
                        study_model=self.study_module_name,
                        module=self.study_module_name
                    )
                except (KeyError, IndexError, ValueError):
                    clean_path = template
                # Replace .pdf extension with _results.png
                base_path = os.path.splitext(clean_path)[0]
                results_image_path = f"{base_path}_results.png"
            else:
                # Default path if plotting is disabled
                results_image_path = f"./Images/fit_results_{self.study_module_name}.png"
            results_image_path = self._apply_figures_dir(results_image_path)

            # Ensure the directory exists
            _ri_dir = os.path.dirname(results_image_path)
            if _ri_dir:
                os.makedirs(_ri_dir, exist_ok=True)
            
            # Save the results as an image
            self._save_results_as_image(results_text, results_image_path)
            self._log(logging.INFO, "Saved results summary as image: %s", results_image_path)
         # ── chi2 dependence scan ──────────────────────────────────────────────
        dep_cfg = fit_cfg.get("chi2_dependence", {})
        if dep_cfg.get("enabled", False):
            self._log(logging.INFO, "%s", "~ " * 20)
            self._log(logging.INFO, "chi2 dependence scan starting...")

            dep_save_dir = dep_cfg.get(
                "save_dir",
                f"./Images/Spectrum/chi2_dependence_{self.study_module_name}/"
            )
            dep_bounds = dep_cfg.get("param_bounds", None)
            if dep_bounds is not None:
                dep_bounds = [tuple(b) for b in dep_bounds]

            profiles = self.study.chi2_dependence(
                par,
                datap2[:, 0],
                framesall, irrepsall, levelsall,
                massN[0] * lattice_size,
                covp2,
                mass_n, mass_k,
                n            = n_refine,
                n_points     = int(dep_cfg.get("n_points", 50)),
                n_sigma      = float(dep_cfg.get("n_sigma", 3.0)),   # ← renamed
                vij          = vijmat,                                # ← new
                param_bounds = dep_bounds,
                param_labels = final_labels,
                save_dir     = dep_save_dir if dep_cfg.get("plot", True) else None,
                show         = bool(dep_cfg.get("show", False)),
            )

            # ── summary table ─────────────────────────────────────────────────
            self._log(logging.INFO, "chi2 dependence summary:")
            self._log(logging.INFO,
                "  %-28s  %-14s  %-14s  %-14s  %-14s",
                "parameter", "best_fit", "scan_min_val", "scan_min_chi2", "delta_chi2",
            )
            self._log(logging.INFO, "  %s", "-" * 82)
            for i, prof in profiles.items():
                finite = np.isfinite(prof["chi2"])
                if finite.any():
                    idx           = int(np.nanargmin(prof["chi2"]))
                    scan_min_chi2 = float(prof["chi2"][idx])
                    scan_min_val  = float(prof["values"][idx])
                else:
                    scan_min_chi2 = np.nan
                    scan_min_val  = np.nan
                delta = scan_min_chi2 - prof["best_chi2"]
                self._log(logging.INFO,
                    "  %-28s  %-14.6g  %-14.6g  %-14.6g  %+.4g",
                    prof["label"], prof["best_val"],
                    scan_min_val, scan_min_chi2, delta,
                )

            if dep_cfg.get("plot", True):
                self._log(logging.INFO,
                    "chi2 dependence plots saved to: %s", dep_save_dir)
        qc_cfg = plot_cfg.get("quantization_condition", {})
        if qc_cfg.get("enabled", True):
            qc_save_path = qc_cfg.get("save_path", f"./Images/Spectrum/qc_{self.study_module_name}.pdf")
            if isinstance(qc_save_path, str):
                qc_save_path = qc_save_path.format(
                    study_module=self.study_module_name,
                    study_model=self.study_module_name,
                    module=self.study_module_name,
                )
            qc_save_path = self._apply_figures_dir(qc_save_path)
            chi2_over_dof = (chi2_val / dof) if dof != 0 else np.nan
            self.study.plot_quantization_condition(
                par,
                vijmat,
                data2cm,
                kept_mom2,
                kept_irreps,
                kept_levels,
                mL         = massN[0] * lattice_size,
                MN         = mass_n,
                MK         = mass_k,
                ecm_min          = float(qc_cfg["ecm_min"]) if "ecm_min" in qc_cfg else None,
                ecm_max          = float(qc_cfg["ecm_max"]) if "ecm_max" in qc_cfg else None,
                n_sweep          = int(qc_cfg.get("n_sweep", 800)),
                clip             = float(qc_cfg.get("clip", 30.0)),
                show_errorbars   = bool(qc_cfg.get("show_errorbars", True)),
                show_model_curve = bool(qc_cfg.get("show_model_curve", True)),
                param_labels     = final_labels,
                chi2_over_dof    = chi2_over_dof,
                save_path        = qc_save_path,
                show             = bool(qc_cfg.get("show", False)),
                show_level_labels= bool(qc_cfg.get("show_level_labels", False)),
            )
            self._log(logging.INFO, "Quantization condition plot saved: %s", qc_save_path)

        # ── bootstrap parameter samples, shared by the plot error bands ──────
        # (fit.bootstrap.save_path npy — used whether or not the bootstrap
        # refit ran this session, so plot-only skip_minimization reruns get
        # the same bands as the original fit)
        boot_par_samples = None
        _bs_path = (self.config.get("fit", {}).get("bootstrap") or {}) \
            .get("save_path")
        if _bs_path and os.path.exists(_bs_path):
            try:
                _bs_arr = np.load(_bs_path)
                if _bs_arr.ndim == 2 and _bs_arr.shape[1] == len(par):
                    boot_par_samples = _bs_arr
                    self._log(logging.INFO,
                              "plot bands: %d bootstrap parameter samples "
                              "from %s", len(boot_par_samples), _bs_path)
                else:
                    self._log(logging.WARNING,
                              "plot bands: %s has shape %s, expected (N, %d) "
                              "— ignoring", _bs_path, _bs_arr.shape, len(par))
            except Exception as _e:
                self._log(logging.WARNING,
                          "plot bands: could not load %s (%s)", _bs_path, _e)

        # ── multi-channel phase shift plot ────────────────────────────────────
        ps_cfg = plot_cfg.get("phase_shifts", {})
        if ps_cfg.get("enabled", True):
            ps_save = self._apply_figures_dir(ps_cfg.get(
                "save_path",
                f"./figures/{self.study_module_name}_phase_shifts.pdf",
            ).replace("{study_module}", self.study_module_name))
            # mixing_convention: "bb" (Blatt-Biedenharn eigenphases, as
            # parametrized), "bar" (Stapp bar phases — Nijmegen/SAID
            # convention, converted via bb_to_bar), or "both" (two figures;
            # the bar one gets a "_bar" suffix).
            _mix = str(ps_cfg.get("mixing_convention", "bb")).strip().lower()
            _convs = ["bb", "bar"] if _mix == "both" else [_mix]
            for _conv in _convs:
                _save = ps_save
                if _mix == "both" and _conv == "bar":
                    _root, _ext = os.path.splitext(ps_save)
                    _save = f"{_root}_bar{_ext or '.pdf'}"
                self.study.plot_phase_shifts_multichannel(
                    par,
                    vijmat,
                    data2cm,
                    kept_mom2,
                    kept_irreps,
                    kept_levels,
                    mL           = massN[0] * lattice_size,
                    MN           = mass_n,
                    MK           = mass_k,
                    ecm_min      = float(ps_cfg["ecm_min"]) if "ecm_min" in ps_cfg else None,
                    ecm_max      = float(ps_cfg["ecm_max"]) if "ecm_max" in ps_cfg else None,
                    n_sweep      = int(ps_cfg.get("n_sweep", 500)),
                    save_path    = _save,
                    show         = bool(ps_cfg.get("show", False)),
                    mN_samples   = massN,
                    L_lattice    = lattice_size,
                    n_par_samples= int(ps_cfg.get("n_par_samples", 400)),
                    par_samples  = boot_par_samples,
                    mixing_convention = _conv,
                    figsize      = tuple(ps_cfg["figsize"]) if "figsize" in ps_cfg else (10, 8),
                    show_level_labels = bool(ps_cfg.get("show_level_labels", True)),
                    bottom_ylabel = ps_cfg.get("bottom_ylabel", "levels"),
                    label_fontsize = ps_cfg.get("label_fontsize"),
                    legend_loc   = ps_cfg.get("legend_loc", "lower left"),
                    pie_size     = float(ps_cfg.get("pie_size", 220.0)),
                )
                self._log(logging.INFO, "Phase shift plot saved (%s): %s",
                          _conv, _save)

        # ── det / eigenvalue diagnostic plot ─────────────────────────────────
        # JSON: "plot": { "det_diagnostics": { "enabled": true, ... } }
        det_cfg     = plot_cfg.get("det_diagnostics", {})
        det_enabled = bool(det_cfg.get("enabled", True))   # on by default
        if det_enabled:
            diag_save_path = det_cfg.get("save_dir")
            if diag_save_path:
                diag_save_path = self._apply_figures_dir(diag_save_path)
            else:
                diag_save_path = (self._apply_figures_dir(None, as_dir=True)
                                  or f"./figures/{self.study_module_name}")
            self._log(logging.INFO, "Plotting det/eigenvalue diagnostics → %s", diag_save_path)
            self.study.plot_omega_and_eigenvalues(
                par,
                vijmat,
                kept_mom2,
                kept_irreps,
                kept_levels,
                data2cm,
                framesall,
                irrepsall,
                levelsall,
                mL          = massN[0] * lattice_size,
                MN          = mass_n,
                MK          = mass_k,
                n_refine    = n_refine,
                n_sweep     = int(det_cfg.get("n_sweep", 1000)),
                save_dir    = diag_save_path,
                show        = bool(det_cfg.get("show", False)),
                eig_ylim    = tuple(det_cfg.get("eig_ylim", [-0.5, 0.5])),
                mN_samples  = massN[1:],
                L           = lattice_size,
                n_par_samples = int(det_cfg.get("n_par_samples", 400)),
                step_mode   = step_mode,
                mN_err      = mN_err,
            )
        else:
            self._log(logging.INFO, "det_diagnostics disabled — skipping")

        # ── per-wave predicted energies (single-wave solo spectra) ────────────
        # JSON: "plot": { "wave_energies": { "enabled": true,
        #                 "n_par_samples": 60, "save_path": ... } }
        # At the best-fit parameters, the QC roots of each partial wave in
        # isolation per irrep(P²), next to data and the full-model roots, with
        # σ68 error bars from bootstrap parameter samples (fit.bootstrap
        # save_path npy if present) or Gaussian vij draws as fallback.
        we_cfg = plot_cfg.get("wave_energies", {})
        if bool(we_cfg.get("enabled", True)):
            we_save = self._apply_figures_dir(we_cfg.get(
                "save_path",
                f"./figures/{self.study_module_name}_wave_energies.pdf",
            ).replace("{study_module}", self.study_module_name))
            we_nsamp = int(we_cfg.get("n_par_samples", 60))
            par_samples = boot_par_samples
            if par_samples is None and vijmat is not None:
                try:
                    par_samples = np.random.default_rng(0).multivariate_normal(
                        np.asarray(par, dtype=float),
                        np.asarray(vijmat, dtype=float), we_nsamp)
                    self._log(logging.INFO,
                              "wave_energies: %d Gaussian vij draws "
                              "(no bootstrap npy found)", we_nsamp)
                except Exception as _e:
                    self._log(logging.WARNING,
                              "wave_energies: vij draws failed (%s) — "
                              "no error bars on predictions", _e)
            # "rcparams": {...} — matplotlib settings for this figure only.
            # The plot reads marker and font sizes from rcParams, so this is
            # what lets a paper figure be drawn at its final on-page size with
            # readable markers/labels (cf. scripts/make_paper_figures.py).
            we_rc = we_cfg.get("rcparams", {}) or {}
            with plt.rc_context(we_rc):
                _we_res = self.study.plot_wave_predicted_energies(
                    par, data2cm, kept_mom2, kept_irreps, kept_levels,
                    mL           = massN[0] * lattice_size,
                    MN           = mass_n,
                    MK           = mass_k,
                    par_samples  = par_samples,
                    n_par_samples= we_nsamp,
                    n_refine     = n_refine,
                    step_mode    = step_mode,
                    save_path    = we_save,
                    show         = bool(we_cfg.get("show", False)),
                    title        = we_cfg.get("title",
                                              f"{self.study_module_name}: "
                                              "per-wave predicted energies"),
                    # inherit the spectrum plot's y-axis label so the two figures
                    # read as one series (per-figure override still possible)
                    ylabel       = we_cfg.get("ylabel", plot_cfg.get("ylabel")),
                    y_from_data  = bool(we_cfg.get("y_from_data", True)),
                    y_top_pad    = float(we_cfg.get("y_top_pad", 0.08)),
                    y_bottom_pad = float(we_cfg.get("y_bottom_pad", 0.05)),
                    show_legend  = bool(we_cfg.get("show_legend", True)),
                    show_level_labels=bool(we_cfg.get("show_level_labels", False)),
                    figsize      = (tuple(we_cfg["figsize"])
                                    if isinstance(we_cfg.get("figsize"), list)
                                    and len(we_cfg["figsize"]) == 2 else None),
                )
            self._log(logging.INFO, "Wave-energy plot saved: %s", we_save)
            # how big are the prediction error bars actually? (they are often
            # smaller than the marker, which reads as "no error bars")
            try:
                # results[(psq, irrep)]["model"] = (energies, sigmas)
                _sig = []
                for _blk in (_we_res or {}).values():
                    _m = _blk.get("model")
                    if isinstance(_m, (tuple, list)) and len(_m) > 1 and _m[1] is not None:
                        _sig.extend(np.atleast_1d(_m[1]).ravel().tolist())
                _sig = np.asarray([x for x in _sig if np.isfinite(x)], dtype=float)
                if _sig.size:
                    self._log(logging.INFO,
                              "wave_energies: full-model root sigma over "
                              "parameter samples — median %.2e, max %.2e "
                              "(in E*/m_ref); marker half-height is ~%.1e",
                              float(np.median(_sig)), float(_sig.max()), 3e-4)
            except Exception as _e:
                self._log(logging.DEBUG, "sigma summary failed: %s", _e)
            # optional JSON dump of the per-block numbers, so a paper figure
            # can be redrawn without re-solving the quantization condition
            _dump = we_cfg.get("dump_path")
            if _dump:
                try:
                    _out = {}
                    for (_psq, _irr), _r in (_we_res or {}).items():
                        _d_e, _d_s = _r["data"]
                        _m_e, _m_s = _r["model"]
                        _out[f"{_psq}|{_irr}"] = {
                            "window": [float(x) for x in _r["window"]],
                            "ni":     [float(x) for x in np.atleast_1d(_r["ni"])],
                            "levels": [int(x) for x in _r.get("levels", [])],
                            "data":   [[float(x) for x in np.atleast_1d(_d_e)],
                                       [float(x) for x in np.atleast_1d(_d_s)]],
                            "model":  [[float(x) for x in np.atleast_1d(_m_e)],
                                       [float(x) for x in np.atleast_1d(_m_s)]],
                            "solo":   {str(_k): [[float(x) for x in np.atleast_1d(_v[0])],
                                                 [float(x) for x in np.atleast_1d(_v[1])]]
                                       for _k, _v in _r["solo"].items()},
                        }
                    os.makedirs(os.path.dirname(os.path.abspath(_dump)), exist_ok=True)
                    with open(_dump, "w") as _fh:
                        json.dump(_out, _fh, indent=1)
                    self._log(logging.INFO, "wave_energies: dumped %d blocks -> %s",
                              len(_out), _dump)
                except Exception as _e:
                    self._log(logging.WARNING, "wave_energies dump failed: %s", _e)
        else:
            self._log(logging.INFO, "wave_energies disabled — skipping")

        # ── QC null-eigenvector wave decomposition ────────────────────────────
        # JSON: "plot": { "eigenvector_decomposition": { "enabled": false, ... } }
        eigvec_cfg    = plot_cfg.get("eigenvector_decomposition", {})
        eigvec_on     = bool(eigvec_cfg.get("enabled", False))   # off by default
        eigvec_method = str(eigvec_cfg.get("method", "eigh"))
        eigvec_nr     = int(eigvec_cfg.get("n_refine", n_refine))

        if eigvec_on:
            self._log(logging.INFO, "%s", "~ " * 20)
            self._log(logging.INFO,
                      "Computing QC null-eigenvector decomposition (method=%s)...",
                      eigvec_method)
            decomp_results = self.study.eigenvector_decomposition(
                par,
                data2cm,
                kept_mom2,
                kept_irreps,
                kept_levels,
                mL        = massN[0] * lattice_size,
                MN        = mass_n,
                MK        = mass_k,
                n_refine  = eigvec_nr,
                step_mode = step_mode,
                method    = eigvec_method,
            )
            pt.print_eigenvector_decomposition(
                decomp_results,
                log_fn=lambda msg: self._log(logging.INFO, "%s", msg),
            )
            if eigvec_cfg.get("plot", False) and decomp_results:
                eigvec_save = self._apply_figures_dir(eigvec_cfg.get(
                    "save_path",
                    f"./figures/{self.study_module_name}/eigenvector_decomposition.pdf",
                ))
                pt.plot_eigenvector_decomposition(
                    self.study,
                    decomp_results,
                    save_path = eigvec_save,
                    show      = bool(eigvec_cfg.get("show", False)),
                    title     = eigvec_cfg.get("title", None),
                )
                self._log(logging.INFO,
                          "Eigenvector decomposition plot saved: %s", eigvec_save)
        else:
            self._log(logging.INFO, "eigenvector_decomposition disabled — skipping")

        # ── fit comparison / kcotδ-vs-p² plot ─────────────────────────────────
        # On by default: with no "fits" list it draws the current best fit
        # alone (the per-run q·cotδ plot). Provide "fits" to overlay several
        # parameter sets.
        # "fit_comparison" is the pre-rename key, still honoured.
        comp_cfg = plot_cfg.get("luscher", plot_cfg.get("fit_comparison", {}))
        if comp_cfg.get("enabled", True):
            fits_list = comp_cfg.get("fits", [])
            if not fits_list:
                # error band: bootstrap samples if available, else vij draws
                fits_list = [{"label": "best fit",
                              "params": [float(x) for x in par],
                              "chi2_dof": (chi2_val / dof) if dof else None,
                              "vij": vijmat,
                              "par_samples": boot_par_samples,
                              "n_par_samples": int(comp_cfg.get(
                                  "n_par_samples", 200))}]
            comp_save = self._apply_figures_dir(comp_cfg.get(
                "save_path",
                f"./figures/{self.study_module_name}/luscher.pdf"
            ))
            self._log(logging.INFO, "Plotting Lüscher kcotδ plot with %d parameter sets...", len(fits_list))

            # Each fit dict should have: label, params, color (opt), linestyle (opt)
            for idx, fit_spec in enumerate(fits_list):
                if 'params' not in fit_spec:
                    self._log(logging.WARNING, "  Fit %d missing 'params' key, skipping", idx+1)
                    continue
                label = fit_spec.get('label', f'Fit {idx+1}')
                self._log(logging.INFO, "  %s: params=%s", label, fit_spec['params'])

            self.study.plot_luscher(
                fits           = fits_list,
                data2cm        = data2cm,
                kept_mom2      = kept_mom2,
                kept_irreps    = kept_irreps,
                kept_levels    = kept_levels,
                mL             = massN[0] * lattice_size,
                MN             = mass_n,
                MK             = mass_k,
                ecm_min        = float(comp_cfg.get("ecm_min")) if "ecm_min" in comp_cfg else None,
                ecm_max        = float(comp_cfg.get("ecm_max")) if "ecm_max" in comp_cfg else None,
                n_sweep        = int(comp_cfg.get("n_sweep", 800)),
                clip           = float(comp_cfg.get("clip", 30.0)),
                show_errorbars = bool(comp_cfg.get("show_errorbars", True)),
                save_path      = comp_save,
                show           = bool(comp_cfg.get("show", False)),
                cov            = covp2,
                data_central   = datap2[:, 0],
                frames_all     = framesall,
                irreps_all     = irrepsall,
                levels_all     = levelsall,
                show_level_labels=bool(comp_cfg.get("show_level_labels", False)),
            )
            self._log(logging.INFO, "Lüscher plot saved: %s", comp_save)
        else:
            self._log(logging.INFO, "luscher disabled — skipping")
        
        # self.study.plot_box_quantization_diagnostics(
        #     par,
        #     kept_mom2,
        #     kept_irreps,
        #     mL     = massN[0] * lattice_size,
        #     MN     = mass_n,
        #     MK     = mass_k,
        #     ecm_min=None,
        #     ecm_max=None,
        #     n_sweep = int(fit_cfg.get("n_refine", 400)),
        #     save_dir = diag_save_path,
        #     show    = bool(fit_cfg.get("det_diag_show", False)),
        #     data2cm   = data2cm
        # )
        # self._log(logging.INFO, "Determinant diagnostic plot saved: %s", diag_save_path)
        # # ── chi2 landscape plot ──────────────────────────────────────────────────
        # landscape_cfg = fit_cfg.get("chi2_landscape", {})
        # if landscape_cfg.get("enabled", False):
        #     n_par = len(par)
        #     if n_par < 2:
        #         self._log(logging.WARNING, "chi2 landscape requires >= 2 parameters, skipping.")
        #     else:
        #         p_idx = tuple(landscape_cfg.get("param_indices", [0, 1]))
        #         l_bounds = landscape_cfg.get("param_bounds", None)
        #         if l_bounds is not None:
        #             l_bounds = [tuple(b) for b in l_bounds]
        #         landscape_save = landscape_cfg.get(
        #             "save_path", f"./Images/Spectrum/chi2_landscape_{self.study_module_name}.png"
        #         )
        #         self._log(logging.INFO, "Computing chi2 landscape for params %s...", p_idx)
        #         self.study.plot_chi2_landscape(
        #             par,
        #             datap2[:, 0],
        #             framesall, irrepsall, levelsall,
        #             massN[0] * lattice_size,
        #             covp2,
        #             mass_n, mass_k,
        #             n             = n_refine,
        #             param_indices = p_idx,
        #             n_grid        = int(landscape_cfg.get("n_grid", 40)),
        #             param_bounds  = l_bounds,
        #             scale         = float(landscape_cfg.get("scale", 3.0)),
        #             save_path     = landscape_save,
        #             show          = bool(landscape_cfg.get("show", False)),
        #             log_scale     = bool(landscape_cfg.get("log_scale", True)),
        #         )
        #         self._log(logging.INFO, "chi2 landscape saved: %s", landscape_save)
        


def main():
    parser = argparse.ArgumentParser(description="Run a fit from a JSON config file")
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--module", dest="study_module", help="Override study module (e.g. sdmixing_study_1p1)")
    parser.add_argument("--bmat-preview-only", action="store_true", help="Only plot B-matrix preview and exit")
    args = parser.parse_args()

    runner = None
    _main_start = datetime.now()
    try:
        runner = GenericFitRunner(args.config, study_module_override=args.study_module, bmat_preview_only=args.bmat_preview_only)
        runner.run()
        _main_end = datetime.now()
        _elapsed  = _main_end - _main_start
        _mins, _secs = divmod(int(_elapsed.total_seconds()), 60)
        print(f"\n{'='*60}")
        print(f"Run finished: {_main_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time  : {_mins}m {_secs}s")
        print(f"{'='*60}\n")
    except Exception as exc:
        if runner is not None and runner.logging_enabled:
            runner.logger.critical("Run failed and must stop: %s", exc)
            runner.logger.exception("Fatal exception details")
        else:
            print(f"CRITICAL: Run failed and must stop: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Ensure no interactive figures remain alive between repeated runs.
        plt.close("all")


if __name__ == "__main__":
    main()