import argparse
import glob
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
import plot_spectrum as _ps
import fit_plots

# Default to headless plotting for CLI fits so repeated minimizations do not
# spawn/accumulate GUI-backed Python app processes on macOS.
if os.environ.get("HPW_HEADLESS", "1").strip().lower() not in ("0", "false", "no", "off"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

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
        print(text)

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
        # Create a figure
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axis('off')  # Turn off the axes

        # Add the text to the figure with left alignment
        ax.text(0.05, 0.95, results_text, fontsize=11, ha='left', va='top', 
                transform=ax.transAxes, fontfamily='monospace')

        # Save the figure as an image
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

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
                    irrep = key.split("_")[0]
                    mom2  = int(key.split("_")[1][-1])
                    level = int(key.split("_")[-1][-1])
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
        covp2  = np.cov(datap2)
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
                    irrep = key.split("_")[0]
                    mom2 = int(key.split("_")[1][-1])
                    level =  int(key.split("_")[-1][-1])
                    entries.append({
                            "level": level,
                            "nP": mom2,
                            "irrep": irrep,
                            "momentum":  self._assign_momentum_to_nP( mom2 ),
                        })
            # Sort entries by nP in ascending order
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
                    shift = f2[name + "/dE_NN"]
                    E_N1 = np.sqrt(massN**2 + f2[name + "/Psq_N1"][0] * (two_pi_over_L) ** 2)
                    E_N2 = np.sqrt(massN**2 + f2[name + "/Psq_N2"][0] * (two_pi_over_L) ** 2)
                    auxread = (shift + E_N1 + E_N2) / massN
                else:
                    auxread = (
                        np.array(f2[name + "/E_N1"])
                        + np.array(f2[name + "/E_N2"])
                        + np.array(f2[name + "/dE_NN"])
                    ) / massN

                auxread2 = np.array(f2[name + "/dE_NN"]) / massN
                if np.mean(auxread) < cutoff2:
                    data2 = np.vstack((data2, auxread))
                    datashifts = np.vstack((datashifts, auxread2))
                    kept_mom2.append(selection_entries[i]["momentum"])
                    kept_irreps.append(selection_entries[i]["irrep"])
                    kept_levels.append(selection_entries[i]["level"])
                    kept_nPvec.append(selection_entries[i]["nP"])
                    kept_entries.append(selection_entries[i])
                else:
                    rejected_entries.append(selection_entries[i])

        if rejected_entries:
            self._log_data_entries("Levels rejected by cutoff", rejected_entries)

        if not kept_nPvec:
            self._log(logging.CRITICAL, "No selected levels passed the cutoff; cannot continue fit.")
            raise ValueError("No selected levels passed the cutoff; adjust selection or cutoff in config.")
        data_Elab_mref = data2
        data2cm = np.sqrt(data2**2 - np.outer(np.array(kept_nPvec), (2 * math.pi / lattice_size / massN) ** 2))
        datap2 = data2cm**2 / 4 - 1.0
        covp2 = np.cov(datap2)
        framesall, irrepsall, levelsall = self._build_grouped_inputs(kept_mom2, kept_irreps)

        energy_cm_data = self.study.build_energy_cm_dict(data2cm, kept_mom2, kept_irreps, kept_levels)
        # print("Energy CM data:", energy_cm_data)
        # ref masses
        fit_cfg = self.config["fit"]
        mass_n = float(fit_cfg.get("MN", 1.0))
        mass_k = float(fit_cfg.get("MK", 1.0))
        plot_cfg = self.config.get("plot", {})
        if plot_cfg.get("enabled", True):
            if plot_cfg.get("save_timestamp", True):
                plot_save_path = self._resolve_plot_save_path(plot_cfg)
            else:
                plot_save_path = plot_cfg.get("save_path", f"./Images/Spectrum/fit_{self.study_module_name}.pdf")
            if "{study_module}" in plot_save_path:
                plot_save_path = plot_save_path.format(study_module=self.study_module_name)
            if isinstance(plot_save_path, str):
                plot_dir = os.path.dirname(plot_save_path)
                if plot_dir:
                    os.makedirs(plot_dir, exist_ok=True)
            self._log(logging.INFO, "Plot save path: %s", plot_save_path)
            _figsize = plot_cfg.get("spectrum_figsize", None)
            if isinstance(_figsize, list) and len(_figsize) == 2:
                _figsize = tuple(_figsize)
            _ps.make_spectrum_plot(
                energy_cm_data,
                massN,
                lattice_size,
                save_path=plot_save_path,
                C_o_M=plot_cfg.get("com_frame", True),
                ylabel=plot_cfg.get("ylabel", r"$E^\star / m_N $"),
                show=bool(plot_cfg.get("show", False)),
                show_ni=bool(plot_cfg.get("show_ni", True)),
                show_level_labels=bool(plot_cfg.get("show_level_labels", True)),
                figsize=_figsize,
                title=plot_cfg.get("title", None),
            )
            
            # Create level-by-level subplot comparison with non-interacting energies
            if plot_cfg.get("level_subplots", False):
                subplot_save_path = self._resolve_subplot_save_path(plot_cfg)
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
                fit_plots.plot_bmatrix_preview(
                    self.study,
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
                )
                self._log(logging.INFO, "B matrix preview saved: %s", preview_save)
                return
        # ── fit ──────────────────────────────────────────────────────────────
        # fit_cfg = self.config["fit"]
        p0 = self._resolve_initial_params(fit_cfg)
        n_refine   = int(fit_cfg.get("n_refine", 400))      # root-finding grid resolution
        step_mode  = str(fit_cfg.get("step_mode", "adaptive"))   # 'adaptive' or 'uniform'
        # mN_err: uncertainty on MN used to set the NI-level buffer automatically.
        # If "mN_err" is not in the JSON, it is auto-derived from the standard
        # deviation of the massN bootstrap samples (massN[0] = central value,
        # massN[1:] = bootstrap samples).
        _mN_err_cfg = fit_cfg.get("mN_err", None)
        if _mN_err_cfg is not None:
            mN_err = float(_mN_err_cfg)
        elif hasattr(massN, "__len__") and len(massN) > 1:
            mN_err = float(np.std(massN))
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
            preview_save = preview_cfg.get(
                "save_path", f"./Images/Spectrum/bmatrix_preview_{self.study_module_name}.pdf"
            )
            self._log(logging.INFO, "Plotting B matrix preview (pre-fit)...")
            fit_plots.plot_bmatrix_preview(
                self.study,
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
        self._log(logging.INFO, "Number of data points: %d", len(datap2[:, 0]))

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
        xatol         = float(fit_cfg.get("xatol", 1e-6))   # param convergence tolerance
        fatol         = float(fit_cfg.get("fatol", 1e-6))   # function convergence tolerance
        if skip_minimization:
            self._log(logging.INFO, "=" * 50)
            self._log(logging.INFO, "SKIP MINIMIZATION: running chi2 with config params only")
            self._log(logging.INFO, "=" * 50)
            par       = np.asarray(p0, dtype=float)
            chi2_val  = self.study._chi2(
                par, datap2[:, 0],
                framesall, irrepsall, levelsall,
                massN[0] * lattice_size,
                covp2, mass_n, mass_k, n=n_refine,
                data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
                step_mode=step_mode, n_refine=n_refine, mN_err=mN_err,
            )
            converged = True
        else:
            fit_start = datetime.now()
            self._log(logging.INFO, "Starting chi2 minimization...")
            self._log(logging.INFO, "%s", "~ " * 20)
            par, chi2_val, converged = self.study.minimizechi2(
            datap2[:, 0],
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
        )
        
        fit_end = datetime.now()
        self._log(
            logging.INFO,
            "Finished chi2 minimization in %.2f s",
            (fit_end - fit_start).total_seconds(),
        )

        num_data_points = len(datap2[:, 0])
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
            datap2[:, 0],
            framesall, irrepsall, levelsall,
            massN[0] * lattice_size,
            covp2, mass_n, mass_k,
            n=n_refine,
            data2cm=data2cm, kept_mom2=kept_mom2, kept_irreps=kept_irreps,
            step_mode=step_mode, n_refine=n_refine, mN_err=mN_err,
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
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(results_image_path), exist_ok=True)
            
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
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(results_image_path), exist_ok=True)
            
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
            chi2_over_dof = (chi2_val / dof) if dof != 0 else np.nan
            fit_plots.plot_quantization_condition(
                self.study,
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
            )
            self._log(logging.INFO, "Quantization condition plot saved: %s", qc_save_path)

        # ── multi-channel phase shift plot ────────────────────────────────────
        ps_cfg = plot_cfg.get("phase_shifts", {})
        if ps_cfg.get("enabled", True):
            ps_save = ps_cfg.get(
                "save_path",
                f"./figures/{self.study_module_name}_phase_shifts.pdf",
            ).replace("{study_module}", self.study_module_name)
            fit_plots.plot_phase_shifts_multichannel(
                self.study,
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
                save_path    = ps_save,
                show         = bool(ps_cfg.get("show", False)),
                mN_samples   = massN,
                L_lattice    = lattice_size,
                n_par_samples= int(ps_cfg.get("n_par_samples", 400)),
            )
            self._log(logging.INFO, "Phase shift plot saved: %s", ps_save)

        # ── det diagonistic plot ──────────────────────────────────────────────────
        diag_save_path = f"./Figures/{self.study_module_name}/det_lambdas.pdf"
        fit_plots.plot_omega_and_eigenvalues(
            self.study,
            par,
            vijmat,
            kept_mom2,
            kept_irreps,
            kept_levels,
            data2cm,
            framesall,       # ← already built earlier in run()
            irrepsall,       # ← already built earlier in run()
            levelsall,       # ← already built earlier in run()
            mL        = massN[0] * lattice_size,
            MN        = mass_n,
            MK        = mass_k,
            n_refine  = n_refine,
            n_sweep   = 1000,
            save_dir  = diag_save_path,
            show      = False,
            eig_ylim  = (-0.5, 0.5),
            mN_samples=massN[1:],
            L= lattice_size,
            n_par_samples = 400,
            step_mode = step_mode,
            mN_err    = mN_err,
        )
        
        # ── QC null-eigenvector wave decomposition (always runs for multi-wave fits)
        eigvec_cfg      = plot_cfg.get("eigenvector_decomposition", {})
        eigvec_n_refine = int(eigvec_cfg.get("n_refine", 100))
        self._log(logging.INFO, "%s", "~ " * 20)
        self._log(logging.INFO, "Computing QC null-eigenvector decomposition...")
        decomp_results = self.study.eigenvector_decomposition(
            par,
            data2cm,
            kept_mom2,
            kept_irreps,
            mL        = massN[0] * lattice_size,
            MN        = mass_n,
            MK        = mass_k,
            n_refine  = eigvec_n_refine,
            step_mode = step_mode,
        )
        fit_plots.print_eigenvector_decomposition(
            decomp_results,
            log_fn=lambda msg: self._log(logging.INFO, "%s", msg),
        )
        if eigvec_cfg.get("plot", False) and decomp_results:
            eigvec_save_path = eigvec_cfg.get(
                "save_path",
                f"./figures/{self.study_module_name}/eigenvector_decomposition.pdf",
            )
            fit_plots.plot_eigenvector_decomposition(
                self.study,
                decomp_results,
                save_path = eigvec_save_path,
                show      = bool(eigvec_cfg.get("show", False)),
                title     = eigvec_cfg.get("title", None),
            )
            self._log(logging.INFO, "Eigenvector decomposition plot saved: %s", eigvec_save_path)

        # ── fit comparison plot (multiple parametrizations) ──────────────────
        comp_cfg = plot_cfg.get("fit_comparison", {})
        if comp_cfg.get("enabled", False):
            fits_list = comp_cfg.get("fits", [])
            if not fits_list:
                self._log(logging.WARNING, "fit_comparison enabled but no fits provided")
            else:
                comp_save = comp_cfg.get(
                    "save_path",
                    f"./figures/{self.study_module_name}/fit_comparison.pdf"
                )
                self._log(logging.INFO, "Plotting fit comparison with %d parameter sets...", len(fits_list))
                
                # Each fit dict should have: label, params, color (opt), linestyle (opt)
                for idx, fit_spec in enumerate(fits_list):
                    if 'params' not in fit_spec:
                        self._log(logging.WARNING, "  Fit %d missing 'params' key, skipping", idx+1)
                        continue
                    label = fit_spec.get('label', f'Fit {idx+1}')
                    self._log(logging.INFO, "  %s: params=%s", label, fit_spec['params'])
                
                fit_plots.plot_fit_comparison(
                    self.study,
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
                )
                self._log(logging.INFO, "Fit comparison plot saved: %s", comp_save)
                
                # Collect labels for the congratulatory message
                labels = [fit.get('label', f'Fit {i+1}') for i, fit in enumerate(fits_list)]
                labels_str = ', '.join(labels)
                
                # Print congratulatory message and exit
                print("\n" + "=" * 60)
                print("We made the fit comparison for these:")
                print(f"  {labels_str}")
                print("Congrats!")
                print("=" * 60 + "\n")
                
                self._log(logging.INFO, "Fit comparison complete. Exiting without running minimization.")
                return  # Exit early, skip minimization
        
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


doc = '''
HPW fit task using GenericFitRunner and B-matrix (BMAT) approach.

Two usage modes:

1. config_file mode – point to a pre-made GenericFitRunner JSON:
     single_channel_fit:
       config_file: path/to/fit.json
       study_module: QC2.my_fit_model   # optional
       bmat_preview_only: false          # optional

2. Auto-config mode – build the JSON automatically from previous task output.
   twoJ, L (partial wave), and twoS are derived from the particle names in
   general/particles.py.  Only the K-matrix initial guesses need to be set:

     single_channel_fit:
       data_file: /path/to/spectrum.hdf5   # omit to auto-discover from project dir

       scattering:
         'N(0)_ref,pi(0)_ref':
           PSQ0:
             - G1u: [0, 1]
           PSQ1:
             - G1: [0, 1]

       # Quantum numbers – twoS auto-derived from particles.py
       # twoJ defaults to |2L - twoS| (minimum J); override if needed.
       L:       0              # partial wave: 0=S, 1=P, 2=D, ...
       k_matrix: polynomial   # polynomial | ERE | zero | coupled
       params:               # K-matrix initial guesses (required)
         - {name: c0, initial: -0.2}
         - {name: c1, initial: 0.7}
       # Optional overrides (normally auto-derived):
       # twoS: 1
       # twoJ: 1

       # fit settings (all optional – defaults shown)
       MN: 1.0
       MK: 1.0
       strategy: nelder-mead
       n_starts: 20
       n_refine: 400
       auto_p0: false
       cutoff: 10.0
       study_module: QC2.my_fit_model
       bmat_preview_only: false   # set true to only plot B-matrix preview
'''

# Partial-wave label → integer L
_WAVE_LABEL_TO_L = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6}
_L_TO_WAVE_LABEL = {v: k for k, v in _WAVE_LABEL_TO_L.items()}

# Momentum vector for each PSQ label / integer – mirrors GenericFitRunner
_PSQ_TO_MOMENTUM = {
    'PSQ0': [0, 0, 0], 0: [0, 0, 0],
    'PSQ1': [0, 0, 1], 1: [0, 0, 1],
    'PSQ2': [1, 1, 0], 2: [1, 1, 0],
    'PSQ3': [1, 1, 1], 3: [1, 1, 1],
    'PSQ4': [0, 0, 2], 4: [0, 0, 2],
}


def _parse_particle_base(part_str):
    """Extract base particle name from strings like 'N(0)_ref' → 'N'."""
    name = part_str.strip()
    for sep in ('(', '_'):
        if sep in name:
            name = name.split(sep)[0]
            break
    return name


def _infer_channel_qn(channel_str, task_params):
    """Infer a quantum_numbers channel dict from the scattering channel string.

    Particle spins are read from ``general.particles.particles``.
    ``twoS`` is computed (or warned about if ambiguous).
    ``L`` comes from ``task_params['L']`` (default 0 = S-wave).
    ``twoJ`` defaults to ``|2L - twoS|`` (minimum J); override via ``task_params['twoJ']``.

    Returns a channel dict on success, or ``None`` if any particle is unknown.
    """
    try:
        from general.particles import particles as PARTICLES
    except ImportError:
        logging.warning("HPWFitTask: could not import general.particles – quantum numbers not auto-inferred.")
        return None

    parts = [p.strip() for p in channel_str.split(',')]
    particle_bases = [_parse_particle_base(p) for p in parts]

    spins = []
    for pname in particle_bases:
        if pname in PARTICLES:
            spins.append(float(PARTICLES[pname]['spin']))
        else:
            logging.warning(f"HPWFitTask: unknown particle '{pname}' – quantum numbers not auto-inferred.")
            return None

    if len(spins) != 2:
        return None

    twoS_max = int(round(2 * (spins[0] + spins[1])))
    twoS_min = int(round(2 * abs(spins[0] - spins[1])))

    # twoS: use explicit value or auto-select
    twoS = int(task_params['twoS']) if (task_params and 'twoS' in task_params) else None
    if twoS is None:
        if twoS_min == twoS_max:
            twoS = twoS_min
        else:
            twoS = twoS_min
            logging.warning(
                f"HPWFitTask: channel '{channel_str}' has multiple spin states "
                f"(twoS = {twoS_min} … {twoS_max}). Defaulting to twoS={twoS}. "
                "Add 'twoS' to task params to override."
            )

    # L: accept integer or spectroscopic letter ('S','P','D',...)
    L_raw = task_params.get('L', 0) if task_params else 0
    if isinstance(L_raw, str):
        L = _WAVE_LABEL_TO_L.get(L_raw.upper(), 0)
    else:
        L = int(L_raw)

    # twoJ: explicit or minimum J = |2L - twoS|
    twoJ = int(task_params['twoJ']) if (task_params and 'twoJ' in task_params) else abs(2 * L - twoS)

    wave_str = f"{twoS + 1}{_L_TO_WAVE_LABEL.get(L, str(L))}{twoJ}"
    channel_name = '_'.join(particle_bases) + f'_{wave_str}'

    k_matrix = task_params.get('k_matrix', 'polynomial') if task_params else 'polynomial'
    params   = task_params.get('params', []) if task_params else []

    if not params:
        logging.warning(
            f"HPWFitTask: no 'params' (K-matrix initial guesses) given for channel "
            f"'{channel_str}'. Add them to task params, e.g.:\n"
            "  params:\n    - {name: c0, initial: -0.2}\n    - {name: c1, initial: 0.7}"
        )

    qn_channel = {
        "name":     channel_name,
        "twoJ":     twoJ,
        "L":        L,
        "Lp":       L,
        "twoS":     twoS,
        "enabled":  True,
        "k_matrix": k_matrix,
    }
    if params:
        qn_channel["params"] = params

    logging.info(
        f"HPWFitTask: inferred quantum numbers for '{channel_str}': "
        f"twoS={twoS}, L={L} ({_L_TO_WAVE_LABEL.get(L,'?')}-wave), twoJ={twoJ} → {wave_str}"
    )
    return qn_channel


def _find_hpw_hdf5(task_params, proj_handler):
    """Return the HDF5 input path.

    Priority:
      1. Explicit ``data_file`` in task_params.
      2. Most-recently-modified .hdf5/.h5 in the fit_spectrum data directory.
      3. Most-recently-modified .hdf5/.h5 anywhere under the project root.
    """
    if task_params and task_params.get('data_file'):
        return task_params['data_file']

    # Try fit_spectrum task data directory first
    for task_name in ('fit_spectrum', 'single_channel_fit'):
        handler = proj_handler.all_tasks.get(task_name)
        if handler:
            search_root = handler.data_dir()
            candidates = (
                glob.glob(os.path.join(search_root, '**', '*.hdf5'), recursive=True)
                + glob.glob(os.path.join(search_root, '**', '*.h5'), recursive=True)
            )
            if candidates:
                candidates.sort(key=os.path.getmtime, reverse=True)
                logging.info(f"HPWFitTask: auto-discovered HDF5 from '{task_name}': {candidates[0]}")
                return candidates[0]

    # Fall back to project root
    candidates = (
        glob.glob(os.path.join(proj_handler.root, '**', '*.hdf5'), recursive=True)
        + glob.glob(os.path.join(proj_handler.root, '**', '*.h5'), recursive=True)
    )
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        logging.info(f"HPWFitTask: auto-discovered HDF5 from project root: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        "HPWFitTask: could not find an HDF5 input file. "
        "Add 'data_file' to the task params or ensure a previous task (fit_spectrum) "
        "has written an HDF5 file to the project data directory."
    )


def _build_data_block(task_params):
    """Convert the YAML 'scattering' dict into a GenericFitRunner 'data' list.

    The scattering format is:
        scattering:
          'channel_name':
            PSQ0:
              - IrrepName: [level_indices, ...]
            PSQ1:
              - IrrepName: [level_indices, ...]

    Returns a list like:
        [{"momentum": [0,0,0], "irreps": {"G1u": [0,1]}}, ...]
    one entry per PSQ with all irreps merged across channels.
    """
    scattering = task_params.get('scattering') if task_params else None
    if not scattering:
        return None

    # Merge irreps/levels from all channels, keyed by PSQ label
    psq_merged = {}  # PSQ_label -> {irrep: [levels]}
    for _channel, psq_dict in scattering.items():
        for psq_label, irrep_list in psq_dict.items():
            if psq_label not in psq_merged:
                psq_merged[psq_label] = {}
            # irrep_list is a list containing one dict: [{IrrepName: [levels]}]
            for irrep_entry in irrep_list:
                if isinstance(irrep_entry, dict):
                    for irrep_name, levels in irrep_entry.items():
                        if irrep_name not in psq_merged[psq_label]:
                            psq_merged[psq_label][irrep_name] = []
                        for lvl in levels:
                            if lvl not in psq_merged[psq_label][irrep_name]:
                                psq_merged[psq_label][irrep_name].append(lvl)

    # Sort by PSQ so the data block is in ascending momentum order
    data_block = []
    for psq_label in sorted(psq_merged.keys()):
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
    # ── HDF5 file ──────────────────────────────────────────────────────────
    h5_file = _find_hpw_hdf5(task_params, proj_handler)

    # ── Lattice size from sigmond ensemble info ────────────────────────────
    lattice_size = int(task_params.get('lattice_size', 48))
    try:
        import fvspectrum.sigmond_util as sigmond_util
        ensemble_info = sigmond_util.get_ensemble_info(general_configs)
        lattice_size = int(ensemble_info.getLatticeXExtent())
    except Exception:
        pass  # fall back to task_params value or default

    # ── Data block from scattering YAML ───────────────────────────────────
    data_block = _build_data_block(task_params)

    # ── Quantum numbers: explicit block wins; otherwise infer from particles.py ─
    qn_block = task_params.get('quantum_numbers', None) if task_params else None
    if qn_block is None and scattering:
        inferred_channels = []
        for channel_str in scattering.keys():
            ch = _infer_channel_qn(channel_str, task_params)
            if ch:
                inferred_channels.append(ch)
        qn_block = {"channels": inferred_channels} if inferred_channels else {}
    elif qn_block is None:
        qn_block = {}

    # ── Output paths via project directory handler ─────────────────────────
    plot_dir = proj_handler.plot_dir()
    log_dir  = proj_handler.log_dir()

    config = {
        "study_module": task_params.get('study_module', 'QC2.my_fit_model'),
        "input": {
            "file":         h5_file,
            "lattice_size": lattice_size,
            "continuum":    bool(task_params.get('continuum', True)),
            "cutoff":       float(task_params.get('cutoff', 10.0)),
        },
        "quantum_numbers": qn_block,
        "fit": {
            "MN":                float(task_params.get('MN', 1.0)),
            "MK":                float(task_params.get('MK', 1.0)),
            "strategy":          task_params.get('strategy', 'nelder-mead'),
            "n_starts":          int(task_params.get('n_starts', 20)),
            "n_refine":          int(task_params.get('n_refine', 400)),
            "step_mode":         task_params.get('step_mode', 'adaptive'),
            "xatol":             float(task_params.get('xatol', 1e-6)),
            "fatol":             float(task_params.get('fatol', 1e-6)),
            "n_maxiter":         int(task_params.get('n_maxiter', 10000)),
            "auto_p0":           bool(task_params.get('auto_p0', False)),
            "skip_minimization": bool(task_params.get('skip_minimization', False)),
        },
        "plot": {
            "enabled":            True,
            "save_path":          os.path.join(plot_dir, "hpw_fit_spectrum.pdf"),
            "save_timestamp":     True,
            "com_frame":          True,
            "show":               False,
            "ylabel":             r"$E^\star / m_N$",
            "show_ni":            True,
            "show_level_labels":  True,
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

    Supports two modes (see module ``doc`` string for YAML examples):

    1. **config_file mode**: provide ``config_file`` in task_params pointing to a
       pre-made GenericFitRunner JSON.  The JSON is used as-is.

    2. **auto-config mode**: omit ``config_file`` and instead provide ``data_file``
       (or rely on auto-discovery from the project directory) plus ``scattering``
       and optionally ``quantum_numbers``.  A JSON config is generated
       automatically and saved to the task log directory.
    """

    @property
    def info(self):
        return doc

    def __init__(self, task_name, proj_handler, general_configs, task_params):
        self.task_name = task_name
        self.proj_handler = proj_handler

        if task_params and 'config_file' in task_params:
            # ── Mode 1: pre-made JSON ──────────────────────────────────────
            config_path = task_params['config_file']
            logging.info(f"HPWFitTask: using provided config_file: {config_path}")
        else:
            # ── Mode 2: auto-generate JSON from task params + project dir ──
            logging.info("HPWFitTask: no config_file provided – auto-generating JSON config.")
            config_dict = _build_hpw_config(task_params or {}, proj_handler, general_configs)
            config_path = os.path.join(proj_handler.log_dir(), 'hpw_fit_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            logging.info(f"HPWFitTask: auto-generated config written to {config_path}")

        study_module      = (task_params or {}).get('study_module', None)
        bmat_preview_only = bool((task_params or {}).get('bmat_preview_only', False))

        self._runner = GenericFitRunner(
            config_path,
            study_module_override=study_module,
            bmat_preview_only=bmat_preview_only,
        )

    def run(self):
        """Run the HPW fit (GenericFitRunner handles internal plotting)."""
        self._runner.run()

    def plot(self):
        """Plotting is handled inside run(); intentional no-op."""
        pass


if __name__ == "__main__":
    main()