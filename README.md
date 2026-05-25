# PyCALQ
Correlator Analysis and Luscher Quantization Condition.

Full analysis chain of the finite volume spectrum from two-point correlators to phase-shifts and other infinite-volume observables using the Lüscher formalism for two-particle scattering channels below the nearest inelastic scattering threshold.

## Prerequisites

[sigmond pybindings (pip branch)](https://github.com/andrewhanlon/sigmond/tree/pip)

## Setup
```
cd PyCALQ/
pip install -r requirements.txt
```

## Sample usage

```
(base) PS C:\cygwin64\home\sarah\lqcd\luscher-scmuscher> run.py -h
usage: run.py [-h] [-g GENERAL] [-t TASKS [TASKS ...]]

options:
  -h, --help            show this help message and exit
  -g GENERAL, --general GENERAL
                        general configuration file
  -t TASKS [TASKS ...], --tasks TASKS [TASKS ...]
                        task(s) configuration file(s)
```

## Tasks

Any task will require a general configuration file. At minimum the general configurations file should
contain:
```
general:
    project_dir: /path/to/directory/
    ensemble_id: cls21_s64_t128_D200
```

More general parameters can be specified:
```
general:
    project_dir: /path/to/directory/
    ensemble_id: cls21_s64_t128_D200
    sampling_info:                      #not required
      mode: Bootstrap #or Jackknife     #default: Jackknife
      number_resampling: 800            #required for Bootstrap
      seed: 3103                        #required for Bootstrap
      boot_skip: 297                    #required for Bootstrap
    tweak_ensemble:                     #not required
      omissions: [2000]                 #default []
      rebin: 10                         #default 1
```

Short descriptions of general configs:
- project directory - (str) new or existing directory of the project
- ensemble_id - (str) the id associated with ensemble infos that are iterated in `fvspectrum/sigmond_utils/ensembles.xml`
- sampling_info - (dict) basic information for how the Monte Carlo samples are to be resampled
    - mode - (str) 'Bootstrap' or 'Jackknife', determines the resampling method
    - number_resampling - (int) determined the number of bootstrap samples
    - seed - (int) determined the seed for the bootstrap random numbers
    - boot_skip - (int) determines how many random numbers to skip before using to determine each sampling index
- tweak_ensemble - (dict) how the original sample set will be modified
    - omissions - (list) list of indexes of the original sample set to be omitted from any calculations
    - rebin - (int) block size of samplings to average over before being resampled

To peform an analysis task, one or more task configurations files must be included. 
Task configuration files follow the structure:
```
tasks:
  - [task1]:
      [task1_configs]
  - [task2]:
      [task2_configs]
  - ...
```
where `[taskn]` should be replaced with one of the following:
- `preview_corrs`
- `average_corrs`
- `rotate_corrs`
- `fit_spectrum`
- `compare_spectrums`
- `single_channel_fit`
and `[taskn_configs]` should be replaced with the required configurations for each task (described below).
Each tasks functions and configuration input are outlined in the following sections. Tasks may be in any order
and multiple of any task, but the input will be sorted according to the list above. If multiple are included, 
then the items of the same task will be ran in the order defined.

Some task config inputs are included in all tasks:
```
  figheight: 6                  #not required #default 6
  figwidth: 8                   #not required #default 8
  info: true                    #not required #default false
  plot: true                    #not required #default true
```

Short descriptions of universal task inputs:
- figheight - (int) sets figure height of matplotlib plots
- figwidth - (int) sets figure width of matplotlib plots
- info - (bool) if true prints out configuration info for the given task
- plot - (bool) if false, no plots are generated

Common task inputs include:
```
    create_pdfs: true             #not required #default true
    create_pickles: true          #not required #default true
    create_summary: true          #not required #default true
    generate_estimates: true              #not required #default true
    raw_data_files:               #required 
    - /latticeQCD/raid3/ahanlon/data/cls21_c103/updated_stats/sigmond.fwd/cls21_c103/nucleon_S0.bin
```

Short descriptions of common task inputs:
- averaged_input_correlators_dir - (list or str) location of averaged data, can be directory(ies) or file(s)
                        if no file is provided, will search the project directory for averaged data
- create_pdfs - (bool) if true, generates matplotlib pdf plots
- create_pickles - (bool) if true, generates matplotlib pickle files
- create_summary - (bool) if true, generated latex pdf summary of all plots
                        and some calclation output
- generate_estimates - (bool) if true, generates csvs of bootstrap or jackknife estimates
- raw_data_files - (str or list) files or directories where raw correlator data is stored, this 
                    data is presumed to be unaveraged over momentum or irrep row
- reference_particle - (str) nickname of particle for reference fit. User can define this name in 'single_hadrons' input parameter in the 'fit_spectrum' task
- tmin - (int) minimum correlator time that the task will performed
- tmax - (int) maximum correlator time that the task will performed

### Preview Correlators
A task to read in and estimate/plot any Lattice QCD temporal correlator data files given.

Task Input
```
- preview_corrs:                  #required
    raw_data_files:               #required 
    - /latticeQCD/raid3/ahanlon/data/cls21_c103/updated_stats/sigmond.fwd/cls21_c103/nucleon_S0.bin
    create_pdfs: true             #not required #default true
    create_pickles: true          #not required #default true
    create_summary: true          #not required #default true
    figheight: 6                  #not required #default 6
    figwidth: 8                   #not required #default 8
    info: true                    #not required #default false
    plot: true                    #not required #default true
    generate_estimates: true              #not required #default true
```

See universal and common task descriptions for information regarding these inputs. 

### Average Correlators
A task to read in and automatically average over any Lattice QCD temporal correlator data files 
                given within the same irrep row and total momentum.

All task input:
```
- average_corrs:                          #required
    raw_data_files:                       #required
    - /latticeQCD/raid3/ahanlon/data/cls21_c103/updated_stats/sigmond.fwd/cls21_c103/nucleon_S0.bin
    average_by_bins: false                #not required #default false
    average_hadron_irrep_info: true       #not required #default true
    average_hadron_spatial_info: true     #not required #default true
    create_pdfs: true                     #not required #default true
    create_pickles: true                    #not required #default true
    create_summary: true                  #not required #default true
    erase_original_matrix_from_memory: false #not required #default false
    figheight: 6                          #not required #default 6
    figwidth: 8                           #not required #default 8
    generate_estimates: true              #not required #default true
    ignore_missing_correlators: true      #not required #default true
    plot: true                            #not required #default true
    separate_mom: true                    #not required #default false
    tmax: 64                              #not required #default 64
    tmin: 0                               #not required #default 0
```

Short descriptions of unique task inputs:
- average_by_bins - (bool) if true, average correlators bin by bin. If false, average using resampling
- average_hadron_irrep_info - (bool) if true, average over hadrons with different irreps (not correlators)
- average_hadron_spatial_info - (bool) if true, average over irreps with different spatial configurations
- erase_original_matrix_from_memory - (bool) if true, saves memory in the sigmond calculation by erasing
                                        the original matrices from memory once added to the sum for averaging
- ignore_missing_correlators - (bool) if true, alert and end program if correlators are missing from a given correlator matrix
- run_tag - (str) user defined tag to add to resulting data file names
- separate_mom - (bool) if true, separates final data and summaries by momentum

See universal and common task input description for info regarding all other inputs. 

### Rotate Correlators
A task to pivot a given correlator matrix and return the time-dependent eigenvalues.

All task input:
```
- rotate_corrs:                           #required
    tN: 5                                 #required
    t0: 5                                 #required
    tD: 10                                #required
    averaged_input_correlators_dir: {project_dir}/1average_corrs/data/bins #not required #default {project_dir}/1average_corrs/data/bins 
    create_pdfs: true                     #not required #default true
    create_pickles: true                  #not required #default true
    create_summary: true                  #not required #default true
    figheight: 6                          #not required #default 6
    figwidth: 8                           #not required #default 8
    generate_estimates: true              #not required #default true
    max_condition_number: 50              #not required #default 50
    only:                                 #not required
    - psq=0
    - isosinglet S=0 E PSQ=3
    omit:                                 #not required (overridden by 'only' setting)
    - psq=0
    - isosinglet S=0 E PSQ=3
    omit_operators: []                    #not required #default []
    pivot_type: 0                         #not required #default 0; 0 - single pivot, 1 - rolling pivot
    plot: true                            #not required #default true
    precompute: true                      #not required #default true
    rotate_by_samplings: true             #not required #default true; otherwise rotate by bins
    run_tag: "unique"                     #not required #default ""
    tmax: 25                              #not required #default 25
    tmin: 2                               #not required #default 2
    used_averaged_bins: true              #not required #default true
```

Unique task input descriptions:
- t0 - (int) metric time for pivots
- tN - (int) normalize time for pivots
- tD - (int) diagonalize time for pivots
- pivot_type - (int) determines pivot type
- max_condition_number - (float) maximum condition number for pivots, (max eigenvalue)/(min eigenvalue)
- only - (list) list of channels to include. Can be of form "psq=0" for all channels of momentum = 0, and 
                    "isosinglet S=0 E PSQ=3" for a given channel. Overrides 'omit' input
- omit - (list) list of channels to omit. Can be of form "psq=0" for all channels of momentum = 0, and 
                    "isosinglet S=0 E PSQ=3" for a given channel. Ignored if 'only' input is present.
- omit_operators - (list) list of operators to omit from their channel's pivot
- precompute - (bool) Specifies whether the bootstrap samples should be precomputed (within sigmond)
- rotate_by_samplings - (bool) if true, generates samples and the rotates by samples. If false, rotates by bins
- used_averaged_bins - (bool) if true, if the program must search the project directory for averaged data, then
                        will look for bins files. If false looks for averaged sampling files

### Fit Correlators
A task for fitting the single hadron and/or rotated correlators in order to determine
    energy spectrum of each channel of interest. If rotated, operator overlaps on the original
    operators are computed and plotted as well.

Task Input:
```
- fit_spectrum:
    default_corr_fit:                     #required unless both default_interacting_corr_fit and default_noninteracting_corr_fit are specified
        model: 1-exp                        #required
        tmin: 15                            #required
        tmax: 25                            #required
        exclude_times: []                   #not required #default []
        initial_params: {}                  #not required #default {}, but if specified, should be a dictionary of param name: value
        noise_cutoff: 0.0                   #not required #default 0
        priors: {}                          #not required #default {}, but if specified, should be a dictionary of param name: {"Mean": mean value,"Error": width value}
        ratio: true                         #not required #default false
        sim_fit: false                      #not required #default false
        tmin_plots:                         #not required #default []
        - model: 1-exp                        #required
          tmin_min: 10                        #required
          tmin_max: 20                        #required
        ...
        tmax_plots:                         #not required #default []
        - model: 1-exp                        #required
          tmax_min: 30                        #required
          tmax_max: 40                        #required
        ...
    reference_particle: pi                #not required #default None
    default_noninteracting_corr_fit: None #not required, but set up is same as default_corr_fit
    default_interacting_corr_fit: None    #not required, but set up is same as default_corr_fit
    correlator_fits:                              #not required #default {}
        operator name:                              #not required, but "operator name" should be replaced with the intended operator 
        tmin: 2                                        #for the fit configuration. Any fit model configuration specified here will
        tmax: 25                                       #override the default
        model: 1-exp 
        ...
    averaged_input_correlators_dir: /some/file    #not required #default is the project directory's average task
    compute_overlaps: true                        #not required #default true
    correlated: true                              #not required #default true
    create_pdfs: true                             #not required #default true
    create_pickles: true                          #not required #default true
    create_summary: true                          #not required #default true
    do_interacting_fits: true                     #not required #default true
    figheight: 6                          #not required #default 6
    figwidth: 8                           #not required #default 8
    only:                                 #not required
    - psq=0
    - isosinglet S=0 E PSQ=3
    ...
    omit:                                 #not required (overridden by 'only' setting)
    - psq=0
    - isosinglet S=0 E PSQ=3
    ...
    generate_estimates: true              #not required #default true
    tN: 5                                 #not required #defualt finds most recently used file
    t0: 5                                 #not required #defualt finds most recently used file
    tD: 10                                #not required #defualt finds most recently used file
    pivot_type: 0                         #not required #defualt finds most recently used file; 0 - single pivot, 1 - rolling pivot
    minimizer_info:                       #not required #defaults below
        chisquare_rel_tol: 0.0001             
        max_iterations: 2000
        minimizer: lmder
        parameter_rel_tol: 1.0e-06
        verbosity: low
    single_hadrons:                       #required for ratio fits. "sh1" and "operator name" should be replaced with
        sh1:                                  #user given single hadron name and relevant operator names in a list ordered by
        - operator name                       #increasing integer momenta
        ...
    single_hadrons_ratio: []              #not required #default [] #set up is like single hadrons, overrides 
                                            #single_hadrons for correlator division for ratio fit (but nothing else)
    non_interacting_levels:               #not required except for ratio fits, needs single_hadrons specified to function
        channel:                          #"channel" should be replaced with channel name
        - [sh1(d1^2), sh2(d1^2)]              #"sh1" and "sh2" should be replaced with single hadron names specified above
        ...                                   #"d1^2" and "d2^2" should be replaced with the integer total momentum of the single hadron
    pivot_file: /some/file                #not required #automatically taken from project if not given
    plot: true                            #not required #default true
    precompute: true                      #not required #default true
    rotated_input_correlators_dir:        #not required #automatically taken from project if not given
    run_tag: ""                           #not required #default "" 
    rotate_run_tag: ""                    #not required #default "" #should correspond to rotate run_tag
    thresholds:                           #not required #default [] #replace "sh1" and "sh2" with user given names
    - [sh1, sh2]
    ...
    use_rotated_samplings: true            #not required #default true => broken
    used_averaged_bins: true               #not required #default true
```

Unique task input descriptions:
- default_corr_fit - (dict) settings for default correlator fit that would be applied to any correlator given
  - model - (str) short string representing model to use for the correlator fit. List of short names can be found [here](https://github.com/andrewhanlon/sigmond_scripts/blob/pip/src/sigmond_scripts/fit_info.py).
  - tmin - (int) min time for correlator fit
  - tmax - (int) max time for correlator fit
  - exclude_times - (list) correlator time values to exclude in the fit
  - initial_params - (dict) set initial parameter values for correlator fit (not actually done). Expect a dict of structure {[(str) parameter name]: [(float) value]}. List of parameter names can be found [here](https://github.com/andrewhanlon/sigmond_scripts/blob/pip/src/sigmond_scripts/fit_info.py).
  - noise_cutoff - (float) if nonzero, fit will exclude any data where (error)/(value) > noise_cutoff
  - priors - (dict) set prior values for parameters of the fit. Expect a dict of structure {[(str) parameter name]: {"Mean":[(float) value],"Error":[(float) value]}}. List of parameter names can be found [here](https://github.com/andrewhanlon/sigmond_scripts/blob/pip/src/sigmond_scripts/fit_info.py).
  - ratio - (bool) if true, will use the input parameter 'non_interacting_levels' to pick single hadrons for the denominator of the new ratio correlator to fit. Ratio correlator will be of the form C(t)/SH1(t)/SH2(t) where SHN(t) indicates the single hadron correlators.
  - sim_fit - (bool) if true, will perform a sim-fit with the single hadron correlators indicated by the 'non_interacting_levels' input parameters. Only set up for 2-exp sim fit. The single hadron fits must also be 2-exp.
  - tmin_plots - (list) list of tmin plots to compute
    - model - (str) same as above
    - tmin_min - (int) min time for tmin values to fit
    - tmin_max - (int) max time for tmin values to fit
  - tmax_plot - (list) list of tmin plots to compute
    - model - (str) same as above
    - tmax_min - (int) min time for tmax values to fit
    - tmax_max - (int) max time for tmax values to fit
- default_noninteracting_corr_fit - (dict) default fit for non-interacting correlators, same structure as 'default_corr_fit' input parameter
- default_interacting_corr_fit - (dict) default fit for interacting correlators, same structure as 'default_corr_fit' input parameter
- correlator_fits - (dict) individual fit specifiers. Expects a dict of form {[(str) operator name]: [(dict) fit params]} where 'fit params' is a dict of 'default_corr_fit' parameters. Any fit parameter specified there will override the default fit specifically for correlator with 'operator name'. 'operator name' should look like "isodoublet S=-1 PSQ=1 A2 k[SS1] 0" or "isosinglet S=-1 P=(0,0,0) G1g ROT 1"
- compute_overlaps - (bool) if true, computes the operator overlaps will be computed and possibly plotted. If false, no calculations are made.
- correlated - (bool) if true, uses correlated fits. If false, does uncorrelated fits. 
- do_interacting_fits - (bool) if true, does interacting correlator fits. If false, skips those fits
- rotate_run_tag - (str) user-defined unique tag associated with input rotated data files. Not combined to spectrum run_tag
- minimizer_info - (dict) determines the minimizer to use and various minimization settings
  - chisquare_rel_tol - (float) determines the precicion to which the chi square is calculated        
  - max_iterations - (int) determines the max number of iterations the minimizer performs
  - minimizer - (str) determines which minimizer to use, available options are 'lmder' (sigmond) or 'scipy' (python)
  - parameter_rel_tol - (float) determines the precision that the parameters are computed to
  - verbosity - (str) determines how much output the fitter will produce in log file (broken?)

### Compare Spectrums
Using the fit results of the fit_spectrum task, plots
    the spectrums side by side for comparisons and puts in summary document. Work in progress.

Task Input:
```
- compare_spectrums:              #required
    compare_plots:                #required # list of comparison plot types
    - compare_gevp:               #not required
        gevp_values:              #required for "compare_gevp", list of pivot configs
        - t0: 8                       #required
            tD: 16                      #required
            tN: 5                       #required
            pivot_type: 0               #not required #default: 0
        - t0: 8
            tD: 18
            tN: 5
        ...
        rebin: 1                      #not required
        run_tag: ''                   #not required #default: ''
        sampling_mode: J              #not required
    - compare_files: []               #not required #default []
    - compare_rebin:                  #not required
        rebin_values: []              #required 
        run_tag: ''                   #not required #default: ''
        sampling_mode: J              #not required
        pivot_type: 0               #not required #default: 0
        t0: 8                         #required 
        tN: 5                         #required 
        tD: 18                        #required 
    - compare_tags:
        filetags: []                  #required
        sampling_mode: J              #not required
        pivot_type: 0               #not required #default: 0
        t0: 8                         #required 
        tN: 5                         #required 
        tD: 18                        #required 
        rebin: 1                      #not required
    figheight: 8                      #not required #default: 8
    figwidth: 15                      #not required #default: 15
    plot: true                        #required
    plot_deltaE: true                 #not required #default: True
    reference_particle: P             #not required #default: None
```

Unique task input descriptions:
- compare_plots - (list) all plots that are desired. plot types include
  - compare_gevp - (dict) spectrum plot for comparing different pivots
  - compare_files - (dict) spectrum plot where different data files are explicitly named in form of {[label], [key]}
  - compare_rebin - (dict) spectrum plot for comparing different rebinning schemes
  - compare_tags - (dict) spectrum plot for comparing different user-defined filetags
  Other than 'compare_files', each setup should contain the base parameters so the code knows which files to look for:
  - sampling_mode - (char) 'B' or 'J' for bootstrap or jackknife resampling methods
  - pivot_type - (int) index for pivot scheme
  - t0 - (int) metric time for pivot
  - tN - (int) normalize time for pivot
  - tD - (int) diagonalize time for pivot
  - rebin - (int) value for blocksize when rebinning
  - run_tag - (str) user defined unique tag associated with analysis
  For each of the plots, a set of parameters will be replaced with a list of that paramer set
  - compare_gevp - 'gevp_values' will replace 'pivot_type', 't0', 'tN', and 'tD' with a list of dict with those items
  - compare_rebin - 'rebin_values' will replace 'rebin' for a list of rebin values
  - compare_tags - 'filetags' will replace 'run_tag' for a list of runtags
- plot_deltaE - include additional plots that compare shifts from non-interacting levels. Only possible if provided in data

### Single Channel Fit
Using the finite-volume spectrum produced by `fit_spectrum` (or any compatible HDF5 file), fits a
K-matrix parametrization to the Lüscher quantization condition using the HPW B-matrix (BMAT) method.
The fit is driven by `QC2/run_HPW_fit.py` (`GenericFitRunner`) and the study module `QC2/my_fit_model.py`.

Two usage modes are supported.

#### Mode 1 – Auto-config (recommended)
All settings live in the tasks YAML. A GenericFitRunner JSON config is generated automatically
and saved to the task log directory. The HDF5 input file is found automatically from the
`fit_spectrum` data directory if `data_file` is omitted.

Quantum numbers (`twoJ`, `twoS`) are derived from the particle names in `general/particles.py`
using the `scattering` channel string. The partial wave `L` defaults to `0` (S-wave) and can
be overridden. Only the K-matrix initial parameter guesses (`params`) must be provided.

```yaml
- single_channel_fit:
    data_file: /path/to/spectrum.hdf5   # omit to auto-discover from fit_spectrum output

    scattering:
      'N(0)_ref,pi(0)_ref':             # two hadrons separated by comma
        PSQ0:
          - G1u: [0, 1, 2]             # irrep: list of level indices
        PSQ1:
          - G1: [0, 1]
        PSQ2:
          - G: [0, 1]
        PSQ3:
          - G: [0, 1]

    # twoS is auto-derived from particles.py; twoJ = |2L - twoS| by default
    L: 0                # partial wave integer (0=S, 1=P, 2=D, ...)
    k_matrix: polynomial  # polynomial | ERE | zero
    params:             # K-matrix initial guesses (required)
      - {name: c0, initial: -0.2}
      - {name: c1, initial: 0.7}

    # Optional overrides
    # twoS: 1
    # twoJ: 1

    # Fit settings (all optional – defaults shown)
    MN: 1.0
    MK: 1.0
    strategy: nelder-mead  # nelder-mead | powell | ...
    n_starts: 20
    n_refine: 400
    auto_p0: false         # estimate initial params from B-matrix before fit
    cutoff: 10.0           # discard levels with E_cm/m_N above this value
    study_module: QC2.my_fit_model
    bmat_preview_only: false  # true = only plot B-matrix preview, skip fit
```

Short descriptions of unique task inputs:
- `data_file` - (str) path to the HDF5 spectrum file. Auto-discovered from the `fit_spectrum`
  data directory if omitted.
- `scattering` - (dict) channel → PSQ → list of `{irrep: [level_indices]}`. Drives both the
  data selection and the quantum-number inference.
- `L` - (int or str) orbital angular momentum of the partial wave: `0`/`'S'`, `1`/`'P'`, `2`/`'D'`, …
- `k_matrix` - (str) K-matrix parametrization type for the channel.
- `params` - (list) K-matrix initial parameter guesses, each as `{name: <str>, initial: <float>}`.
- `twoS` - (int) override auto-derived total spin (2×S). Needed when multiple spin states are
  possible (e.g. NN).
- `twoJ` - (int) override auto-derived total angular momentum (2×J).
- `MN`, `MK` - (float) reference masses for nucleon and kaon in units of `m_ref`.
- `strategy` - (str) optimizer strategy passed to `scipy.optimize.minimize`.
- `n_starts` - (int) number of random restarts for the minimizer.
- `n_refine` - (int) grid resolution for root-finding.
- `auto_p0` - (bool) if true, estimates initial parameters from the B-matrix before minimizing.
- `cutoff` - (float) energy cutoff; levels above this are excluded from the fit.
- `study_module` - (str) Python import path for the fit study module.
- `bmat_preview_only` - (bool) if true, produces only the B-matrix preview plot and skips the fit.

#### Mode 2 – Pre-made JSON config
For full control, or when reusing a config from a previous standalone run, point to a
pre-made GenericFitRunner JSON file:

```yaml
- single_channel_fit:
    config_file: test_configs/c103_hpw_fit.json
    study_module: QC2.my_fit_model   # optional override of value in JSON
    bmat_preview_only: false
```

See `test_configs/c103_hpw_fit.json` for a fully annotated example of the JSON format.

#### Standalone usage (no PyCALQ pipeline)
`run_HPW_fit.py` can also be run directly from the command line with only a JSON config:

```bash
# Full fit
python QC2/run_HPW_fit.py test_configs/c103_hpw_fit.json

# B-matrix preview only (no minimization)
python QC2/run_HPW_fit.py test_configs/c103_hpw_fit.json --bmat-preview-only

# Override the study module
python QC2/run_HPW_fit.py test_configs/c103_hpw_fit.json --module QC2.my_fit_model
```



## Setting up a New Task
In order to create additional tasks, one should set up a task using the folowing skeleton:
```
import logging

doc = '''
essential documentation
'''

class MyTaskName:
    @property
    def info(self):
        return doc

    def __init__( self, task_name, proj_files_handler, general_configs, task_configs ):
        self.proj_files_handler= proj_files_handler
        #initialize your task, store default input in self.proj_files_handler.log_dir() (basically, throw the full possible input with all parameters where all the assumed parameters have been filled in in there)

    def run( self ):
        pass
        # do the task, produce the data, data goes in self.proj_files_handler.data_dir(), info/warning/errors about the process goes in self.proj_files_handler.log_dir() (if any)

    def plot( self ):
        pass
        # make the plots, store in self.proj_files_handler.plot_dir(), again, any log/error warnings go in self.proj_files_handler.log_dir() as well (if any)
```

Once the new class is set up. Add the class to task manager list. Note, if numbers on current tasks are modified then PyCALQ will not know where that data is anymore. Be sure to modify the filenames in the project or rerun.

Example of adding a new task to `task_manager.py`:
```
class Task(Enum): #encode tasks into enum
    preview_corrs = 0
    average_corrs = 1
    rotate_corrs = 2
    fit_spectrum = 3
    toy_corrs = 4
    compare_spectrums = 5
    new_task = 6
```

Then, open `pycalq.py` and add the task to `DEFAULT_TASKS`, `TASK_MAP`, and `TASK_DOC` in the same manner as the current tasks. If the task uses sigmond mcobshandler to manage data and memory, add the task to `SIGMOND_TASKS` and update the `dependencies` and `raw_data_dependence` variables at the top of `fvspectrum/sigmond_project_handler.py` accordingly.


## To Do

Items that need to be fixed:
 - Spectrum task: the estimates for interacting and noninteracting need to be separated because they occasionally have the same channel name. 
 - spectrum task: the fits to nonzero momentum single hadrons do not need to be calculated unless calculating the operator overlaps. if operator overlaps are turned off, then do not calculate these fits. 

Desired updates:
 - internal setup for slurm or other scheduler systems
