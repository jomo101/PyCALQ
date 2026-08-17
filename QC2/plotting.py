import math
import cmath
#import jax
#import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import FancyBboxPatch
import scipy
#import seaborn as sns
import time
import itertools

from scipy import optimize    
from scipy.optimize import root_scalar
from scipy.optimize import brentq        
from scipy import integrate 
from scipy.optimize import fsolve
from scipy.optimize import minimize
#from sympy import *

import os

twoPi = 2.0 * np.pi


def apply_plot_style(use_tex=True):
    """Apply a consistent plotting style for all figures in this workspace.

    With use_tex, text is compiled by LaTeX in its default Computer Modern —
    the same font as the notes/paper documents — instead of Times (naming
    "Times" in font.serif makes matplotlib's TeX backend load the times
    package). The CM entries below also steer the non-TeX mathtext fallback.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "figure.titlesize": 18,
        "figure.labelsize": 16,
        "axes.titlepad": 15,
        "text.usetex": bool(use_tex),
        "lines.linewidth": 1.5,
        "lines.markersize": 8,
    })


_set_plot_style = apply_plot_style


apply_plot_style(use_tex=True)
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
def add_rounded_rect(ax, x_left, y_bottom, width, height,
                     radius_px=5, facecolor='#404040', alpha=0.2,
                     edgecolor='none', linewidth=0.0, zorder=0):
    """
    Draw a rounded rectangle on `ax` with corner radius specified in pixels,
    so it looks correct regardless of axis data scaling.

    Parameters
    ----------
    ax         : matplotlib Axes
    x_left     : left edge in data coordinates
    y_bottom   : bottom edge in data coordinates
    width      : width in data coordinates
    height     : height in data coordinates
    radius_px  : corner radius in pixels (display units)
    facecolor  : fill color
    alpha      : transparency
    edgecolor  : edge color
    linewidth  : edge line width
    zorder     : drawing order
    """
    fig = ax.get_figure()
    trans = ax.transData

    x_right = x_left + width
    y_top = y_bottom + height

    dl = trans.transform([
        [x_left,  y_bottom],
        [x_right, y_top],
    ])
    px_l, px_b = dl[0]
    px_r, px_t = dl[1]

    r = min(radius_px, abs(px_r - px_l) / 2, abs(px_t - px_b) / 2)

    # Cubic bezier control point offset for approximating a quarter circle
    k = r * 0.5522847498

    # Path in display (pixel) coordinates
    # Each corner: line arrives at (corner - r), two control points, end at (corner + r)
    verts_display = [
        # Start at bottom-left, just right of corner
        (px_l + r, px_b),
        # Bottom edge to bottom-right corner
        (px_r - r, px_b),
        (px_r - r + k, px_b), (px_r, px_b + r - k), (px_r, px_b + r),
        # Right edge to top-right corner
        (px_r, px_t - r),
        (px_r, px_t - r + k), (px_r - r + k, px_t), (px_r - r, px_t),
        # Top edge to top-left corner
        (px_l + r, px_t),
        (px_l + r - k, px_t), (px_l, px_t - r + k), (px_l, px_t - r),
        # Left edge to bottom-left corner
        (px_l, px_b + r),
        (px_l, px_b + r - k), (px_l + r - k, px_b), (px_l + r, px_b),
        # Close
        (px_l + r, px_b),
    ]

    codes = [
        mpath.Path.MOVETO,
        # Bottom edge + bottom-right corner
        mpath.Path.LINETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        # Right edge + top-right corner
        mpath.Path.LINETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        # Top edge + top-left corner
        mpath.Path.LINETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        # Left edge + bottom-left corner
        mpath.Path.LINETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        # Close
        mpath.Path.CLOSEPOLY,
    ]

    inv = trans.inverted()
    verts_data = inv.transform(verts_display)

    path = mpath.Path(verts_data, codes)
    patch = mpatches.PathPatch(
        path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        clip_on=True,
    )
    ax.add_patch(patch)
    return patch

def _format_irrep_label(irrep):
    irrep = str(irrep)
    if len(irrep) <= 1:
        return irrep
    return rf"${irrep[0]}_{{{irrep[1:]}}}$"


def _halve_yticks(ax=None, max_ticks=8):
    """Set evenly spaced y-ticks across the visible y-range, at most `max_ticks`."""
    if ax is None:
        ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ticks = np.linspace(ymin, ymax, max_ticks)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

def free_spectrum( m, L, n, nP ):
    m_sq = m * m
    p1_sq = ( twoPi / L )**2 * np.inner(n,n)        # (2*pi/L)^2 * n^2
    p2_sq = ( twoPi / L )**2 * np.inner(nP-n,nP-n)  # (2*pi/L)^2 * (nP - n)^2
    energy1 = np.sqrt( m_sq + p1_sq )
    energy2 = np.sqrt( m_sq + p2_sq )
    return energy1 + energy2


def generate_free_spectrum_lab( m, L, nP, En_max ):
    P_sq = ( twoPi / L )**2 * np.dot(nP,nP)         # (2*pi/L)^2 * nP^2
    n_max = 3.0                                     # max integer for particle momentum
    Energies = []
    intList = np.array(np.arange(-n_max,n_max+1))
    for i in itertools.product(intList,intList,intList):
        n = np.array(i)
        En = free_spectrum(m, L, n, nP)    # moving frame energy
        En = np.round( En, 10 )
        if En not in Energies:
            if En < En_max:
                Energies.append(En)
    Energies.sort()
    return Energies
# generate free spectrum for given L (units of mass), nP, and max energy Ecm_max (units of mass)
def generate_free_spectrum( m, L, nP, Ecm_max ):
    P_sq = ( twoPi / L )**2 * np.dot(nP,nP)         # (2*pi/L)^2 * nP^2
    n_max = 3.0                                     # max integer for particle momentum
    Energies = []
    intList = np.array(np.arange(-n_max,n_max+1))
    for i in itertools.product(intList,intList,intList):
        n = np.array(i)
        En = free_spectrum(m, L, n, nP)    # moving frame energy
        Ecm = np.sqrt( En**2 - P_sq )   # convert to CM frame
        Ecm = np.round( Ecm, 10 )
        if Ecm not in Energies:
            if Ecm < Ecm_max:
                Energies.append(Ecm)
    Energies.sort()
    return Energies

def spectrum_L_data( m , nP, Ecm_max, L):
    """
    Generate free spectrum data for a given L, m, nP, and Ecm_max.
    
    Args:
        m (float): Mass parameter
        nP (array-like): Momentum vector
        Ecm_max (float): Maximum energy in CM frame
        L (float): Lattice size in units of mass
    
    Returns:
        list: List of energies for the given L
    """
    Ecm = generate_free_spectrum(m, L, nP, Ecm_max)
    return Ecm

def get_spectrum_data( m, nP, Ecm_max, L_range ):
    data=[]
    level_data = []
    return_data = []
    max_level, level, L_start = 0, 0, 0
    # grab all energies for given L
    for L in L_range:
        Ecm = generate_free_spectrum(m,  L, nP, Ecm_max )
        if len(Ecm) > max_level:
            max_level = len(Ecm)
        data.append(Ecm )

    # combine L_list with energies
    for _ in range(max_level):
        for i in data:
            if len(i) >= level+1:
                level_data.append(i[level])
            else:
                L_start += 1
        return_data.append([L_range[L_start:],level_data.copy()])
        level += 1
        L_start = 0
        level_data.clear()

    return return_data

def _apply_horizontal_jitter(points, cluster_tol, max_offset):
    """
    Given a list of (x, y, ecm_key, color, error, marker) tuples sharing the same base x,
    nudge x positions so that vertically-clustered points are distinguishable.
    Returns a list of (new_x, y, ecm_key, color, error, marker).
    """
    if len(points) <= 1:
        return [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in points]

    # Sort by y value
    sorted_pts = sorted(points, key=lambda p: p[1])
    n = len(sorted_pts)

    # Group into clusters based on y proximity
    clusters = []
    current_cluster = [0]
    for i in range(1, n):
        if abs(sorted_pts[i][1] - sorted_pts[current_cluster[0]][1]) <= cluster_tol:
            current_cluster.append(i)
        else:
            clusters.append(current_cluster)
            current_cluster = [i]
    clusters.append(current_cluster)

    result = [None] * n
    for cluster in clusters:
        m = len(cluster)
        if m == 1:
            idx = cluster[0]
            p = sorted_pts[idx]
            result[idx] = (p[0], p[1], p[2], p[3], p[4], p[5])
        else:
            # Scale the spread so fewer points cluster closer to center
            # Works well for 5 points (full range), tighter for 2 points (~40% range)
            scale = min(m / 5.0, 1.0)
            scaled_offset = max_offset * scale
            offsets = np.linspace(-scaled_offset, scaled_offset, m)
            for k, idx in enumerate(cluster):
                p = sorted_pts[idx]
                result[idx] = (p[0] + offsets[k], p[1], p[2], p[3], p[4], p[5])

    return result


# ── Centralized point-style registry ──────────────────────────────────────────
_LEVEL_COLORS  = list(plt.cm.tab20.colors)  # 20 distinct colours, indexed by level
_IRREP_MARKERS = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">", "h", "8", "p", "H"]


class PointStyleRegistry:
    """
    Globally consistent colour + marker mapping for spectrum data points.

    - colour : determined solely by level_idx  (0 → tab20[0], 1 → tab20[1], …)
    - marker : determined by (psq_label, irrep), assigned on first-encounter order

    Create one instance and share it across all plot panels in a session so
    that every panel uses identical colour / shape encoding.
    """

    def __init__(self):
        self._marker_map  = {}    # {(psq_str, irrep_str): marker_char}
        self._seen_levels = set()

    def color(self, level_idx):
        self._seen_levels.add(int(level_idx))
        return _LEVEL_COLORS[int(level_idx) % len(_LEVEL_COLORS)]

    def marker(self, psq, irrep):
        key = (str(psq), str(irrep))
        if key not in self._marker_map:
            self._marker_map[key] = "o"#_IRREP_MARKERS[len(self._marker_map) % len(_IRREP_MARKERS)]
        return self._marker_map[key]

    def style(self, level_idx, psq, irrep):
        """Return (color, marker) for a data point."""
        return self.color(level_idx), self.marker(psq, irrep)

    def irrep_legend_handles(self):
        """Build legend handles grouped by PSQ (one column per PSQ, rows = irreps).

        Returns
        -------
        handles : list
        labels  : list
        ncol    : int  — number of columns (= number of distinct PSQs)
        """
        import matplotlib.lines as mlines

        # Group entries by PSQ
        psq_groups = {}   # psq_num_str -> [(irrep, marker), ...]
        for (psq, irrep), mkr in self._marker_map.items():
            psq_num = psq.replace("PSQ", "") if psq.startswith("PSQ") else psq
            psq_groups.setdefault(psq_num, []).append((irrep, mkr))

        if not psq_groups:
            return [], [], 1

        sorted_psqs = sorted(psq_groups.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        n_cols = len(sorted_psqs)
        n_rows = max(len(v) for v in psq_groups.values())
        ms     = plt.rcParams.get('lines.markersize', 8) * 1.1

        # Flat list in *row-major* order so matplotlib fills left→right then down,
        # giving us one PSQ per column.
        grid_handles = [None] * (n_rows * n_cols)
        grid_labels  = [''] * (n_rows * n_cols)

        for col, psq_num in enumerate(sorted_psqs):
            for row, (irrep, mkr) in enumerate(psq_groups[psq_num]):
                fmt_irrep = _format_irrep_label(irrep)
                label = rf"{fmt_irrep}({psq_num})"
                h = mlines.Line2D(
                    [], [],
                    color='dimgray', marker=mkr, linestyle='None',
                    markersize=ms,
                    markerfacecolor='none',
                    markeredgecolor='dimgray', markeredgewidth=1.6,
                    label=label,
                )
                grid_handles[row * n_cols + col] = h
                grid_labels [row * n_cols + col] = label

        # Fill empty grid cells with invisible handles so the grid stays aligned
        _empty = mlines.Line2D([], [], color='none', marker='none',
                               linestyle='None', label='')
        handles = [h if h is not None else _empty for h in grid_handles]
        labels  = [lb for lb in grid_labels]

        return handles, labels, n_cols


def plot_energies_vs_irreps(
        data,
        mN,
        L,
        C_o_M=True,
        save_path=None,
        ylabel=None,
        y_cluster_tol=0.02,
        cluster_max_offset=0.08,
        figsize=None,
        show=False,
        show_ni=True,
        show_level_labels=False,
        style_registry=None,
        title=None,
        predictions=None,
        wave_legend=None,
        ylim=None,
        y_from_data=True,
        y_top_pad=0.08,
        y_bottom_pad=0.05,
        highlight=None,
):
    """
    Publication-quality E*/m_N spectrum plot.

    Parameters
    ----------
    data : dict
        {psq_label: {irrep: {level_idx: [central, boot1, ...]}}}
        as returned by build_energy_cm_dict.
    mN : array-like
        Nucleon mass bootstrap samples (mN[0] = central value).
    L : int or float
        Lattice size.
    C_o_M : bool
        Plot in Centre-of-Mass frame (True) or lab frame.
    save_path : str or None
        File path to save the figure.
    ylabel : str or None
        y-axis label.
    y_cluster_tol : float
        Vertical proximity threshold for horizontal-jitter clustering.
    cluster_max_offset : float
        Maximum horizontal nudge applied to the outermost jittered point.
    figsize : tuple or None
        (width, height) in inches; auto-scaled from column count if None.
    show : bool
        Call plt.show() after plotting.
    show_ni : bool
        Draw non-interacting energy bands.
    show_level_labels : bool
        Annotate each point with its integer level index.  Off by default:
        the indices collide wherever levels cluster and they are not part of
        any published figure.  Turn on only for level-selection diagnostics.
    style_registry : PointStyleRegistry or None
        Shared colour/marker registry.  A fresh one is created if None.
    title : str or None
        Figure title.
    predictions : list of dict or None
        Predicted-spectrum overlays (e.g. from
        ``my_fit_model.predict_energy_cm_dict``).  Each dict has:
          - 'data'  : nested {PSQxx: {irrep: {level_idx: [Ecm_pred]}}} dict
          - 'label' : legend label
          - 'color' : line colour (optional)
          - 'colors': optional nested dict, same shape as 'data', giving a
            per-level colour override — e.g. the dominant-wave colour from the
            QC eigenvector decomposition, matching the phase-shift plot's wave
            colours.  Levels missing from 'colors' fall back to 'color'.
          - 'linestyle', 'linewidth' : optional
        Each predicted level is drawn as a short horizontal tick at the same
        irrep(psq) x-position as the data, with a small per-prediction offset so
        multiple predictions sit side-by-side.  A legend is added when set.
    wave_legend : dict or None
        Optional {label: color} entries appended to the legend as the colour
        key for per-level 'colors' (e.g. {r"$^3S_1$": "C0", ...}).
    ylim : (lo, hi) or None
        Explicit y-axis limits.  Overrides the automatic calculation entirely.
    y_from_data : bool
        When True (default), the automatic y-limits frame the DATA points only,
        so predicted/non-interacting levels above the top data level do not
        inflate the axis (they simply clip).  When False, every drawn value
        (data + predictions) is included, as before.
    y_top_pad, y_bottom_pad : float
        Fractional headroom above / below the framed range (fraction of that
        range).  Ignored when `ylim` is given.
    highlight : list of dict or None
        Ring selected levels to show which ones a fit actually used.  Each
        dict has:
          - 'levels' : {PSQn: {irrep: [level_idx, ...]}} — the selection
          - 'label'  : legend label
          - 'color'  : ring colour
          - 'lw'     : ring linewidth (optional, default 1.7)
        Drawn as an open circle behind the marker, so the level's own colour
        and error bar stay visible.  Overlapping selections ring the same
        point more than once, each at a slightly larger radius.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if style_registry is None:
        style_registry = PointStyleRegistry()

    psq_map = {
        'PSQ0': np.array([0, 0, 0]),
        'PSQ1': np.array([0, 0, 1]),
        'PSQ2': np.array([1, 1, 0]),
        'PSQ3': np.array([1, 1, 1]),
        'PSQ4': np.array([0, 0, 2]),
    }

    irrep_gap = 0.2
    x_mapping = {}
    x_labels = []
    all_yvals = []
    data_yvals = []      # data points only (used to frame the y-axis)
    points_per_xpos = {}
    psq_x_positions_map = {}
    xpos_key = {}        # x_pos -> (psq, irrep), for the `highlight` lookup

    # ── First pass: collect all data points ───────────────────────────────────
    for psq_index, (psq, irreps_dict) in enumerate(data.items()):
        psq_base = psq_index * 0.1  # small base offset per PSQ to ensure different PSQs don't overlap
        psq_x_positions = []

        for irrep_idx, (irrep, levels) in enumerate(irreps_dict.items()):
            irrep_label = f"{_format_irrep_label(irrep)} ({psq[-1]})"
            if irrep_label not in x_mapping:
                x_mapping[irrep_label] = psq_base + len(x_mapping) * irrep_gap
                x_labels.append(irrep_label)

            x_pos = x_mapping[irrep_label]
            psq_x_positions.append(x_pos)
            xpos_key[x_pos] = (psq, irrep)

            if x_pos not in points_per_xpos:
                points_per_xpos[x_pos] = []

            for level_idx, values in levels.items():
                color, marker = style_registry.style(level_idx, psq, irrep)
                central_value = values[0]
                error = np.std(values[1:]) if len(values) > 1 else 0.0
                all_yvals.extend([central_value + error, central_value - error])
                data_yvals.extend([central_value + error, central_value - error])
                points_per_xpos[x_pos].append(
                    (x_pos, central_value, level_idx, color, error, marker)
                )

        if psq_x_positions:
            psq_x_positions_map[psq] = psq_x_positions

    # ── Figure setup ──────────────────────────────────────────────────────────
    n_cols = len(x_labels)
    if figsize is None:
        figsize = (max(4.0, 1.8 + n_cols * 0.7), 6.134)
    fig, ax = plt.subplots(figsize=figsize, dpi=400)

    # ── Second pass: jitter and plot data points ───────────────────────────────
    ms     = plt.rcParams.get('lines.markersize', 8)
    ann_fs = plt.rcParams.get('legend.fontsize', 12) * 0.80

    # `highlight` selections, normalised to {(psq, irrep, level): [ring_idx, ...]}
    _hl_sets = []
    for _h in (highlight or []):
        _members = set()
        for _psq, _irs in (_h.get('levels') or {}).items():
            for _ir, _lvls in _irs.items():
                for _lv in _lvls:
                    _members.add((str(_psq), str(_ir), int(_lv)))
        _hl_sets.append((_members, _h))

    for x_pos, points in points_per_xpos.items():
        jittered = _apply_horizontal_jitter(points, y_cluster_tol, cluster_max_offset)
        _pk = xpos_key.get(x_pos)
        for (jx, y, level_idx, color, error, marker) in jittered:
            # ring first (lower zorder) so the marker and its bar stay on top
            if _pk is not None:
                _r = 0
                for _members, _h in _hl_sets:
                    if (_pk[0], _pk[1], int(level_idx)) not in _members:
                        continue
                    ax.plot(
                        jx, y, marker='o', linestyle='None',
                        markersize=ms * (1.75 + 0.45 * _r),
                        markerfacecolor='none',
                        markeredgecolor=_h.get('color', '#b2182b'),
                        markeredgewidth=_h.get('lw', 1.7),
                        alpha=0.95, zorder=4,
                    )
                    _r += 1
            ax.errorbar(
                jx, y, yerr=error,
                fmt=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=0.8,
                markersize=ms,
                capsize=7, capthick=1.2, elinewidth=0.9,
                alpha=0.90,
                zorder=5,
            )
            if show_level_labels:
                ax.annotate(
                    str(int(level_idx)),
                    xy=(jx, y),
                    xytext=(3, 3), textcoords='offset points',
                    ha='left', va='bottom',
                    color='dimgray',
                    fontsize=ann_fs,
                )

    # ── Predicted-level overlays (fits at fixed params) ───────────────────────
    pred_handles = []
    wave_handles = []      # defined here too: `highlight` can build a legend
                           # with no predictions passed at all
    if predictions:
        import matplotlib.lines as _mlines
        n_pred = len(predictions)
        if n_pred > 1:
            offs = np.linspace(-0.5, 0.5, n_pred) * (irrep_gap * 0.5)
        else:
            offs = [0.0]
        half = irrep_gap / np.pi * 0.9
        for pi, pspec in enumerate(predictions):
            pdata   = pspec.get("data", {})
            pcolor  = pspec.get("color", None)
            pcolors = pspec.get("colors", None)   # per-level (wave) colours
            plabel  = pspec.get("label", f"prediction {pi + 1}")
            pls     = pspec.get("linestyle", "-")
            plw     = pspec.get("linewidth", 2.0)
            for psq, irreps_dict in pdata.items():
                for irrep, levels in irreps_dict.items():
                    irrep_label = f"{_format_irrep_label(irrep)} ({psq[-1]})"
                    if irrep_label not in x_mapping:
                        continue
                    xp = x_mapping[irrep_label] + offs[pi]
                    for level_idx, values in levels.items():
                        if not values:
                            continue
                        yv = float(values[0])
                        if not np.isfinite(yv):
                            continue
                        c = pcolor
                        if pcolors is not None:
                            c = (pcolors.get(psq, {}).get(irrep, {})
                                 .get(level_idx, pcolor))
                        all_yvals.append(yv)
                        ax.hlines(yv, xp - half, xp + half, color=c,
                                  lw=plw, ls=pls, zorder=4, alpha=0.9)
            pred_handles.append(_mlines.Line2D(
                [], [], color=pcolor if pcolors is None else '0.3',
                lw=plw, ls=pls, label=plabel))
        if wave_legend:
            for wlab, wcol in wave_legend.items():
                wave_handles.append(_mlines.Line2D([], [], color=wcol, lw=3.0,
                                                   ls='-', label=wlab))

    # ── Non-interacting energy bands ──────────────────────────────────────────
    if show_ni:
        for psq, x_positions in psq_x_positions_map.items():
            if not x_positions or psq not in psq_map:
                continue
            nP = psq_map[psq]
            all_spectra = []
            for m_sample in mN:
                if C_o_M:
                    spectrum = generate_free_spectrum(m_sample, L, nP, 1.54)
                    all_spectra.append(np.array(spectrum) / m_sample)
                else:
                    spectrum = generate_free_spectrum_lab(m_sample, L, nP, 2.1)
                    all_spectra.append(np.array(spectrum) / m_sample)
            if not all_spectra:
                continue
            max_len = max(len(s) for s in all_spectra)
            spectra_arr = np.full((len(all_spectra), max_len), np.nan)
            for i, s in enumerate(all_spectra):
                spectra_arr[i, :len(s)] = s
            # Only draw NI levels within the data y-range + small margin
            _y_lo = min(all_yvals) - 0.02 if all_yvals else 1.95
            _y_hi = max(all_yvals) + 0.02 if all_yvals else 2.40
            for level in range(spectra_arr.shape[1]):
                level_vals = spectra_arr[:, level]
                level_vals = level_vals[~np.isnan(level_vals)]
                if len(level_vals) == 0:
                    continue
                mean = level_vals[0]
                # Skip NI levels outside the data energy window
                if mean < _y_lo or mean > _y_hi:
                    continue
                std  = ((np.percentile(level_vals[1:], 84) - np.percentile(level_vals[1:], 16)) / 2
                        if len(level_vals) > 1 else 0.0)
                for x_pos in x_positions:
                    x_lo = x_pos - irrep_gap / np.pi
                    x_hi = x_pos + irrep_gap / np.pi
                    ax.hlines(mean, x_lo, x_hi,
                              color='#8c8c8c', linestyle=(0, (5, 2.5)),
                              linewidth=1.9, alpha=0.95, zorder=1)
                    if std > 0:
                        ax.fill_between([x_lo, x_hi], mean - std, mean + std,
                                        color='#404040', alpha=0.15, zorder=0)

    # ── y limits ──────────────────────────────────────────────────────────────
    _ylo = _yhi = None
    if ylim is not None:
        _ylo, _yhi = float(ylim[0]), float(ylim[1])
    else:
        base = data_yvals if (y_from_data and data_yvals) else all_yvals
        if base:
            ymin_d, ymax_d = min(base), max(base)
            yrange = ymax_d - ymin_d or 0.1
            _ylo = ymin_d - y_bottom_pad * yrange
            _yhi = ymax_d + y_top_pad * yrange
    if _ylo is not None:
        ax.set_ylim(_ylo, _yhi)

    # ── Two-particle threshold line ────────────────────────────────────────────
    if all_yvals and min(all_yvals) < 2.0:
        ax.axhline(y=2.0, color='#AAAAAA', linestyle=':', linewidth=1.0, alpha=0.7, zorder=0)
        x_vals = list(x_mapping.values())
        ax.text(
            (min(x_vals) + max(x_vals)) / 2, 2.0,
            r'$2m_N$',
            ha='center', va='top',
            color='#888888',
            fontsize=plt.rcParams.get('legend.fontsize', 12),
        )

    # ── Axes formatting ────────────────────────────────────────────────────────
    tick_fs = plt.rcParams.get('xtick.labelsize', 16)
    ax.set_xticks([x_mapping[lb] for lb in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=tick_fs)
    # y-ticks at 0.05 spacing WITHIN the computed range (a tick above the top
    # would otherwise drag the view up to it)
    if _ylo is not None:
        _t0 = np.ceil(_ylo / 0.05) * 0.05
        yticks = np.arange(_t0, _yhi + 1e-9, 0.05)
    else:
        yticks = np.array([2.00, 2.05, 2.10, 2.15])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.2f}" for t in yticks], rotation=0, ha='right', fontsize=tick_fs)
    #ax.tick_params(axis='y', labelsize=tick_fs * 0.78)

    # ← ADD THESE TWO LINES to kill the horizontal dead space:
    x_vals = list(x_mapping.values())
    ax.set_xlim(min(x_vals) - irrep_gap * 0.75, max(x_vals) + irrep_gap * 0.75)
    if ylabel:
        #ax.set_ylabel(ylabel)
        ax.set_ylabel(ylabel, fontsize=tick_fs * 1.2, labelpad=15)
    # if ylabel:
    #     ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    import matplotlib.lines as _mlines
    # a highlight with no label draws its rings but contributes no legend entry
    # (stacked panels: label the key once, on the first panel only)
    hl_handles = [
        _mlines.Line2D([], [], color='none', marker='o', ls='none',
                       markerfacecolor='none',
                       markeredgecolor=_h.get('color', '#b2182b'),
                       markeredgewidth=_h.get('lw', 1.7),
                       markersize=plt.rcParams.get('lines.markersize', 8) * 0.95,
                       label=_h['label'])
        for _, _h in _hl_sets if _h.get('label')
    ]

    if pred_handles or hl_handles:
        data_handle = _mlines.Line2D([], [], color='0.2', marker='o', ls='none',
                                     markeredgecolor='black', markeredgewidth=0.8,
                                     label='lattice data')
        _leg_fs = plt.rcParams.get('legend.fontsize', 12) * 0.62
        _leg_kw = dict(framealpha=0.85, fontsize=_leg_fs, handlelength=1.6,
                       borderpad=0.3, labelspacing=0.25, handletextpad=0.5,
                       borderaxespad=0.25)
        if wave_handles:
            # two compact legends side by side at the top centre:
            # fit styles (left) and the wave colour key (right)
            leg1 = ax.legend(handles=[data_handle] + pred_handles + hl_handles,
                             loc='upper center', bbox_to_anchor=(0.40, 1.0),
                             **_leg_kw)
            ax.add_artist(leg1)
            ax.legend(handles=wave_handles, loc='upper center',
                      bbox_to_anchor=(0.66, 1.0), **_leg_kw)
        else:
            _handles = [data_handle] + pred_handles + hl_handles
            if hl_handles:
                # ring handles are round and much taller than the line handles
                # this spacing was tuned for, so they collide with the row above
                # at labelspacing 0.25.  One row, with room around each marker.
                _leg_kw.update(labelspacing=0.6, handletextpad=0.7,
                               columnspacing=1.5, borderpad=0.45)
            ax.legend(handles=_handles, loc='upper center',
                      ncol=(len(_handles) if hl_handles else 1), **_leg_kw)

    if _ylo is not None:
        ax.set_ylim(_ylo, _yhi)   # re-assert after tick edits

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    return fig, ax


# ── (Legacy alias kept for back-compat — new code should call the function above)
def plot_energies_vs_irreps_old_DO_NOT_USE(data, mN, L, C_o_M=True, save_path=None, ylabel=None,
                             y_cluster_tol=0.02, cluster_max_offset=0.08):
    """
    y_cluster_tol    : points within this y-distance of each other are considered clustered
    cluster_max_offset: maximum horizontal nudge applied to the outermost point in a cluster
    """
    psq_map = {
        'PSQ0': np.array([0, 0, 0]),
        'PSQ1': np.array([0, 0, 1]),
        'PSQ2': np.array([1, 1, 0]),
        'PSQ3': np.array([1, 1, 1]),
        'PSQ4': np.array([0, 0, 2])
    }

    line_color = '#333333'
    line_alpha = 0.5
    line_style = '--'
    line_width = 1

    level_colors_list = list(plt.cm.tab20.colors)
    ecm_color_map = {i: level_colors_list[i % len(level_colors_list)] for i in range(20)}

    # Marker shapes per irrep (matching QC plot style)
    marker_shapes = ["o", "s", "D", "^", "v", "P", "*", "X", "<", ">", "h", "8", "p", "H"]
    irrep_marker_map = {}  # (psq, irrep) -> marker

    x_mapping = {}
    x_labels = []

    psq_gap = 0.2
    irrep_gap = 0.3
    level_padding = 0.1
    psq_ranges = {}

    all_yvals = []

    # --- First pass: collect all points per x_pos ---
    # Structure: { irrep_label -> { x_pos: [(x_pos, central_value, ecm_key, color, error, marker), ...] } }
    points_per_xpos = {}  # x_pos (float) -> list of (x_pos, y, ecm_key, color, error, marker)
    psq_x_positions_map = {}  # psq -> list of x_pos

    for psq_index, (psq, irreps_dict) in enumerate(data.items()):
        psq_base = psq_index * psq_gap
        psq_x_positions = []

        for irrep_idx, (irrep, levels) in enumerate(irreps_dict.items()):
            # Assign marker per (psq, irrep) combination
            irrep_key = (psq, irrep)
            if irrep_key not in irrep_marker_map:
                global_idx = len(irrep_marker_map)
                irrep_marker_map[irrep_key] = marker_shapes[global_idx % len(marker_shapes)]
            
            marker = irrep_marker_map[irrep_key]
            
            irrep_label = f"{_format_irrep_label(irrep)} ({psq[-1]})"
            if irrep_label not in x_mapping:
                x_mapping[irrep_label] = psq_base + len(x_mapping) * irrep_gap
                x_labels.append(irrep_label)

            x_pos = x_mapping[irrep_label]
            psq_x_positions.append(x_pos)

            if x_pos not in points_per_xpos:
                points_per_xpos[x_pos] = []

            for ecm_key, values in levels.items():
                color = ecm_color_map[ecm_key]
                central_value = values[0]
                error = np.std(values[1:]) if len(values) > 1 else 0.0
                all_yvals.append(central_value + error)
                all_yvals.append(central_value - error)
                points_per_xpos[x_pos].append((x_pos, central_value, ecm_key, color, error, marker))

        if psq_x_positions:
            psq_ranges[psq] = (min(psq_x_positions), max(psq_x_positions))
        psq_x_positions_map[psq] = psq_x_positions

    # Make single-channel plots visually narrower.
    fig_width = 4 if len(x_labels) <= 2 else 6
    plt.figure(figsize=(fig_width, 6.134), dpi=300, facecolor='white')
    from matplotlib.patches import FancyBboxPatch
    ax = plt.gca()

    # --- Second pass: apply jitter and plot ---
    for x_pos, points in points_per_xpos.items():
        jittered = _apply_horizontal_jitter(points, y_cluster_tol, cluster_max_offset)
        for (jx, y, ecm_key, color, error, marker) in jittered:
            plt.errorbar(
                jx,
                y,
                yerr=error,
                fmt=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor='black',
                markersize=8,
                markeredgewidth=0.8,
                alpha=0.85,
                capsize=4,
                linewidth=1.5
            )
            plt.annotate(
                str(ecm_key),
                xy=(jx, y),
                xytext=(3, 3),
                textcoords='offset points',
                ha='left',
                va='bottom',
                color='black',
                alpha=0.9,
                fontsize=8
            )

    # --- Plot non-interacting spectrum lines ---
    # Draw per-irrep column so that single-irrep PSQs still get a visible band.
    for psq, x_positions in psq_x_positions_map.items():
        if not x_positions:
            continue
        nP = psq_map[psq]
        all_spectra = []
        for m_sample in mN:
            if C_o_M:
                spectrum = generate_free_spectrum(m_sample, L, nP, 1.54)
                all_spectra.append(np.array(spectrum) / m_sample)
            else:
                spectrum = generate_free_spectrum_lab(m_sample, L, nP, 2.1)
                all_spectra.append(np.array(spectrum) / m_sample)

        max_len = max(len(s) for s in all_spectra)
        spectra_arr = np.full((len(all_spectra), max_len), np.nan)
        for i, s in enumerate(all_spectra):
            spectra_arr[i, :len(s)] = s

        for level in range(spectra_arr.shape[1]):
            level_vals = spectra_arr[:, level]
            level_vals = level_vals[~np.isnan(level_vals)]
            if len(level_vals) == 0:
                continue
            mean = level_vals[0]#np.mean(level_vals)
            std = (np.percentile(level_vals[1:], 84) - np.percentile(level_vals[1:], 16)) / 2
            band_height = 2.0 * std

            for x_pos in x_positions:
                x_left = x_pos - irrep_gap / np.pi
                x_right = x_pos + irrep_gap / np.pi
                w = x_right - x_left
                h = band_height

                # Dashed line at mean (always visible)
                ax.hlines(
                    y=mean,
                    xmin=x_left, xmax=x_right,
                    color='#404040',
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=1,
                )

                if h > 0:
                    pad = 0.0
                    rect = FancyBboxPatch(
                        (x_left, mean - 0.5 * h),
                        w, h,
                        boxstyle=f"round,pad={pad}",
                        linewidth=0.0,
                        edgecolor='none',
                        facecolor='#404040',
                        alpha=0.50,
                        zorder=2,
                        clip_on=True,
                    )
                    ax.add_patch(rect)

    if all_yvals:
        ymin = min(all_yvals)
        ymax = max(all_yvals)
        yrange = ymax - ymin
        plt.ylim(ymin - 0.05 * yrange, ymax + 0.1 * yrange)

    # Two-particle threshold line at E* = 2m_N
    if all_yvals and min(all_yvals) < 2.0:
        x_vals = list(x_mapping.values())
        x_mid = (min(x_vals) + max(x_vals)) / 2
        ax.axhline(y=2.0, color="#C0C0C0", linestyle='--', linewidth=1.0, alpha=0.4, zorder=0)
        ax.text(x_mid, 2.0, r'$2m_N$', ha='center', va='top', color='#C0C0C0', alpha=0.9)

    plt.xticks(
        ticks=[x_mapping[label] for label in x_labels],
        labels=x_labels,
        rotation=45,
        ha='right'
    )
    #_halve_yticks()

    if ylabel:
        plt.ylabel(ylabel)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

# def plot_energies_vs_irreps(data, mN, L, C_o_M = True, save_path= None, ylabel=None):
#     # print("Data received for plotting:", data)  # Debugging output
#     # print("Mass normalization factor (mN):", mN)  # Debugging output
#     # print("Lattice size (L):", L)  # Debugging output

#     # Marker settings (all circles now with outlines)
#     # markers = {
#     #     'ecm_0': 'o', 'ecm_1': 'o', 'ecm_2': 'o', 'ecm_3': 'o', 'ecm_4': 'o'
#     # }

#     # # Colors for irreps using a categorical palette
#     # irrep_colors = itertools.cycle(plt.cm.tab10.colors)
#     psq_map = {
#         'PSQ0': np.array([0, 0, 0]),
#         'PSQ1': np.array([0, 0, 1]),
#         'PSQ2': np.array([1, 1, 0]),
#         'PSQ3': np.array([1, 1, 1]),
#         'PSQ4': np.array([0, 0, 2])
#     }
#     # Custom styling parameters
#     line_color = '#333333'  # Dark gray for better contrast
#     line_alpha = 0.5
#     line_style = '--'
#     line_width = 1
    
#     # # Map for ecm_keys to colors
#     # ecm_color_map = {}
#      # Color settings - using larger categorical palette
#     # Pre-assign colors to level indices so that level N always gets the same color
#     level_colors_list = list(plt.cm.tab20.colors)
#     ecm_color_map = {i: level_colors_list[i % len(level_colors_list)] for i in range(20)}
#     added_to_legend = set()  # Track legend entries

#     # Initialize x-axis positions and labels
#     x_mapping = {}
#     x_labels = []
    
#     # # Spacing parameters
#     # psq_gap = 1  # Gap between PSQs
#     # irrep_gap = 0.25  # Gap between irreps within same PSQ
#     # Spacing parameters
#     psq_gap = 0.2  # Reduced spacing between PSQ groups
#     irrep_gap = 0.3  # Width allocated per irrep within PSQ
#     level_padding = 0.1  # Padding within irrep's space
#     # Dictionary to track x-ranges for each PSQ group
#     psq_ranges = {}

#     plt.figure(figsize=(10, 6.134), dpi=300, facecolor='white')
#     plt.rcParams['font.family'] = 'serif'  # Consistent font style
#     # Collect all y-values for auto y-limits
#     all_yvals = []

#     for psq_index, (psq, irreps_dict) in enumerate(data.items()):
#         psq_base = psq_index * psq_gap  # Base position for this PSQ
        
#         # Store x positions for this PSQ group
#         psq_x_positions = []
#         # Calculate PSQ group boundaries
#         num_irreps = len(irreps_dict)
#         psq_start = psq_base
#         psq_end = psq_base + (num_irreps * irrep_gap)         
#         for irrep, levels in irreps_dict.items():
#             # Create composite irrep label
#             irrep_label = f"{irrep} ({psq[-1]})"
            
#             if irrep_label not in x_mapping:
#                 x_mapping[irrep_label] = psq_base + len(x_mapping) * irrep_gap
#                 x_labels.append(irrep_label)

#             x_pos = x_mapping[irrep_label]
#             psq_x_positions.append(x_pos)

#             for ecm_key, values in levels.items():
#                 # Use pre-assigned color for this level index
#                 color = ecm_color_map[ecm_key]
                
#                 # Calculate values
#                 central_value = values[0] # * mN[0]
#                 error = np.std(values[1:]) if len(values) > 1 else 0.0
#                 all_yvals.append(central_value + error)
#                 all_yvals.append(central_value - error)

#                 # plt.errorbar(
#                 #     x_pos,
#                 #     central_value,
#                 #     yerr=error,
#                 #     #fmt=markers[ecm_key],
#                 #     color=color,  # Error bar and marker edge color
#                 #     markerfacecolor=color,
#                 #     markeredgecolor='black',
#                 #     markersize=8,
#                 #     markeredgewidth=1,
#                 #     alpha=0.85,
#                 #     capsize=4,
#                 #     linewidth=1.5,
#                 #     label=f"{ecm_key}" if psq_index == 0 and irrep == list(irreps_dict.keys())[0] else None
#                 # )
#                 plt.errorbar(
#                     x_pos,
#                     central_value,
#                     yerr=error,
#                     fmt='o',  # Force circle marker
#                     color=color,
#                     markerfacecolor=color,
#                     markeredgecolor='black',
#                     markersize=8,  # Smaller markers
#                     markeredgewidth=0.8,
#                     alpha=0.85,
#                     capsize=4,
#                     linewidth=1.5
#                 )
#                 plt.annotate(
#                     str(ecm_key),
#                     xy=(x_pos, central_value),
#                     xytext=(6, 4),
#                     textcoords='offset points',
#                     ha='left',
#                     va='bottom',
#                     fontsize=11,
#                     color='black',
#                     alpha=0.9
#                 )
#             # Calculate PSQ group boundaries after collecting all x positions
#         if psq_x_positions:
#             psq_ranges[psq] = (min(psq_x_positions), max(psq_x_positions))
#     # Plot spectrum lines for each PSQ group using calculated ranges
#     for psq, (xmin, xmax) in psq_ranges.items():
#         nP = psq_map[psq]
#          # Compute free spectrum for all mN samples
#         all_spectra = []
#         for m_sample in mN:
#             if C_o_M:
#                 spectrum = generate_free_spectrum(m_sample, L, nP, 1.54)
#                 all_spectra.append(np.array(spectrum) / m_sample)  # <-- divide by mN here
#             else:
#                 spectrum = generate_free_spectrum_lab(m_sample, L, nP, 2.1)
#                 all_spectra.append(np.array(spectrum) / m_sample)  # <-- divide by mN here
#         # Pad spectra to the same length with np.nan
#         max_len = max(len(s) for s in all_spectra)
#         spectra_arr = np.full((len(all_spectra), max_len), np.nan)
#         for i, s in enumerate(all_spectra):
#             spectra_arr[i, :len(s)] = s
#         # For each level, plot mean and error band
#         for level in range(spectra_arr.shape[1]):
#             level_vals = spectra_arr[:, level]
#             # Only use non-nan values
#             level_vals = level_vals[~np.isnan(level_vals)]
#             if len(level_vals) == 0:
#                 continue
#             mean = np.mean(level_vals)
#             std = (np.percentile(level_vals, 84) - np.percentile(level_vals, 16)) / 2
#             #error = np.percentile(values[1:], 84) - np.percentile(values[1:], 16)
#             #std = np.std(level_vals)
#             # Plot mean as dashed line
#             plt.hlines(
#                 y=mean,
#                 xmin=xmin - irrep_gap/2,
#                 xmax=xmax + irrep_gap/2,
#                 color=line_color,
#                 linestyle=line_style,
#                 linewidth=line_width,
#                 alpha=line_alpha,
#                 zorder=1
#             )
#             # Plot error band
#             plt.fill_between(
#                 [xmin - irrep_gap/2, xmax + irrep_gap/2],
#                 mean - std, mean + std,
#                 color=line_color,
#                 alpha=0.30,
#                 zorder=0
#             )
#         # spectrum = generate_free_spectrum(mN[0], L, nP, 1.54)
        
#         # for spec in spectrum:
#         #     plt.hlines(
#         #         y=spec,
#         #         xmin=xmin - irrep_gap/2,  # Adjust for padding
#         #         xmax=xmax + irrep_gap/2,
#         #         color=line_color,
#         #         linestyle=line_style,
#         #         linewidth=line_width,
#         #         alpha=line_alpha,
#         #         zorder=1
#         #     )

#     # Set y-limits with a margin above the highest value
#     if all_yvals:
#         ymin = min(all_yvals)
#         ymax = max(all_yvals)
#         yrange = ymax - ymin
#         plt.ylim(ymin  - 0.05 * yrange, ymax + 0.1 * yrange)  # 5% margin above

#     # Axis formatting
#     plt.xticks(
#         ticks=[x_mapping[label] for label in x_labels],
#         labels=x_labels,
#         rotation=45,
#         ha='right',
#         fontsize=10
#     )
#     plt.yticks(fontsize=10)
    
#     #plt.xlabel("Irreps (PSQ Group)", fontsize=12, labelpad=10)
#     if ylabel:
#         plt.ylabel(ylabel, fontsize=12)
    
#     # Grid and legend styling
#     # plt.grid(True, linestyle='--', alpha=0.3)
#     # plt.legend(
#     #     loc="upper left",
#     #     bbox_to_anchor=(1.05, 1),
#     #     frameon=True,
#     #     framealpha=0.9,
#     #     edgecolor='black'
#     # )

#     # Tight layout and save
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#         #os.system(f'open {save_path}')


def plot_level_subplots_with_noninteracting(energy_cm_data, noninteracting_energies, 
                                           kept_irreps, kept_levels, kept_nPvec, 
                                           massN, lattice_size, save_path=None, ylabel=None,
                                           subplot_width=2.2, subplot_height=3.0):
    """
    Create subplots for each energy level, showing interacting vs non-interacting energies.
    Uses the same plotting style as the regular plot_energies_vs_irreps function.
    
    Args:
        energy_cm_data: Dictionary with energy data from build_energy_cm_dict
        noninteracting_energies: Dictionary with non-interacting energy levels
        kept_irreps: List of irrep labels for each level
        kept_levels: List of level numbers for each entry
        kept_nPvec: List of momentum squared values for each entry
        massN: Mass normalization array
        lattice_size: Lattice size parameter
        save_path: Path to save the plot
        ylabel: Y-axis label
    """
    # Momentum vector mapping (same as in regular plotting)
    psq_map = {
        'PSQ0': np.array([0, 0, 0]),
        'PSQ1': np.array([0, 0, 1]),
        'PSQ2': np.array([1, 1, 0]),
        'PSQ3': np.array([1, 1, 1]),
        'PSQ4': np.array([0, 0, 2])
    }
    
    psq_map_reverse = {
        0: 'PSQ0', 1: 'PSQ1', 2: 'PSQ2', 3: 'PSQ3', 4: 'PSQ4'
    }
    
    # Styling parameters (same as regular plotting)
    line_color = '#333333'
    line_alpha = 0.5
    line_style = '--'
    line_width = 1
    
    # Collect all data points for subplot organization
    data_points = []
    for i, (irrep, level, nP) in enumerate(zip(kept_irreps, kept_levels, kept_nPvec)):
        psq_label = psq_map_reverse.get(nP, f'PSQ{nP}')
        
        # Find corresponding energy data
        level_data = None
        if psq_label in energy_cm_data and irrep in energy_cm_data[psq_label]:
            if level in energy_cm_data[psq_label][irrep]:
                level_data = energy_cm_data[psq_label][irrep][level]
        
        if level_data is not None:
            data_points.append({
                'irrep': irrep,
                'level': level,
                'nP': nP,
                'psq_label': psq_label,
                'energy_data': level_data,
                'momentum_vec': psq_map[psq_label],
                'label': f"{irrep}_{psq_label}_level{level}"
            })
    
    # Sort data points by PSQ and irrep for consistent ordering
    data_points.sort(key=lambda x: (x['nP'], x['irrep'], x['level']))
    
    n_plots = len(data_points)
    if n_plots == 0:
        print("No data points found for subplot creation")
        return
        
    # Calculate grid layout (prefer wider than tall)
    cols = int(np.ceil(np.sqrt(n_plots * 1.5)))
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(subplot_width * cols, subplot_height * rows),
        dpi=300,
        facecolor='white',
        sharex=True,
    )
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    # Colors (same color scheme as regular plotting)
    level_colors_list = list(plt.cm.tab20.colors)
    ecm_color_map = {i: level_colors_list[i % len(level_colors_list)] for i in range(20)}
    
    for i, dp in enumerate(data_points):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get interacting energy data
        energy_values = dp['energy_data']
        central_value = energy_values[0]
        
        # Calculate error using percentiles (same as regular plotting)
        if len(energy_values) > 1:
            error = (np.percentile(energy_values[1:], 84) - np.percentile(energy_values[1:], 16)) / 2
        else:
            error = 0.0
        
        # Collect y values for proper axis limits
        all_yvals = [central_value + error, central_value - error]
        
        # Use correct color for this level (same as regular plotting)
        interacting_color = ecm_color_map.get(dp['level'], level_colors_list[0] if level_colors_list else '#2E86C1')
        
        # Plot interacting energy with same style as regular plotting
        ax.errorbar(
            0, central_value, yerr=error,
            fmt='o',
            color=interacting_color,
            markerfacecolor=interacting_color,
            markeredgecolor='black',
            markersize=8,
            markeredgewidth=0.8,
            alpha=0.85,
            capsize=4,
            linewidth=1.5
        )
        
        # Add level annotation (same as regular plotting)
        ax.annotate(
            str(dp['level']),
            xy=(0, central_value),
            xytext=(6, 4),
            textcoords='offset points',
            ha='left',
            va='bottom',
            color='black',
            alpha=0.9
        )
        
        # Calculate free spectrum with error bands (same method as regular plotting)
        nP_vec = dp['momentum_vec']
        
        # Compute free spectrum for all mN samples
        all_spectra = []
        for m_sample in massN:
            spectrum = generate_free_spectrum(m_sample, lattice_size, nP_vec, central_value + 0.4)
            all_spectra.append(np.array(spectrum) / m_sample)  # Normalize by mN here
        
        # Pad spectra to the same length with np.nan
        if all_spectra:
            max_len = max(len(s) for s in all_spectra)
            spectra_arr = np.full((len(all_spectra), max_len), np.nan)
            for j, s in enumerate(all_spectra):
                spectra_arr[j, :len(s)] = s
            
            x_range = [-0.4, 0.4]  # Horizontal line extent
            
            # Plot all non-interacting levels with error bands
            for level_idx in range(spectra_arr.shape[1]):
                level_vals = spectra_arr[:, level_idx]
                # Only use non-nan values
                level_vals = level_vals[~np.isnan(level_vals)]
                if len(level_vals) == 0:
                    continue
                    
                mean_energy = np.mean(level_vals)
                std_energy = (np.percentile(level_vals, 84) - np.percentile(level_vals, 16)) / 2
                
                # Plot mean as dashed line (same style as regular plotting)
                ax.hlines(
                    y=mean_energy,
                    xmin=x_range[0], xmax=x_range[1],
                    color=line_color,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=line_alpha,
                    zorder=1
                )
                
                # Plot error band (same style as regular plotting)
                ax.fill_between(
                    x_range,
                    mean_energy - std_energy, mean_energy + std_energy,
                    color=line_color,
                    alpha=0.30,
                    zorder=0
                )
        
        # Set extremely tight y-limits zoomed right on the interacting energy
        y_center = central_value
        y_range = .004#max(error * 0.4, 0.005)  # Extremely tight zoom - 0.8x error or minimum 0.005
        ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        # Formatting for each subplot
        ax.set_xlim(-0.5, 0.5)
        psq_num = str(dp['psq_label']).replace('PSQ', '')
        ax.set_xticks([0.0])
        ax.set_xticklabels([f"{_format_irrep_label(dp['irrep'])} ({psq_num})"])
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        _halve_yticks(ax, max_ticks=6)
        
        if ylabel and i % cols == 0:  # Only leftmost subplots get y-label
            ax.set_ylabel(ylabel)
    
    # Hide unused subplots
    for i in range(len(data_points), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Level subplots saved to: {save_path}")


def plot_energies_vs_irreps_quick(data, mN, L, C_o_M = True, save_path= None, ylabel=None):
    # print("Data received for plotting:", data)  # Debugging output
    # print("Mass normalization factor (mN):", mN)  # Debugging output
    # print("Lattice size (L):", L)  # Debugging output

    # Marker settings (all circles now with outlines)
    # markers = {
    #     'ecm_0': 'o', 'ecm_1': 'o', 'ecm_2': 'o', 'ecm_3': 'o', 'ecm_4': 'o'
    # }

    # # Colors for irreps using a categorical palette
    # irrep_colors = itertools.cycle(plt.cm.tab10.colors)
    psq_map = {
        'PSQ0': np.array([0, 0, 0]),
        'PSQ1': np.array([0, 0, 1]),
        'PSQ2': np.array([1, 1, 0]),
        'PSQ3': np.array([1, 1, 1]),
        'PSQ4': np.array([0, 0, 2])
    }
    # Custom styling parameters
    line_color = '#333333'  # Dark gray for better contrast
    line_alpha = 0.5
    line_style = '--'
    line_width = 1
    
    # # Map for ecm_keys to colors
    # ecm_color_map = {}
     # Color settings - using larger categorical palette
    # Pre-assign colors to level indices so that level N always gets the same color
    level_colors_list = list(plt.cm.tab20.colors)
    ecm_color_map = {i: level_colors_list[i % len(level_colors_list)] for i in range(20)}
    added_to_legend = set()  # Track legend entries

    # Initialize x-axis positions and labels
    x_mapping = {}
    x_labels = []
    
    # # Spacing parameters
    # psq_gap = 1  # Gap between PSQs
    # irrep_gap = 0.25  # Gap between irreps within same PSQ
    # Spacing parameters
    psq_gap = 0.2  # Reduced spacing between PSQ groups
    irrep_gap = 0.3  # Width allocated per irrep within PSQ
    level_padding = 0.1  # Padding within irrep's space
    # Dictionary to track x-ranges for each PSQ group
    psq_ranges = {}

    plt.figure(figsize=(8,6), dpi=300, facecolor='white')
    # Collect all y-values for auto y-limits
    all_yvals = []

    for psq_index, (psq, irreps_dict) in enumerate(data.items()):
        psq_base = psq_index * psq_gap  # Base position for this PSQ
        
        # Store x positions for this PSQ group
        psq_x_positions = []
        # Calculate PSQ group boundaries
        num_irreps = len(irreps_dict)
        psq_start = psq_base
        psq_end = psq_base + (num_irreps * irrep_gap)         
        for irrep, levels in irreps_dict.items():
            # Create composite irrep label
            irrep_label = f"{_format_irrep_label(irrep)} ({psq[-1]})"
            
            if irrep_label not in x_mapping:
                x_mapping[irrep_label] = psq_base + len(x_mapping) * irrep_gap
                x_labels.append(irrep_label)

            x_pos = x_mapping[irrep_label]
            psq_x_positions.append(x_pos)

            for ecm_key, values in levels.items():
                # Use pre-assigned color for this level index
                color = ecm_color_map[ecm_key]
                
                # Calculate values
                central_value = values[0] # * mN[0]
                error = np.std(values[1:]) if len(values) > 1 else 0.0
                all_yvals.append(central_value + error)
                all_yvals.append(central_value - error)

                # plt.errorbar(
                #     x_pos,
                #     central_value,
                #     yerr=error,
                #     #fmt=markers[ecm_key],
                #     color=color,  # Error bar and marker edge color
                #     markerfacecolor=color,
                #     markeredgecolor='black',
                #     markersize=8,
                #     markeredgewidth=1,
                #     alpha=0.85,
                #     capsize=4,
                #     linewidth=1.5,
                #     label=f"{ecm_key}" if psq_index == 0 and irrep == list(irreps_dict.keys())[0] else None
                # )
                plt.errorbar(
                    x_pos,
                    central_value,
                    yerr=error,
                    fmt='o',  # Force circle marker
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor='black',
                    markersize=8,  # Smaller markers
                    markeredgewidth=0.8,
                    alpha=0.85,
                    capsize=4,
                    linewidth=1.5
                )
                plt.annotate(
                    str(ecm_key),
                    xy=(x_pos, central_value),
                    xytext=(6, 4),
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    fontsize=11,
                    color='black',
                    alpha=0.9
                )
            # Calculate PSQ group boundaries after collecting all x positions
        if psq_x_positions:
            psq_ranges[psq] = (min(psq_x_positions), max(psq_x_positions))
    # Plot spectrum lines for each PSQ group using calculated ranges
    for psq, (xmin, xmax) in psq_ranges.items():
        nP = psq_map[psq]
         # Compute free spectrum for all mN samples
        all_spectra = []
        for m_sample in mN:
            if C_o_M:
                spectrum = generate_free_spectrum(m_sample, L, nP, 1.54)
                all_spectra.append(np.array(spectrum) / m_sample)  # <-- divide by mN here
            else:
                spectrum = generate_free_spectrum_lab(m_sample, L, nP, 2.1)
                all_spectra.append(np.array(spectrum) / m_sample)  # <-- divide by mN here
        # Pad spectra to the same length with np.nan
        max_len = max(len(s) for s in all_spectra)
        spectra_arr = np.full((len(all_spectra), max_len), np.nan)
        for i, s in enumerate(all_spectra):
            spectra_arr[i, :len(s)] = s
        # For each level, plot mean and error band
        for level in range(spectra_arr.shape[1]):
            level_vals = spectra_arr[:, level]
            # Only use non-nan values
            level_vals = level_vals[~np.isnan(level_vals)]
            if len(level_vals) == 0:
                continue
            mean = np.mean(level_vals)
            std = (np.percentile(level_vals, 84) - np.percentile(level_vals, 16)) / 2
            #error = np.percentile(values[1:], 84) - np.percentile(values[1:], 16)
            #std = np.std(level_vals)
            # Plot mean as dashed line
            plt.hlines(
                y=mean,
                xmin=xmin - irrep_gap/2,
                xmax=xmax + irrep_gap/2,
                color=line_color,
                linestyle=line_style,
                linewidth=line_width,
                alpha=line_alpha,
                zorder=1
            )
            # Plot error band
            plt.fill_between(
                [xmin - irrep_gap/2, xmax + irrep_gap/2],
                mean - std, mean + std,
                color=line_color,
                alpha=0.30,
                zorder=0
            )
        # spectrum = generate_free_spectrum(mN[0], L, nP, 1.54)
        
        # for spec in spectrum:
        #     plt.hlines(
        #         y=spec,
        #         xmin=xmin - irrep_gap/2,  # Adjust for padding
        #         xmax=xmax + irrep_gap/2,
        #         color=line_color,
        #         linestyle=line_style,
        #         linewidth=line_width,
        #         alpha=line_alpha,
        #         zorder=1
        #     )

    # Set y-limits with a margin above the highest value
    if all_yvals:
        ymin = min(all_yvals)
        ymax = max(all_yvals)
        yrange = ymax - ymin
        plt.ylim(ymin  - 0.05 * yrange, ymax + 0.1 * yrange)  # 5% margin above

    # Axis formatting
    plt.xticks(
        ticks=[x_mapping[label] for label in x_labels],
        labels=x_labels,
        rotation=45,
        ha='right',
        fontsize=10
    )
    plt.yticks(fontsize=10)
    
    #plt.xlabel("Irreps (PSQ Group)", fontsize=12, labelpad=10)
    if ylabel:
        plt.ylabel(ylabel, fontsize=12)


# ── Fit-specific plot helpers ──────────────────────────────────────────────────
# Lazy imports from my_fit_model avoid a circular dependency (my_fit_model
# imports plotting at module level, so plotting cannot import my_fit_model at
# module level).  The deferred `from my_fit_model import …` inside each
# function body runs only when the function is first called, by which point
# both modules are fully initialised.

def plot_bmatrix_preview(*args, **kwargs):
    from my_fit_model import plot_bmatrix_preview as _f
    return _f(*args, **kwargs)

def plot_quantization_condition(*args, **kwargs):
    from my_fit_model import plot_quantization_condition as _f
    return _f(*args, **kwargs)

def plot_phase_shifts_multichannel(*args, **kwargs):
    from my_fit_model import plot_phase_shifts_multichannel as _f
    return _f(*args, **kwargs)

def plot_luscher(*args, **kwargs):
    from my_fit_model import plot_luscher as _f
    return _f(*args, **kwargs)

# back-compat alias for the pre-rename name
plot_fit_comparison = plot_luscher

def plot_chi2_landscape(*args, **kwargs):
    from my_fit_model import plot_chi2_landscape as _f
    return _f(*args, **kwargs)

def plot_box_quantization_diagnostics(*args, **kwargs):
    from my_fit_model import plot_box_quantization_diagnostics as _f
    return _f(*args, **kwargs)

def plot_omega_and_eigenvalues(*args, **kwargs):
    from my_fit_model import plot_omega_and_eigenvalues as _f
    return _f(*args, **kwargs)

def print_eigenvector_decomposition(*args, **kwargs):
    from my_fit_model import print_eigenvector_decomposition as _f
    return _f(*args, **kwargs)

def plot_eigenvector_decomposition(*args, **kwargs):
    from my_fit_model import plot_eigenvector_decomposition as _f
    return _f(*args, **kwargs)
  