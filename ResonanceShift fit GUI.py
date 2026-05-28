"""
Interactive Resonance-Shift Fit GUI for OriginPro (embedded Python).

Run this from Origin's Python window. It:
  1) reads data from workbook 'LossMat36Res1', sheet 'Temperature_dependent'
     (the resonance-shift layout: a temperature vector in column 0, then, per
     power, three columns whose ROLE is in the Long Name -- 'fr' / 'Dfr' /
     'err' -- and whose POWER (dBm) is in the Comment label). No photon-number
     sheet is used: photon number does not enter a resonance-shift model.
  2) opens a Tkinter GUI where you can:
       * trim the data by power range and temperature range BEFORE fitting,
       * pick which powers AND which temperatures to include via checkboxes
         (a point is fit only if BOTH its power and its temperature are
         checked) -- so you can fit a SINGLE power if you want,
       * edit parameter ranges (F_delta_TLS0, offset, alpha, Tc),
       * edit the reference temperature T_REF live,
       * choose the QP frequency-shift model: clean/BCS conductivity, dirty
         limit (kT/Delta(T))^4, the original digamma-style x/sinh(x) form, or
         no QP at all (pure TLS),
       * choose a fitting strategy (Differential Evolution + local polish,
         random restarts, or local polish only from the current sliders),
       * stop a running fit,
       * adjust each parameter with a slider and see the model overlay update
         live, with a residuals panel below,
  3) on "Export to Origin", writes results back to the same workbook
     (TLS_Fit_Parameters / TLS_Fit_Data / Resonance_shift_total_Fit sheets)
     and creates a graph 'ResonanceShiftPlot' -- overwriting any previous run.

The model is the fractional resonance shift  Delta f_r / f_r  vs temperature,
referenced to T_REF so every contribution passes through zero there:

    model(T) = [ TLS(T)  - TLS(T_REF) ]
             + [ QP(T)   - QP(T_REF)  ]      (if a QP model is selected)
             + offset                        (if offset is included)

The GUI is modal: while it is open you cannot interact with Origin. Close it
to return control to Origin.
"""

# ---------------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------------
SRC_BOOK    = 'LossMat36Res1'
TD_SHEET    = 'Temperature_dependent'

# Long-Name prefixes that identify each column block in TD_SHEET.
# The power (dBm) for every data column is read from its Comment label.
LN_FR  = 'fr'      # resonance frequency      columns  (e.g. 'fr_P1')
LN_DFR = 'Dfr'     # resonance-shift          columns  (e.g. 'Dfr_P1' / 'Δfr_P1')
LN_ERR = 'err'     # frequency-error          columns  (e.g. 'err_P1')

# Default reference temperature (K) -- editable live in the GUI.
DEFAULT_T_REF = 0.049

# Output target names (fixed -- re-running a fit overwrites these).
OUT_PARAMS_SHEET = 'TLS_Fit_Parameters'
OUT_DATA_SHEET   = 'TLS_Fit_Data'
OUT_CURVES_SHEET = 'Resonance_shift_total_Fit'
OUT_GRAPH_NAME   = 'ResonanceShiftPlot'
OUT_GRAPH_TEMPLATE = 'TLSTempFit'   # falls back to default template if missing

# ---------------------------------------------------------------------------
import math
import time
import threading
import traceback
import queue
from itertools import cycle

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from scipy import constants
from scipy.special import digamma, kv, iv
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d

import originpro as op

h, kB, hbar = constants.h, constants.k, constants.hbar

# Free parameters of the resonance-shift model.
PARAM_NAMES = ['F_delta_TLS0', 'offset', 'alpha', 'Tc']
LOG_PARAMS  = {'F_delta_TLS0'}        # this slider is log-scale
UNITS       = {'Tc': 'K'}

DEFAULT_RANGES = {
    'F_delta_TLS0': (1e-6, 5e-5),
    'offset':       (-5e-5, 5e-5),
    'alpha':        (0.01, 10.0),
    'Tc':           (3.0, 9.25),
}

# QP model choices -- the value strings are what the worker thread reads.
QP_MODELS = [
    ('bcs',   'Clean / BCS conductivity   (sigma1+i*sigma2)'),
    ('dirty', 'Dirty limit   (kB*T / Delta(T))^4'),
    ('simple','Original digamma-style   x / sinh(x)'),
    ('none',  'No QP   (pure TLS)'),
]


# ===========================================================================
# Physics
# ===========================================================================
def tls_shift(T, F0, fref):
    """Fractional TLS frequency shift (absolute, not yet referenced)."""
    T = np.maximum(np.asarray(T, dtype=float), 1e-9)
    x = h * fref / (kB * T)
    return (F0 / np.pi) * (
        np.real(digamma(0.5 - 1.0 / (2 * 1j * np.pi) * x))
        - np.log(x / (2 * np.pi))
    )


def delta_bcs(T, Tc):
    """
    Temperature-dependent BCS gap Delta(T), closed-form tanh approximation:

        Delta(T) = Delta0 * tanh( (pi*kB*Tc/Delta0) * sqrt( a*(Tc/T - 1) ) )

    with Delta0 = 1.764*kB*Tc and a = 1.74 (standard fit, accurate to <1%).
    Delta -> 0 for T >= Tc.
    """
    T = np.asarray(T, dtype=float)
    D0 = 1.764 * kB * Tc
    a = 1.74
    Tsafe = np.minimum(np.maximum(T, 1e-9), Tc * (1 - 1e-9))
    inside = a * (Tc / Tsafe - 1.0)
    arg = (np.pi * kB * Tc / D0) * np.sqrt(np.maximum(inside, 0.0))
    gap = D0 * np.tanh(arg)
    return np.where(T < Tc, gap, 0.0)


def qp_shift_dirty(T, alpha, Tc):
    """Dirty-limit QP shift: -(alpha/2) * (kB*T/Delta(T))^4, T-dependent gap."""
    Delta_T = delta_bcs(T, Tc)
    Delta_T = np.where(Delta_T > 0, Delta_T, np.inf)
    ratio = (kB * np.asarray(T, dtype=float) / Delta_T) ** 4
    return -0.5 * alpha * ratio


def qp_shift_bcs(T, alpha, Tc, fr):
    """Full BCS conductivity QP shift with T-dependent gap Delta(T)."""
    omega = 2 * np.pi * fr
    T = np.maximum(np.asarray(T, dtype=float), 0.01)
    Delta_T = delta_bcs(T, Tc)
    Delta_T = np.where(Delta_T > 0, Delta_T, np.inf)

    x = hbar * omega / (2 * kB * T)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        exp_term = np.exp(-Delta_T / (kB * T))
        s1 = (4 * Delta_T / (hbar * omega)) * exp_term * np.sinh(x) * kv(0, x)
        s2 = (np.pi * Delta_T / (hbar * omega)) * (
            1 - np.sqrt(2 * np.pi * kB * T / Delta_T) * exp_term
        ) - 2 * np.exp(-Delta_T / (kB * T) - x) * iv(0, x)
        z = s1 + 1j * s2
        mags = np.abs(z)
        angle = np.angle(z)
        ratio_term = (mags / (np.pi * Delta_T / (hbar * omega))) ** (-0.5)
        phase_term = np.sin(angle / 2) + np.cos(angle / 2)
        out = -1.0 * alpha * ((phase_term * ratio_term) - 1.0)
    return np.where(np.isfinite(out), out, 0.0)


def qp_shift_simple(T, alpha, Tc):
    """Original digamma-script QP form: -(alpha/2) * x / sinh(x), Delta0 gap."""
    D0 = 1.764 * kB * Tc
    T = np.maximum(np.asarray(T, dtype=float), 1e-9)
    x = D0 / (kB * T)
    return -0.5 * alpha * x * (1.0 / np.sinh(x))


def qp_eval(model, T, alpha, Tc, fref):
    """Dispatch the chosen QP model. Returns absolute shift (not referenced)."""
    if model == 'bcs':
        return qp_shift_bcs(T, alpha, Tc, fref)
    if model == 'dirty':
        return qp_shift_dirty(T, alpha, Tc)
    if model == 'simple':
        return qp_shift_simple(T, alpha, Tc)
    return np.zeros_like(np.asarray(T, dtype=float))


def model_frac(params, T, fref, Tref, qp_model='bcs', include_offset=True):
    """
    Fractional resonance shift, referenced to Tref.

    params = [F0, offset, alpha, Tc]
    """
    F0, offset, alpha, Tc = params
    T = np.asarray(T, dtype=float)

    dT_tls = tls_shift(T, F0, fref)
    dRef_tls = tls_shift(np.array([Tref]), F0, fref)[0]
    out = dT_tls - dRef_tls

    if qp_model and qp_model != 'none':
        dT_qp = qp_eval(qp_model, T, alpha, Tc, fref)
        dRef_qp = qp_eval(qp_model, np.array([Tref]), alpha, Tc, fref)[0]
        out = out + (dT_qp - dRef_qp)

    if include_offset:
        out = out + offset

    return out


def model_breakdown(params, T, fref, Tref, qp_model='bcs', include_offset=True):
    """Return (tls, qp, offset_arr, total), each referenced where appropriate."""
    F0, offset, alpha, Tc = params
    T = np.asarray(T, dtype=float)

    dT_tls = tls_shift(T, F0, fref)
    dRef_tls = tls_shift(np.array([Tref]), F0, fref)[0]
    tls = dT_tls - dRef_tls

    if qp_model and qp_model != 'none':
        dT_qp = qp_eval(qp_model, T, alpha, Tc, fref)
        dRef_qp = qp_eval(qp_model, np.array([Tref]), alpha, Tc, fref)[0]
        qp = dT_qp - dRef_qp
    else:
        qp = np.zeros_like(T)

    off = np.full_like(T, offset if include_offset else 0.0)
    return tls, qp, off, tls + qp + off


def weighted_mse(y, yfit, err):
    w = 1.0 / np.maximum(err ** 2, 1e-30)
    return np.sum(w * (y - yfit) ** 2) / np.sum(w)


# ===========================================================================
# Read data from the Origin source workbook (resonance-shift layout)
# ===========================================================================
def _classify_block(long_name):
    """Return 'fr', 'dfr', 'err', or None from a column Long Name."""
    if not long_name:
        return None
    s = str(long_name).strip()
    sl = s.lower()
    if sl.startswith(LN_ERR.lower()):
        return 'err'
    # shift columns: 'Dfr...' or the unicode-delta 'Δfr...'
    if sl.startswith(LN_DFR.lower()) or s.startswith('\u0394'):
        return 'dfr'
    if sl.startswith(LN_FR.lower()):
        return 'fr'
    return None


def read_origin_source():
    """
    Read the 'Temperature_dependent' sheet.

    NO photon-number sheet is used -- this is a resonance-shift fit and photon
    number never enters the model. Instead, every data column carries its power
    (dBm) in its COMMENT label, and its role (fr / dfr / err) in its LONG NAME:

        col 0                       : temperature vector  T (K)
        Long Name 'fr_*',  Comment '<power>'   : resonance frequency
        Long Name 'Dfr_*', Comment '<power>'   : resonance shift
        Long Name 'err_*', Comment '<power>'   : frequency error

    Columns are matched into (fr, dfr, err) triples by their Comment power, so
    the absolute column order does not matter.

    Returns a dict of equal-length numpy arrays, one entry per (power, T) point:
        {P, freq, dfr, err, T_K, T_mK, frac, sigma}
    where frac = dfr/fr and sigma = err/fr (the quantities actually fit).
    """
    book = op.find_book('w', SRC_BOOK)
    if book is None:
        raise RuntimeError("Workbook '%s' not found in this Origin project."
                           % SRC_BOOK)

    td = op.find_sheet('w', '[%s]%s' % (SRC_BOOK, TD_SHEET))
    if td is None:
        raise RuntimeError("Sheet '%s' not found in '%s'."
                           % (TD_SHEET, SRC_BOOK))

    T_vec = np.array(td.to_list(0), dtype=float)

    # ---- scan every column; group by power (from Comment) and role (Long Name) ----
    # blocks[power] = {'fr': col, 'dfr': col, 'err': col}
    blocks = {}
    for c in range(1, td.cols):
        try:
            ln = td.get_label(c, type='L')   # Long Name -> role
            cm = td.get_label(c, type='C')   # Comment   -> power (dBm)
        except Exception:
            continue
        role = _classify_block(ln)
        if role is None:
            continue
        try:
            P = float(str(cm).strip())
        except (TypeError, ValueError):
            continue
        blocks.setdefault(P, {})[role] = c

    if not blocks:
        raise RuntimeError(
            "No fr/Dfr/err columns with numeric power Comments found in "
            "'[%s]%s'. Check that each data column's power is in its Comment "
            "and its role (fr/Dfr/err) is in its Long Name." % (SRC_BOOK, TD_SHEET))

    P_, fr_, dfr_, err_, TK_, TmK_, frac_, sig_ = ([] for _ in range(8))

    for P in sorted(blocks):
        cols = blocks[P]
        if not all(r in cols for r in ('fr', 'dfr', 'err')):
            continue   # incomplete triple for this power -> skip

        raw_fr  = td.to_list(cols['fr'])
        raw_dfr = td.to_list(cols['dfr'])
        raw_err = td.to_list(cols['err'])

        # skip columns whose shift data is missing/incomplete
        if (len(raw_dfr) < len(T_vec)
                or any(v is None or (isinstance(v, float) and math.isnan(v))
                       for v in raw_dfr[:len(T_vec)])):
            continue

        n = min(len(T_vec), len(raw_fr), len(raw_dfr), len(raw_err))
        for k in range(n):
            T = T_vec[k]
            fr, dfr, er = raw_fr[k], raw_dfr[k], raw_err[k]
            if None in (T, fr, dfr, er):
                continue
            try:
                T = float(T); fr = float(fr); dfr = float(dfr); er = float(er)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(T) and np.isfinite(fr) and np.isfinite(dfr)
                    and np.isfinite(er)):
                continue
            if fr == 0 or er <= 0:
                continue
            P_.append(P); fr_.append(fr); dfr_.append(dfr); err_.append(er)
            TK_.append(T); TmK_.append(T * 1e3)
            frac_.append(dfr / fr); sig_.append(er / fr)

    if len(frac_) == 0:
        raise RuntimeError("No usable data points found in '[%s]%s'."
                           % (SRC_BOOK, TD_SHEET))

    return dict(
        P     = np.array(P_,   dtype=float),
        freq  = np.array(fr_,  dtype=float),
        dfr   = np.array(dfr_, dtype=float),
        err   = np.array(err_, dtype=float),
        T_K   = np.array(TK_,  dtype=float),
        T_mK  = np.array(TmK_, dtype=float),
        frac  = np.array(frac_, dtype=float),
        sigma = np.array(sig_,  dtype=float),
    )


def fref_for_power(filt, p, Tref):
    """Interpolate fr at Tref for a given power from the filtered data."""
    sel = np.where(filt['P'] == p)[0]
    if sel.size == 0:
        return float(np.nanmean(filt['freq']))
    T_s = filt['T_K'][sel]
    f_s = filt['freq'][sel]
    order = np.argsort(T_s)
    T_s, f_s = T_s[order], f_s[order]
    return float(np.interp(Tref, T_s, f_s))


# ===========================================================================
# Fitting strategies  (single power at a time; one fref per power)
# ===========================================================================
class StopFitException(Exception):
    pass


def _make_objective(d, fref, Tref, qp_model, include_offset, stop_flag):
    eval_count = [0]

    def objective(p):
        if stop_flag is not None and stop_flag.is_set():
            raise StopFitException()
        eval_count[0] += 1
        if eval_count[0] % 100 == 0:
            time.sleep(0.001)   # micro-yield so Tk stays responsive
        yfit = model_frac(p, d['T_K'], fref, Tref, qp_model, include_offset)
        return weighted_mse(d['frac'], yfit, d['sigma'])

    return objective


def fit_diff_evolution(d, fref, Tref, ranges, qp_model, include_offset,
                       n_polish=200, de_maxiter=300, de_popsize=15,
                       stop_flag=None, progress_cb=None):
    bounds = [ranges[n] for n in PARAM_NAMES]
    objective = _make_objective(d, fref, Tref, qp_model, include_offset,
                                stop_flag)

    iters = [0]

    def de_callback(xk, *args, **kwargs):
        if stop_flag is not None and stop_flag.is_set():
            raise StopFitException()
        iters[0] += 1
        if progress_cb:
            progress_cb(iters[0], de_maxiter)

    de = differential_evolution(
        objective, bounds=bounds, popsize=de_popsize, maxiter=de_maxiter,
        tol=1e-4, workers=1, polish=False, disp=False, callback=de_callback)
    best_p, best_mse = de.x, de.fun

    if stop_flag is None or not stop_flag.is_set():
        if progress_cb:
            progress_cb(-1, -1)   # indeterminate during local polish
        try:
            loc = minimize(objective, best_p, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': n_polish, 'disp': False})
            if loc.fun < best_mse:
                best_p, best_mse = loc.x, loc.fun
        except StopFitException:
            raise
        except Exception:
            pass

    return list(best_p), float(best_mse)


def fit_random_restarts(d, fref, Tref, ranges, qp_model, include_offset,
                        n_attempts=1000, seed_params=None,
                        stop_flag=None, progress_cb=None):
    bounds = [ranges[n] for n in PARAM_NAMES]
    objective = _make_objective(d, fref, Tref, qp_model, include_offset,
                                stop_flag)
    best_p = list(seed_params) if seed_params is not None else None
    best_mse = np.inf
    if best_p is not None:
        best_mse = objective(best_p)

    for k in range(n_attempts):
        if progress_cb:
            progress_cb(k + 1, n_attempts)
        if stop_flag is not None and stop_flag.is_set():
            raise StopFitException()
        p0 = []
        for nm in PARAM_NAMES:
            lo, hi = ranges[nm]
            if nm in LOG_PARAMS and lo > 0 and hi > 0:
                p0.append(10 ** np.random.uniform(np.log10(lo), np.log10(hi)))
            else:
                p0.append(np.random.uniform(lo, hi))
        try:
            loc = minimize(objective, p0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 60})
            if loc.fun < best_mse:
                best_mse = loc.fun
                best_p = list(loc.x)
        except StopFitException:
            raise
        except Exception:
            continue

    return best_p, float(best_mse)


def fit_local_only(d, fref, Tref, ranges, qp_model, include_offset,
                   seed_params, stop_flag=None, progress_cb=None):
    bounds = [ranges[n] for n in PARAM_NAMES]
    objective = _make_objective(d, fref, Tref, qp_model, include_offset,
                                stop_flag)
    iters = [0]

    def loc_callback(xk, *args, **kwargs):
        if stop_flag is not None and stop_flag.is_set():
            raise StopFitException()
        iters[0] += 1
        if progress_cb:
            progress_cb(iters[0], 500)

    try:
        loc = minimize(objective, list(seed_params), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 500},
                       callback=loc_callback)
        return list(loc.x), float(loc.fun)
    except StopFitException:
        raise
    except Exception:
        return list(seed_params), float(objective(list(seed_params)))


# ===========================================================================
# Origin write helpers
# ===========================================================================
def get_or_make_sheet(book, name):
    ws = op.find_sheet('w', '[%s]%s' % (SRC_BOOK, name))
    if ws is not None:
        return ws
    try:
        return book.add_sheet(name)
    except Exception:
        book.add_sheet()
        new_ws = list(book)[-1]
        new_ws.name = name
        return new_ws


def clear_sheet(ws, n_cols):
    if ws.cols < n_cols:
        ws.cols = n_cols
    try:
        op.lt_exec('wks.col=1; wks.nCols=%d; del !%s;' % (n_cols, ws.lt_range()))
    except Exception:
        pass
    ws.cols = n_cols
    for c in range(n_cols):
        for t in ('L', 'U', 'C'):
            try:
                ws.set_label(c, '', type=t)
            except Exception:
                pass


def write_outputs_to_origin(book, filt, idx, params, fits_by_power,
                            Tref, qp_model, include_offset,
                            mse, r2, sse, sst, strategy, n_attempts,
                            export_points=200, dense_points=1000):
    """
    fits_by_power : dict  power -> (params, mse, fref)   for every fitted power
    params        : the params of the single 'active' power (slider state) used
                    for the global R2/SSE reporting
    """
    f = filt
    P_arr = f['P'][idx]
    powers = sorted(set(P_arr.tolist()))

    # ---- dense temperature grid spanning the *kept* data ----
    T_kept = f['T_K'][idx]
    T_min, T_max = float(np.min(T_kept)), float(np.max(T_kept))
    T_dense = np.linspace(T_min, T_max, dense_points)
    idx_exp = np.round(np.linspace(0, dense_points - 1,
                                   export_points)).astype(int)
    T_exp_mK = (T_dense[idx_exp] * 1e3).tolist()

    # =================== 1) TLS_Fit_Parameters ===================
    headers = (['Power'] + PARAM_NAMES
               + ['Chi2', 'QP_model', 'Include_offset', 'T_REF_K',
                  'Strategy', 'N_attempts'])
    ps = get_or_make_sheet(book, OUT_PARAMS_SHEET)
    clear_sheet(ps, len(headers))
    for i, hd in enumerate(headers):
        ps.set_label(i, hd, type='L')
    ps.set_label(0, 'dBm', type='U')
    if 'Tc' in PARAM_NAMES:
        ps.set_label(1 + PARAM_NAMES.index('Tc'), 'K', type='U')
    ps.set_label(headers.index('T_REF_K'), 'K', type='U')

    col_data = {hd: [] for hd in headers}
    for p in powers:
        pp, mm, fr = fits_by_power.get(p, (params, np.nan, np.nan))
        col_data['Power'].append(p)
        for nm, v in zip(PARAM_NAMES, pp):
            col_data[nm].append(v)
        col_data['Chi2'].append(mm)
        col_data['QP_model'].append(qp_model)
        col_data['Include_offset'].append(int(bool(include_offset)))
        col_data['T_REF_K'].append(Tref)
        col_data['Strategy'].append(strategy)
        col_data['N_attempts'].append(int(n_attempts))
    for i, hd in enumerate(headers):
        ps.from_list(i, col_data[hd])

    # =================== 2) Resonance_shift_total_Fit ===================
    # per power: TLS-only, QP-only, Total  (all referenced, x1e6)
    cs = get_or_make_sheet(book, OUT_CURVES_SHEET)
    clear_sheet(cs, 1 + 3 * len(powers))
    cs.set_label(0, 'Temperature', type='L')
    cs.set_label(0, 'mK', type='U')
    cs.from_list(0, T_exp_mK)
    for j, p in enumerate(powers):
        pp, _, fr = fits_by_power.get(p, (params, np.nan, np.nan))
        tls, qp, off, tot = model_breakdown(pp, T_dense, fr, Tref,
                                            qp_model, include_offset)
        base = 1 + 3 * j
        for k, (lab, arr) in enumerate([
                ('TLS', tls), ('QP', qp), ('Total', tot)]):
            col = base + k
            cs.set_label(col, '%s Df/f' % lab, type='L')
            cs.set_label(col, 'x1e-6', type='U')
            cs.set_label(col, '%g dBm' % p, type='C')
            cs.from_list(col, (arr[idx_exp] * 1e6).tolist())

    # =================== 3) TLS_Fit_Data ===================
    # measured fractional data + errors per power on the native T grid
    Ts = sorted(set(T_kept.tolist()))
    tls_ws = get_or_make_sheet(book, OUT_DATA_SHEET)
    clear_sheet(tls_ws, 1 + 2 * len(powers))
    tls_ws.set_label(0, 'Temperature', type='L')
    tls_ws.set_label(0, 'mK', type='U')
    tls_ws.from_list(0, [t * 1e3 for t in Ts])
    for j, p in enumerate(powers):
        d_col, e_col = [], []
        for T in Ts:
            m = (f['T_K'][idx] == T) & (P_arr == p)
            if np.any(m):
                ii = idx[np.where(m)[0][0]]
                d_col.append(float(f['frac'][ii]) * 1e6)
                e_col.append(float(f['sigma'][ii]) * 1e6)
            else:
                d_col.append(np.nan)
                e_col.append(np.nan)
        cd = 1 + j
        ce = 1 + len(powers) + j
        pp, _, _ = fits_by_power.get(p, (params, np.nan, np.nan))
        tls_ws.set_label(cd, 'Dfr/fr', type='L')
        tls_ws.set_label(cd, 'x1e-6', type='U')
        tls_ws.set_label(cd, '%g dBm, F0=%.2e' % (p, pp[0]), type='C')
        tls_ws.from_list(cd, d_col)
        tls_ws.set_label(ce, 'sigma(Dfr/fr)', type='L')
        tls_ws.set_label(ce, 'x1e-6', type='U')
        tls_ws.set_label(ce, '%g dBm' % p, type='C')
        tls_ws.from_list(ce, e_col)

    # =================== 4) Graph ===================
    gp = op.find_graph(OUT_GRAPH_NAME)
    if gp is None:
        try:
            gp = op.new_graph(template=OUT_GRAPH_TEMPLATE, lname=OUT_GRAPH_NAME)
        except Exception:
            gp = op.new_graph(lname=OUT_GRAPH_NAME)
        try:
            gp.name = OUT_GRAPH_NAME
        except Exception:
            pass
    else:
        try:
            for ly in gp:
                ly.remove()
            gp.add_layer()
        except Exception:
            pass
    ly = gp[0]
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
              '#0072B2', '#D55E00', '#CC79A7', '#999999', '#66A61E']
    shapes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for j, p in enumerate(powers):
        cd = 1 + j
        ce = 1 + len(powers) + j
        try:
            sc = ly.add_plot(tls_ws, cd, 0, type='s', colyerr=ce)
            sc.color = colors[j % len(colors)]
            try:
                sc.symbol_kind = shapes[j % len(shapes)]
                sc.symbol_size = 4
            except Exception:
                pass
            pp, _, _ = fits_by_power.get(p, (params, np.nan, np.nan))
            try:
                sc.name = '%g dBm, F0=%.2e' % (p, pp[0])
            except Exception:
                pass
        except Exception:
            pass
        try:
            total_col = 1 + 3 * j + 2
            ln = ly.add_plot(cs, total_col, 0, type='l')
            ln.color = colors[j % len(colors)]
            try:
                ln.line_width = 2
            except Exception:
                pass
        except Exception:
            pass
    try:
        ly.title = 'Resonance Shift Fit'
    except Exception:
        pass
    try:
        ly.rescale()
    except Exception:
        pass


# ===========================================================================
# GUI
# ===========================================================================
def eng(value):
    if value is None or not np.isfinite(value):
        return 'NaN'
    if value == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(value))) // 3 * 3)
    return '%.3fe%d' % (value / 10 ** exp, exp)


class ResShiftFitApp(tk.Toplevel):
    def __init__(self, master, raw):
        super().__init__(master)
        self.title("Resonance-Shift Fit  -  source: [%s]%s"
                   % (SRC_BOOK, TD_SHEET))
        self.geometry("1280x900")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.raw = raw
        self.param_ranges = dict(DEFAULT_RANGES)
        self.best_params = None
        self.best_mse = None
        self.fits_by_power = {}     # power -> (params, mse, fref)
        self.fit_thread = None
        self.stop_flag = threading.Event()
        self.gui_queue = queue.Queue()

        self.power_list = []
        self.temp_list = []
        self.power_checks = {}
        self.temp_checks = {}
        self.power_idx = {}
        self.temp_idx = {}
        self.scatter_artists = {}
        self.fit_line_artists = {}
        self.resid_artists = {}

        self._build_widgets()
        self._apply_filter()
        self._init_plots()
        self._poll_queue()

    # ---- thread-safe queue pump ----
    def _poll_queue(self):
        try:
            while True:
                msg = self.gui_queue.get_nowait()
                cmd = msg[0]
                if cmd == 'progress':
                    self._update_progress(msg[1], msg[2])
                elif cmd == 'finish':
                    self._fit_finished(msg[1], msg[2], msg[3], msg[4])
                elif cmd == 'status':
                    self.lbl_status.config(text=msg[1])
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(50, self._poll_queue)

    # ---- layout ----
    def _build_widgets(self):
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=8, pady=8)
        self.tab_setup = ttk.Frame(nb); nb.add(self.tab_setup, text='Setup')
        self.tab_fit = ttk.Frame(nb); nb.add(self.tab_fit, text='Fit & Plot')
        self.tab_model = ttk.Frame(nb); nb.add(self.tab_model,
                                               text='Model Equations')
        self._build_setup_tab(self.tab_setup)
        self._build_fit_tab(self.tab_fit)
        self._build_model_tab(self.tab_model)
        self.notebook = nb

    def _build_setup_tab(self, parent):
        flt = ttk.LabelFrame(parent, text='Pre-fit Filter', padding=10)
        flt.pack(fill='x', padx=10, pady=10)

        Pmin, Pmax = float(np.min(self.raw['P'])), float(np.max(self.raw['P']))
        Tmin, Tmax = (float(np.min(self.raw['T_mK'])),
                      float(np.max(self.raw['T_mK'])))

        ttk.Label(flt, text='Power range (dBm):').grid(row=0, column=0, sticky='e')
        self.var_pmin = tk.DoubleVar(self, value=Pmin)
        self.var_pmax = tk.DoubleVar(self, value=Pmax)
        ttk.Entry(flt, textvariable=self.var_pmin, width=10).grid(row=0, column=1, padx=4)
        ttk.Label(flt, text='to').grid(row=0, column=2)
        ttk.Entry(flt, textvariable=self.var_pmax, width=10).grid(row=0, column=3, padx=4)
        ttk.Label(flt, text='(data: %g .. %g)' % (Pmin, Pmax)
                  ).grid(row=0, column=4, padx=8, sticky='w')

        ttk.Label(flt, text='Temperature range (mK):').grid(row=1, column=0, sticky='e')
        self.var_tmin = tk.DoubleVar(self, value=Tmin)
        self.var_tmax = tk.DoubleVar(self, value=Tmax)
        ttk.Entry(flt, textvariable=self.var_tmin, width=10).grid(row=1, column=1, padx=4)
        ttk.Label(flt, text='to').grid(row=1, column=2)
        ttk.Entry(flt, textvariable=self.var_tmax, width=10).grid(row=1, column=3, padx=4)
        ttk.Label(flt, text='(data: %g .. %g)' % (Tmin, Tmax)
                  ).grid(row=1, column=4, padx=8, sticky='w')

        ttk.Button(flt, text='Apply Filter', command=self._on_apply_filter
                   ).grid(row=0, column=5, rowspan=2, padx=10)

        # ---- model options ----
        mo = ttk.LabelFrame(parent, text='Model Options', padding=10)
        mo.pack(fill='x', padx=10, pady=10)

        ttk.Label(mo, text='Reference temperature T_REF (K):'
                  ).grid(row=0, column=0, sticky='e', padx=4, pady=2)
        self.var_tref = tk.DoubleVar(self, value=DEFAULT_T_REF)
        ttk.Entry(mo, textvariable=self.var_tref, width=12
                  ).grid(row=0, column=1, padx=4, pady=2, sticky='w')
        ttk.Label(mo, text='(curves are referenced to pass through 0 here)'
                  ).grid(row=0, column=2, columnspan=2, sticky='w', padx=8)

        ttk.Label(mo, text='QP model:').grid(row=1, column=0, sticky='ne',
                                             padx=4, pady=2)
        self.var_qp = tk.StringVar(self, value='bcs')
        qp_frame = ttk.Frame(mo)
        qp_frame.grid(row=1, column=1, columnspan=3, sticky='w')
        for r, (val, lab) in enumerate(QP_MODELS):
            ttk.Radiobutton(qp_frame, text=lab, variable=self.var_qp,
                            value=val, command=self._on_model_opt_change
                            ).grid(row=r, column=0, sticky='w', padx=2, pady=1)

        self.var_offset = tk.BooleanVar(self, value=True)
        ttk.Checkbutton(mo, text='Include constant offset',
                        variable=self.var_offset,
                        command=self._on_model_opt_change
                        ).grid(row=2, column=0, columnspan=2, sticky='w',
                               padx=4, pady=(8, 2))

        # ---- parameter ranges ----
        rng = ttk.LabelFrame(parent, text='Parameter Ranges', padding=10)
        rng.pack(fill='x', padx=10, pady=10)
        for c, head in enumerate(['Parameter', 'Min', 'Max']):
            ttk.Label(rng, text=head, font=('Arial', 10, 'bold')
                      ).grid(row=0, column=c, padx=6, pady=4)
        self.range_vars = {}
        for i, nm in enumerate(PARAM_NAMES):
            lo, hi = self.param_ranges[nm]
            vmin = tk.DoubleVar(self, value=lo)
            vmax = tk.DoubleVar(self, value=hi)
            self.range_vars[nm] = (vmin, vmax)
            label = nm + (' (%s)' % UNITS[nm] if nm in UNITS else '')
            ttk.Label(rng, text=label).grid(row=i + 1, column=0, sticky='e',
                                            padx=6, pady=2)
            ttk.Entry(rng, textvariable=vmin, width=16).grid(row=i + 1, column=1, padx=4, pady=2)
            ttk.Entry(rng, textvariable=vmax, width=16).grid(row=i + 1, column=2, padx=4, pady=2)

        # ---- strategy ----
        st = ttk.LabelFrame(parent, text='Fitting Strategy', padding=10)
        st.pack(fill='x', padx=10, pady=10)
        self.var_strategy = tk.StringVar(self, value='de_local')
        ttk.Radiobutton(st, text='Differential Evolution + L-BFGS-B polish',
                        variable=self.var_strategy, value='de_local'
                        ).grid(row=0, column=0, sticky='w', padx=6, pady=2)
        ttk.Radiobutton(st, text='Random restarts (L-BFGS-B from each start)',
                        variable=self.var_strategy, value='random'
                        ).grid(row=1, column=0, sticky='w', padx=6, pady=2)
        ttk.Radiobutton(st, text='Local polish only (from current sliders)',
                        variable=self.var_strategy, value='local'
                        ).grid(row=2, column=0, sticky='w', padx=6, pady=2)
        ttk.Label(st, text='Random restarts / polish iterations:'
                  ).grid(row=0, column=1, padx=20, sticky='e')
        self.var_attempts = tk.IntVar(self, value=500)
        ttk.Entry(st, textvariable=self.var_attempts, width=10
                  ).grid(row=0, column=2, padx=4, sticky='w')

        sb = ttk.LabelFrame(parent, text='Filtered Dataset Summary', padding=10)
        sb.pack(fill='x', padx=10, pady=10)
        self.lbl_summary = ttk.Label(sb, text='(no filter applied yet)',
                                     font=('Consolas', 10), justify='left')
        self.lbl_summary.pack(anchor='w')

    def _build_fit_tab(self, parent):
        top = ttk.Frame(parent); top.pack(fill='x', padx=10, pady=6)
        self.btn_fit = ttk.Button(top, text='Run Fit (selected powers)',
                                  command=self._on_run_fit)
        self.btn_fit.pack(side='left', padx=4)
        self.btn_stop = ttk.Button(top, text='Stop', command=self._on_stop_fit,
                                   state='disabled')
        self.btn_stop.pack(side='left', padx=4)
        ttk.Button(top, text='Reset Sliders to Range Mid',
                   command=self._reset_sliders_mid).pack(side='left', padx=4)
        ttk.Button(top, text='Export to Origin',
                   command=self._on_export).pack(side='left', padx=12)

        self.progress_var = tk.DoubleVar(self)
        self.progress = ttk.Progressbar(top, variable=self.progress_var,
                                        maximum=100, length=200)
        self.progress.pack(side='right', padx=10)
        self.lbl_status = ttk.Label(top, text='', foreground='#005')
        self.lbl_status.pack(side='right', padx=10)

        paned = ttk.PanedWindow(parent, orient='vertical')
        paned.pack(fill='both', expand=True, padx=10, pady=6)

        controls_outer = ttk.Frame(paned)
        paned.add(controls_outer, weight=1)
        ctrl_canvas = tk.Canvas(controls_outer, highlightthickness=0,
                                borderwidth=0)
        ctrl_scroll = ttk.Scrollbar(controls_outer, orient='vertical',
                                    command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_scroll.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)
        controls = ttk.Frame(ctrl_canvas)
        controls_window = ctrl_canvas.create_window((0, 0), window=controls,
                                                    anchor='nw')

        def _ctrl_configure(_e):
            ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all'))
            ctrl_canvas.itemconfigure(controls_window,
                                      width=ctrl_canvas.winfo_width())
        controls.bind('<Configure>', _ctrl_configure)
        ctrl_canvas.bind('<Configure>', _ctrl_configure)

        def _on_wheel(event):
            delta = -1 if event.delta > 0 else 1
            if hasattr(event, 'num') and event.num in (4, 5):
                delta = -1 if event.num == 4 else 1
            ctrl_canvas.yview_scroll(delta, 'units')
        ctrl_canvas.bind('<Enter>',
                         lambda e: ctrl_canvas.bind_all('<MouseWheel>', _on_wheel))
        ctrl_canvas.bind('<Leave>',
                         lambda e: ctrl_canvas.unbind_all('<MouseWheel>'))

        sl = ttk.LabelFrame(controls, text='Manual Parameter Adjustment',
                            padding=8)
        sl.pack(fill='x', padx=4, pady=(2, 6))
        self.slider_vars = {}
        self.slider_labels = {}
        self.slider_widgets = {}
        for i, nm in enumerate(PARAM_NAMES):
            lo, hi = self.param_ranges[nm]
            ttk.Label(sl, text=nm + ':', font=('Arial', 9, 'bold')
                      ).grid(row=i, column=0, padx=(8, 4), pady=4, sticky='e')
            dv = tk.DoubleVar(self)
            self.slider_vars[nm] = dv
            if nm in LOG_PARAMS and lo > 0 and hi > 0:
                dv.set((np.log10(lo) + np.log10(hi)) / 2)
                frm, to = np.log10(lo), np.log10(hi)
            else:
                dv.set((lo + hi) / 2)
                frm, to = lo, hi
            sld = ttk.Scale(sl, variable=dv, from_=frm, to=to,
                            orient='horizontal', length=300,
                            command=lambda v, name=nm: self._on_slider(name, v))
            sld.grid(row=i, column=1, padx=4, pady=4)
            self.slider_widgets[nm] = sld
            vlbl = ttk.Label(sl, text='', font=('Consolas', 9), width=16)
            vlbl.grid(row=i, column=2, padx=(4, 14), pady=4, sticky='w')
            self.slider_labels[nm] = vlbl
            self._refresh_slider_label(nm)

        self.ds_frame = ttk.LabelFrame(
            controls,
            text='Powers and Temperatures to include   '
                 '(a point is fit only if BOTH its power and its temperature '
                 'are checked)',
            padding=8)
        self.ds_frame.pack(fill='x', padx=4, pady=(2, 6))

        plot_frame = ttk.LabelFrame(paned, text='Fit Plot & Residuals',
                                    padding=4)
        paned.add(plot_frame, weight=4)
        self.plot_container = ttk.Frame(plot_frame)
        self.plot_container.pack(fill='both', expand=True)
        self.fig = None; self.canvas = None
        self.ax_main = None; self.ax_resid = None

        def _set_initial_sash(_e=None):
            try:
                total_h = paned.winfo_height()
                if total_h > 200:
                    paned.sashpos(0, min(340, total_h // 3))
            except Exception:
                pass
        paned.bind('<Map>', _set_initial_sash)
        self.after(200, _set_initial_sash)

    def _build_model_tab(self, parent):
        fig_eq = plt.Figure(figsize=(8, 6), facecolor='white')
        ax = fig_eq.add_subplot(111); ax.axis('off')

        eqs = [
            ("Fractional resonance shift (referenced to T_REF):", 0.93,
             r"$\frac{\Delta f_r}{f_r}(T) = [\delta_{TLS}(T)-\delta_{TLS}(T_{ref})]"
             r"+[\delta_{QP}(T)-\delta_{QP}(T_{ref})]+\mathrm{offset}$", 0.85),
            ("TLS shift:", 0.74,
             r"$\delta_{TLS}(T)=\frac{F\delta_{TLS0}}{\pi}\left[\mathrm{Re}\,\psi"
             r"\left(\frac{1}{2}-\frac{1}{2\pi i}\frac{h f_r}{k_B T}\right)"
             r"-\ln\frac{h f_r}{2\pi k_B T}\right]$", 0.66),
            ("QP - dirty limit:", 0.55,
             r"$\delta_{QP}^{dirty}(T)=-\frac{\alpha}{2}\left(\frac{k_B T}{\Delta(T)}\right)^{4}$", 0.48),
            ("QP - clean / BCS:", 0.38,
             r"$\delta_{QP}^{BCS}(T)=-\alpha\left[\left(\sin\frac{\phi}{2}+\cos\frac{\phi}{2}\right)"
             r"\left(\frac{|\sigma|}{\sigma_2^{(0)}}\right)^{-1/2}-1\right],\ \sigma=\sigma_1+i\sigma_2$", 0.30),
            ("BCS gap (closed-form):", 0.19,
             r"$\Delta(T)=\Delta_0\tanh\left(\frac{\pi k_B T_c}{\Delta_0}\sqrt{1.74\left(\frac{T_c}{T}-1\right)}\right),"
             r"\ \Delta_0=1.764\,k_B T_c$", 0.11),
        ]
        for title, ty, eq, ey in eqs:
            ax.text(0.5, ty, title, fontsize=12, ha='center', va='center',
                    fontweight='bold')
            ax.text(0.5, ey, eq, fontsize=15, ha='center', va='center')
        fig_eq.subplots_adjust(top=0.97, bottom=0.03)
        canvas_eq = FigureCanvasTkAgg(fig_eq, master=parent)
        canvas_eq.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
        canvas_eq.draw()

    # ---- filtering ----
    def _on_model_opt_change(self):
        self._refresh_plot()

    def _on_apply_filter(self):
        try:
            self._apply_filter()
        except Exception as e:
            messagebox.showerror("Filter error", str(e), parent=self)
            return
        self._rebuild_dataset_checkboxes()
        if self.canvas is not None:
            self._init_plots()

    def _apply_filter(self):
        Pmin, Pmax = self.var_pmin.get(), self.var_pmax.get()
        Tmin, Tmax = self.var_tmin.get(), self.var_tmax.get()
        if Pmin > Pmax: Pmin, Pmax = Pmax, Pmin
        if Tmin > Tmax: Tmin, Tmax = Tmax, Tmin
        m = ((self.raw['P'] >= Pmin) & (self.raw['P'] <= Pmax) &
             (self.raw['T_mK'] >= Tmin) & (self.raw['T_mK'] <= Tmax))
        self.filt = {k: v[m] for k, v in self.raw.items()}
        self.power_list = sorted(set(self.filt['P'].tolist()))
        self.temp_list = sorted(set(self.filt['T_mK'].tolist()))
        self.power_idx = {p: np.where(self.filt['P'] == p)[0]
                          for p in self.power_list}
        self.temp_idx = {t: np.where(self.filt['T_mK'] == t)[0]
                         for t in self.temp_list}
        n_pts = len(self.filt['frac'])
        self.lbl_summary.config(text=(
            'Points: %d   |   distinct powers: %d   |   '
            'distinct temperatures: %d\n'
            'Power kept: [%g, %g] dBm     Temperature kept: [%g, %g] mK'
        ) % (n_pts, len(self.power_list), len(self.temp_list),
             Pmin, Pmax, Tmin, Tmax))
        if hasattr(self, 'ds_frame'):
            self._rebuild_dataset_checkboxes()

    def _rebuild_dataset_checkboxes(self):
        if not hasattr(self, 'ds_frame'):
            return
        for w in self.ds_frame.winfo_children():
            w.destroy()
        self.power_checks.clear()
        self.temp_checks.clear()

        pw_box = ttk.LabelFrame(self.ds_frame, text='Powers (dBm)', padding=6)
        pw_box.grid(row=0, column=0, sticky='nsew', padx=6, pady=4)
        tp_box = ttk.LabelFrame(self.ds_frame, text='Temperatures (mK)',
                                padding=6)
        tp_box.grid(row=0, column=1, sticky='nsew', padx=6, pady=4)
        self.ds_frame.grid_columnconfigure(0, weight=1)
        self.ds_frame.grid_columnconfigure(1, weight=1)

        pw_btns = ttk.Frame(pw_box)
        pw_btns.grid(row=0, column=0, columnspan=8, sticky='w', pady=(0, 4))
        ttk.Button(pw_btns, text='All', width=5,
                   command=lambda: self._set_all_powers(True)).pack(side='left', padx=2)
        ttk.Button(pw_btns, text='None', width=5,
                   command=lambda: self._set_all_powers(False)).pack(side='left', padx=2)
        tp_btns = ttk.Frame(tp_box)
        tp_btns.grid(row=0, column=0, columnspan=8, sticky='w', pady=(0, 4))
        ttk.Button(tp_btns, text='All', width=5,
                   command=lambda: self._set_all_temps(True)).pack(side='left', padx=2)
        ttk.Button(tp_btns, text='None', width=5,
                   command=lambda: self._set_all_temps(False)).pack(side='left', padx=2)

        # Powers default OFF so you opt in to exactly the ones you want.
        cols_pw = 4
        for i, p in enumerate(self.power_list):
            v = tk.BooleanVar(self, value=False)
            self.power_checks[p] = v
            ttk.Checkbutton(pw_box, text='%g' % p, variable=v,
                            command=self._on_dataset_toggle
                            ).grid(row=1 + i // cols_pw, column=i % cols_pw,
                                   sticky='w', padx=6, pady=1)
        # Temperatures default ON.
        cols_tp = 6
        for i, t in enumerate(self.temp_list):
            v = tk.BooleanVar(self, value=True)
            self.temp_checks[t] = v
            ttk.Checkbutton(tp_box, text='%g' % t, variable=v,
                            command=self._on_dataset_toggle
                            ).grid(row=1 + i // cols_tp, column=i % cols_tp,
                                   sticky='w', padx=6, pady=1)

    def _set_all_powers(self, value):
        for v in self.power_checks.values():
            v.set(bool(value))
        self._on_dataset_toggle()

    def _set_all_temps(self, value):
        for v in self.temp_checks.values():
            v.set(bool(value))
        self._on_dataset_toggle()

    def _checked_powers(self):
        return [p for p in self.power_list
                if self.power_checks.get(p) and self.power_checks[p].get()]

    def _checked_temps(self):
        return [t for t in self.temp_list
                if self.temp_checks.get(t) and self.temp_checks[t].get()]

    def _checked_indices(self):
        powers_on = self._checked_powers()
        temps_on = self._checked_temps()
        if not powers_on or not temps_on:
            return np.array([], dtype=int)
        f = self.filt
        mask = np.isin(f['P'], powers_on) & np.isin(f['T_mK'], temps_on)
        return np.where(mask)[0]

    def _on_dataset_toggle(self):
        if self.ax_main is None:
            return
        powers_on = set(self._checked_powers())
        for p, cont in self.scatter_artists.items():
            vis = (p in powers_on)
            cont[0].set_visible(vis)
            for ln in cont[1]:
                ln.set_visible(vis)
            for ln in cont[2]:
                ln.set_visible(vis)
            if p in self.fit_line_artists:
                self.fit_line_artists[p].set_visible(vis)
            if p in self.resid_artists:
                self.resid_artists[p].set_visible(vis)
        self._refresh_plot()

    # ---- sliders ----
    def _slider_value_to_param(self, nm):
        v = self.slider_vars[nm].get()
        return 10 ** float(v) if nm in LOG_PARAMS else float(v)

    def _set_param_to_slider(self, nm, value):
        if nm in LOG_PARAMS and value > 0:
            self.slider_vars[nm].set(np.log10(value))
        else:
            self.slider_vars[nm].set(value)

    def _refresh_slider_label(self, nm):
        v = self._slider_value_to_param(nm)
        txt = eng(v) if nm in LOG_PARAMS else '%.5g' % v
        self.slider_labels[nm].config(text=txt)

    def _on_slider(self, nm, _val):
        self._refresh_slider_label(nm)
        self._refresh_plot()

    def _reset_sliders_mid(self):
        for nm in PARAM_NAMES:
            lo = self.range_vars[nm][0].get()
            hi = self.range_vars[nm][1].get()
            self.param_ranges[nm] = (lo, hi)
            if nm in LOG_PARAMS and lo > 0 and hi > 0:
                lo_, hi_ = np.log10(lo), np.log10(hi)
                self.slider_widgets[nm].configure(from_=lo_, to=hi_)
                self.slider_vars[nm].set((lo_ + hi_) / 2)
            else:
                self.slider_widgets[nm].configure(from_=lo, to=hi)
                self.slider_vars[nm].set((lo + hi) / 2)
            self._refresh_slider_label(nm)
        self._refresh_plot()

    def _current_params(self):
        return [self._slider_value_to_param(nm) for nm in PARAM_NAMES]

    # ---- plotting ----
    def _init_plots(self):
        for w in self.plot_container.winfo_children():
            w.destroy()
        self.fig, (self.ax_main, self.ax_resid) = plt.subplots(
            2, 1, figsize=(9, 5),
            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        self.fig.patch.set_facecolor('white')
        self.ax_main.set_ylabel(r'$\Delta f_r/f_r$  ($\times10^{-6}$)')
        self.ax_resid.set_ylabel('weighted resid')
        self.ax_resid.set_xlabel('Temperature (mK)')
        self.ax_main.grid(True, which='both', alpha=0.3)
        self.ax_resid.grid(True, alpha=0.3)
        self.ax_resid.axhline(0, color='k', lw=0.7)

        self.scatter_artists.clear()
        self.fit_line_artists.clear()
        self.resid_artists.clear()

        cmap = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for p in self.power_list:
            color = next(cmap)
            idx = self.power_idx[p]
            T_mK = self.filt['T_mK'][idx]
            y = self.filt['frac'][idx] * 1e6
            s = self.filt['sigma'][idx] * 1e6
            order = np.argsort(T_mK)
            T_mK, y, s = T_mK[order], y[order], s[order]
            cont = self.ax_main.errorbar(T_mK, y, yerr=s, fmt='o', ms=4,
                                         alpha=0.85, color=color, capsize=2,
                                         label='%g dBm' % p)
            self.scatter_artists[p] = cont
            ln, = self.ax_main.plot([], [], lw=1.8, color=color)
            self.fit_line_artists[p] = ln
            rln, = self.ax_resid.plot([], [], 'o-', ms=3, lw=0.8, color=color)
            self.resid_artists[p] = rln
            # start hidden (powers default OFF)
            cont[0].set_visible(False)
            for el in cont[1]:
                el.set_visible(False)
            for el in cont[2]:
                el.set_visible(False)
            ln.set_visible(False)
            rln.set_visible(False)

        n_pow = len(self.power_list)
        if 0 < n_pow <= 24:
            ncol = 1 if n_pow <= 8 else 2
            self.ax_main.legend(fontsize=8, loc='upper left',
                                bbox_to_anchor=(1.01, 1.0), ncol=ncol,
                                handletextpad=0.4, columnspacing=0.8,
                                borderaxespad=0.2)
        right_margin = 0.78 if n_pow > 8 else 0.83
        self.fig.subplots_adjust(left=0.11, right=right_margin, top=0.92,
                                 bottom=0.13, hspace=0.10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(self.canvas, self.plot_container).update()
        self._on_dataset_toggle()

    def _refresh_plot(self):
        if self.ax_main is None:
            return
        params = self._current_params()
        Tref = float(self.var_tref.get())
        qp_model = self.var_qp.get()
        include_offset = bool(self.var_offset.get())
        powers_on = set(self._checked_powers())
        temps_on = set(self._checked_temps())

        used_T = []
        all_resid = []

        for p in self.power_list:
            ln = self.fit_line_artists[p]
            rln = self.resid_artists[p]
            cont = self.scatter_artists[p]
            if p not in powers_on:
                ln.set_data([], [])
                rln.set_data([], [])
                continue

            idx_all = self.power_idx[p]
            T_mK_all = self.filt['T_mK'][idx_all]
            keep = np.array([t in temps_on for t in T_mK_all], dtype=bool)
            idx = idx_all[keep]
            if idx.size == 0:
                ln.set_data([], [])
                rln.set_data([], [])
                cont[0].set_data([], [])
                continue

            T = self.filt['T_K'][idx]
            y = self.filt['frac'][idx]
            s = self.filt['sigma'][idx]
            order = np.argsort(T)
            T_s, y_s, s_s = T[order], y[order], s[order]

            fref = fref_for_power(self.filt, p, Tref)
            cont[0].set_data(T_s * 1e3, y_s * 1e6)

            # smooth model curve over this power's T span
            T_line = np.linspace(T_s.min(), T_s.max(), 400)
            yfit_line = model_frac(params, T_line, fref, Tref, qp_model,
                                   include_offset)
            ln.set_data(T_line * 1e3, yfit_line * 1e6)

            yfit_pts = model_frac(params, T_s, fref, Tref, qp_model,
                                  include_offset)
            resid = (y_s - yfit_pts) / np.maximum(s_s, 1e-30)
            rln.set_data(T_s * 1e3, resid)
            used_T.append(T_s * 1e3)
            all_resid.append(resid)

        idx_on = self._checked_indices()
        mse = self._mse_for_indices(idx_on, params, Tref, qp_model,
                                    include_offset) if idx_on.size else float('nan')
        title = 'Manual overlay   |   weighted MSE = %.3e' % mse
        if self.best_mse is not None:
            title += '   |   last fit MSE = %.3e' % self.best_mse
        self.ax_main.set_title(title, fontsize=10)

        if used_T:
            allT = np.concatenate(used_T)
            self.ax_main.set_xlim(allT.min() * 0.97, allT.max() * 1.03)
        if all_resid:
            r = np.concatenate(all_resid)
            r = r[np.isfinite(r)]
            if r.size:
                m = max(3, np.percentile(np.abs(r), 99))
                self.ax_resid.set_ylim(-m * 1.1, m * 1.1)
        self.ax_main.relim()
        self.ax_main.autoscale_view(scalex=False)
        self.canvas.draw_idle()

    def _mse_for_indices(self, idx, params, Tref, qp_model, include_offset):
        f = self.filt
        # group by power (each power has its own fref)
        total_w = 0.0
        acc = 0.0
        for p in set(f['P'][idx].tolist()):
            sub = idx[f['P'][idx] == p]
            fref = fref_for_power(self.filt, p, Tref)
            yfit = model_frac(params, f['T_K'][sub], fref, Tref, qp_model,
                              include_offset)
            w = 1.0 / np.maximum(f['sigma'][sub] ** 2, 1e-30)
            acc += np.sum(w * (f['frac'][sub] - yfit) ** 2)
            total_w += np.sum(w)
        return float(acc / total_w) if total_w > 0 else float('nan')

    # ---- fit orchestration ----
    def _on_run_fit(self):
        idx = self._checked_indices()
        powers_on = self._checked_powers()
        if idx.size < 3 or not powers_on:
            messagebox.showwarning(
                "No data", "Check at least one power (and some temperatures).",
                parent=self)
            return

        ranges = {}
        for nm, (vmin, vmax) in self.range_vars.items():
            lo, hi = float(vmin.get()), float(vmax.get())
            if lo >= hi:
                messagebox.showerror("Bad range",
                                     "Parameter %s: min must be < max." % nm,
                                     parent=self)
                return
            ranges[nm] = (lo, hi)
            self.param_ranges[nm] = (lo, hi)
            if nm in LOG_PARAMS and lo > 0 and hi > 0:
                self.slider_widgets[nm].configure(from_=np.log10(lo),
                                                  to=np.log10(hi))
            else:
                self.slider_widgets[nm].configure(from_=lo, to=hi)

        strategy = self.var_strategy.get()
        n_attempts = int(self.var_attempts.get())
        Tref = float(self.var_tref.get())
        qp_model = self.var_qp.get()
        include_offset = bool(self.var_offset.get())

        # Read all Tk variables HERE on the main thread (they are not
        # thread-safe). Build per-power data dicts + frefs + the seed.
        seed_params = self._current_params()
        clamped_seed = []
        for nm, v in zip(PARAM_NAMES, seed_params):
            lo, hi = ranges[nm]
            clamped_seed.append(min(max(float(v), lo), hi))

        f = self.filt
        per_power = []
        for p in powers_on:
            sub = idx[f['P'][idx] == p]
            if sub.size < 3:
                continue
            d = {
                'T_K':   f['T_K'][sub],
                'frac':  f['frac'][sub],
                'sigma': f['sigma'][sub],
            }
            fref = fref_for_power(self.filt, p, Tref)
            per_power.append((p, d, fref))

        if not per_power:
            messagebox.showwarning(
                "Not enough points",
                "Each selected power needs at least 3 checked temperatures.",
                parent=self)
            return

        self.stop_flag.clear()
        self.btn_fit.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.lbl_status.config(text='fitting %d power(s)...' % len(per_power))
        self.progress_var.set(0)
        self.progress.configure(mode='determinate')

        def worker():
            try:
                results = {}
                n_pw = len(per_power)
                for pi, (p, d, fref) in enumerate(per_power):

                    def thread_safe_progress(curr, tot, _pi=pi, _np=n_pw):
                        # blend per-power progress into overall bar
                        if tot and tot > 0:
                            frac = (_pi + curr / tot) / _np * 100.0
                            self.gui_queue.put(('progress', frac, 100))
                        else:
                            self.gui_queue.put(('progress', -1, -1))

                    self.gui_queue.put(
                        ('status', 'fitting %g dBm (%d/%d)...'
                         % (p, pi + 1, n_pw)))

                    if strategy == 'de_local':
                        pp, mm = fit_diff_evolution(
                            d, fref, Tref, ranges, qp_model, include_offset,
                            n_polish=max(50, n_attempts // 5),
                            de_maxiter=300, de_popsize=15,
                            stop_flag=self.stop_flag,
                            progress_cb=thread_safe_progress)
                    elif strategy == 'random':
                        pp, mm = fit_random_restarts(
                            d, fref, Tref, ranges, qp_model, include_offset,
                            n_attempts=n_attempts, seed_params=clamped_seed,
                            stop_flag=self.stop_flag,
                            progress_cb=thread_safe_progress)
                    else:
                        pp, mm = fit_local_only(
                            d, fref, Tref, ranges, qp_model, include_offset,
                            clamped_seed, stop_flag=self.stop_flag,
                            progress_cb=thread_safe_progress)
                    results[p] = (list(pp), float(mm), float(fref))

                self.gui_queue.put(('finish', results, Tref, None, None))
            except StopFitException:
                self.gui_queue.put(('finish', None, None, None, None))
            except Exception:
                tb = traceback.format_exc()
                self.gui_queue.put(('finish', None, None, None,
                                    "Strategy: %s\n\n%s" % (strategy, tb)))

        self.fit_thread = threading.Thread(target=worker, daemon=True)
        self.fit_thread.start()

    def _on_stop_fit(self):
        self.stop_flag.set()
        self.lbl_status.config(text='stopping...')

    def _update_progress(self, current, total):
        if total is None or total <= 0:
            self.progress.configure(mode='indeterminate')
            self.progress.start(10)
        else:
            self.progress.configure(mode='determinate')
            self.progress.stop()
            self.progress_var.set(max(0, min(100, current)))

    def _fit_finished(self, results, Tref, _unused, err):
        self.btn_fit.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.progress.stop()
        self.progress.configure(mode='determinate')
        self.progress_var.set(100 if results else 0)

        if err is not None:
            self.lbl_status.config(text='fit error')
            messagebox.showerror("Fit error", err, parent=self)
            return
        if not results:
            self.lbl_status.config(text='fit cancelled')
            return

        self.fits_by_power.update(results)
        # set sliders to the best-MSE power so the overlay shows a good fit
        best_p = min(results, key=lambda k: results[k][1])
        best_params, best_mse, _ = results[best_p]
        self.best_params = list(best_params)
        self.best_mse = float(best_mse)
        for nm, val in zip(PARAM_NAMES, best_params):
            self._set_param_to_slider(nm, float(val))
            self._refresh_slider_label(nm)
        msg = '   '.join('%g dBm: MSE=%.2e' % (p, results[p][1])
                         for p in sorted(results))
        self.lbl_status.config(text='done.  ' + msg)
        self._refresh_plot()
        self.notebook.select(self.tab_fit)

    # ---- export ----
    def _on_export(self):
        idx = self._checked_indices()
        powers_on = self._checked_powers()
        if idx.size == 0 or not powers_on:
            messagebox.showwarning("No data", "Check some powers first.",
                                   parent=self)
            return
        Tref = float(self.var_tref.get())
        qp_model = self.var_qp.get()
        include_offset = bool(self.var_offset.get())
        slider_params = self._current_params()

        # Ensure every selected power has params: if not fitted, use sliders.
        fits = {}
        for p in powers_on:
            if p in self.fits_by_power:
                fits[p] = self.fits_by_power[p]
            else:
                fref = fref_for_power(self.filt, p, Tref)
                fits[p] = (list(slider_params), float('nan'), float(fref))

        # global goodness on the active/best params (slider state)
        f = self.filt
        gw = ga = gsse = 0.0
        ymean = float(np.mean(f['frac'][idx]))
        gsst = float(np.sum((f['frac'][idx] - ymean) ** 2))
        for p in powers_on:
            sub = idx[f['P'][idx] == p]
            pp, _, fr = fits[p]
            yfit = model_frac(pp, f['T_K'][sub], fr, Tref, qp_model,
                              include_offset)
            w = 1.0 / np.maximum(f['sigma'][sub] ** 2, 1e-30)
            ga += np.sum(w * (f['frac'][sub] - yfit) ** 2)
            gw += np.sum(w)
            gsse += np.sum((f['frac'][sub] - yfit) ** 2)
        mse = float(ga / gw) if gw > 0 else float('nan')
        r2 = 1.0 - gsse / gsst if gsst > 0 else float('nan')

        try:
            book = op.find_book('w', SRC_BOOK)
            if book is None:
                raise RuntimeError("Workbook '%s' not found." % SRC_BOOK)
            self.lbl_status.config(text='exporting to Origin...')
            self.update_idletasks()
            write_outputs_to_origin(
                book, self.filt, idx, slider_params, fits,
                Tref, qp_model, include_offset,
                mse, r2, gsse, gsst,
                self.var_strategy.get(), int(self.var_attempts.get()))
            self.lbl_status.config(text='exported   MSE=%.3e  R2=%.4f'
                                   % (mse, r2))
            messagebox.showinfo(
                "Exported",
                "Results written to workbook '%s'.\n"
                "Sheets: %s, %s, %s\nGraph: %s" % (
                    SRC_BOOK, OUT_PARAMS_SHEET, OUT_DATA_SHEET,
                    OUT_CURVES_SHEET, OUT_GRAPH_NAME),
                parent=self)
        except Exception:
            tb = traceback.format_exc()
            self.lbl_status.config(text='export failed')
            messagebox.showerror("Export error", tb, parent=self)

    def _on_close(self):
        self.stop_flag.set()
        try:
            self.destroy()
        except Exception:
            pass


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    try:
        raw = read_origin_source()
    except Exception as e:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Cannot read source", str(e))
        root.destroy()
        return
    root = tk.Tk(); root.withdraw()
    app = ResShiftFitApp(root, raw)
    root.wait_window(app)
    try:
        root.destroy()
    except Exception:
        pass


if __name__ == '__main__':
    main()