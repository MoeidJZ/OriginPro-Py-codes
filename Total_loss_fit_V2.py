"""
Interactive Loss-Fit GUI for OriginPro (embedded Python).

Run this from Origin's Python window. It:
  1) reads data from workbook 'LossRes1' (sheets: photon_numbers, freq_values,
     delta, delta_error) -- same layout as the old Total_loss_fit.py script,
  2) opens a Tkinter GUI where you can:
       * trim the data by power range and temperature range BEFORE fitting,
       * pick which (power, temperature) datasets to include via checkboxes,
       * edit parameter ranges,
       * choose fitting strategy (Differential Evolution + local polish, or
         random sampling, or local polish only),
       * stop a running fit,
       * adjust each parameter with a slider and see the model overlay update
         live, with a residuals panel below,
  3) on "Export to Origin", writes results back to the same workbook
     (FitParams / Loss_Raw / Loss_Fitted / Loss_Breakdown sheets) and creates
     a graph 'LossPlot' from the 'TotalLoss' template -- overwriting any
     previous run.

The GUI is modal: while it is open you cannot interact with Origin. Close it
to return control to Origin.
"""

# ---------------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------------
SRC_BOOK         = 'LossRes1'

# Output target names (fixed -- re-running a fit overwrites these).
OUT_PARAMS_SHEET    = 'FitParams'
OUT_RAW_SHEET       = 'Loss_Raw'
OUT_FIT_SHEET       = 'Loss_Fitted'
OUT_BREAK_SHEET     = 'Loss_Breakdown'
OUT_GRAPH_NAME      = 'LossPlot'
OUT_GRAPH_TEMPLATE  = 'TotalLoss'   # falls back to default template if missing

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
from scipy.special import kn
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d

import originpro as op

h, kB = constants.h, constants.k

PARAM_NAMES = ['F_delta_TLS0', 'D', 'beta1', 'beta2', 'AQP', 'Tc', 'Q_other']
LOG_PARAMS  = {'F_delta_TLS0', 'D', 'AQP', 'Q_other'}   # sliders are log-scale

DEFAULT_RANGES = {
    'F_delta_TLS0': (1e-7, 1e-4),
    'D':            (0.1,  1e3),
    'beta1':        (0.01, 2.0),
    'beta2':        (0.1,  1.5),
    'AQP':          (1e-6, 1e6),
    'Tc':           (1,  15),
    'Q_other':      (1e4,  1e8),
}


# ===========================================================================
# Model (stabilized form, lifted from the Origin embedded script)
# ===========================================================================
def model_total(params, f, n, T):
    """ params = [F0, D, b1, b2, AQP, Tc, Qo] ; f, n, T are arrays """
    F0, D, b1, b2, AQP, Tc, Qo = params
    f = np.asarray(f, dtype=float)
    n = np.asarray(n, dtype=float)
    T = np.asarray(T, dtype=float)

    hf_2kT = h * f / (2 * kB * T)
    tt = np.tanh(np.clip(hf_2kT, -50, 50))

    denom_tls = np.sqrt(
        1.0 + (n ** b2) / np.maximum(D * T ** b1, 1e-15) * tt
    )
    tls = F0 * tt / denom_tls

    # QP contribution
    qp_term = np.zeros_like(T)
    valid = (T > 0) & (f > 0)
    if np.any(valid):
        Tv = T[valid]; fv = f[valid]
        hf_2kT_v = h * fv / (2 * kB * Tv)
        Δ0 = 1.764 * kB * Tc
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            exp_term  = np.exp(np.minimum(Δ0 / (kB * Tv), 80))
            sinh_term = np.sinh(np.clip(hf_2kT_v, 1e-10, 30))
            k0_v      = kn(0, hf_2kT_v)
            k0_v      = np.where(np.isfinite(k0_v), k0_v, 1e-30)
            qp_v      = AQP * exp_term / (sinh_term * k0_v)
        qp_term[valid] = 1.0 / np.maximum(qp_v, 1e-15)

    return tls + qp_term + 1.0 / Qo


def model_breakdown(params, f, n, T):
    """ Return (tls, qp, offset, total) -- same numerical guards as model_total. """
    F0, D, b1, b2, AQP, Tc, Qo = params
    f = np.asarray(f, dtype=float)
    n = np.asarray(n, dtype=float)
    T = np.asarray(T, dtype=float)

    hf_2kT = h * f / (2 * kB * T)
    tt = np.tanh(np.clip(hf_2kT, -50, 50))
    denom_tls = np.sqrt(
        1.0 + (n ** b2) / np.maximum(D * T ** b1, 1e-15) * tt
    )
    tls = F0 * tt / denom_tls

    qp = np.zeros_like(T)
    valid = (T > 0) & (f > 0)
    if np.any(valid):
        Tv = T[valid]; fv = f[valid]
        hf_2kT_v = h * fv / (2 * kB * Tv)
        Δ0 = 1.764 * kB * Tc
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            exp_term  = np.exp(np.minimum(Δ0 / (kB * Tv), 80))
            sinh_term = np.sinh(np.clip(hf_2kT_v, 1e-10, 30))
            k0_v      = kn(0, hf_2kT_v)
            k0_v      = np.where(np.isfinite(k0_v), k0_v, 1e-30)
            qp_v      = AQP * exp_term / (sinh_term * k0_v)
        qp[valid] = 1.0 / np.maximum(qp_v, 1e-15)

    offset = np.full_like(T, 1.0 / Qo)
    return tls, qp, offset, tls + qp + offset


def weighted_mse(y, yfit, err):
    w = 1.0 / np.maximum(err ** 2, 1e-30)
    return np.sum(w * (y - yfit) ** 2) / np.sum(w)


# ===========================================================================
# Read data from the Origin source workbook
# ===========================================================================
def read_origin_source():
    """
    Returns a dict of equal-length numpy arrays with one entry per data point:
        {P, ph, freq, delta, delta_err, T_K, T_mK}
    Raises RuntimeError on any structural problem.
    """
    book = op.find_book('w', SRC_BOOK)
    if book is None:
        raise RuntimeError("Workbook '%s' not found in this Origin project."
                           % SRC_BOOK)
    ws_ph = op.find_sheet('w', '[%s]photon_numbers' % SRC_BOOK)
    ws_fr = op.find_sheet('w', '[%s]freq_values'    % SRC_BOOK)
    ws_d  = op.find_sheet('w', '[%s]delta'          % SRC_BOOK)
    ws_de = op.find_sheet('w', '[%s]delta_error'    % SRC_BOOK)
    if None in (ws_ph, ws_fr, ws_d, ws_de):
        raise RuntimeError(
            "Workbook '%s' must contain sheets: photon_numbers, "
            "freq_values, delta, delta_error." % SRC_BOOK)

    raw_P = ws_ph.to_list(0)

    P_, ph_, fr_, d_, de_, TK_, TmK_ = [], [], [], [], [], [], []

    for col in range(1, ws_ph.cols):
        lbl = ws_ph.get_label(col, type='C') or ''
        try:
            T_mK = float(lbl)
        except ValueError:
            continue
        T_K = T_mK * 1e-3
        raw_ph = ws_ph.to_list(col)
        raw_fr = ws_fr.to_list(col)
        raw_d  = ws_d.to_list(col)
        raw_de = ws_de.to_list(col)

        n = min(len(raw_P), len(raw_ph), len(raw_fr), len(raw_d), len(raw_de))
        for i in range(n):
            P, ph, fr, d, de = (raw_P[i], raw_ph[i], raw_fr[i],
                                raw_d[i], raw_de[i])
            if None in (P, ph, fr, d, de):
                continue
            try:
                P  = float(P);  ph = float(ph); fr = float(fr)
                d  = float(d);  de = float(de)
            except (TypeError, ValueError):
                continue
            if de <= 0 or not np.isfinite(d) or not np.isfinite(de):
                continue
            P_.append(P);  ph_.append(ph); fr_.append(fr)
            d_.append(d);  de_.append(de)
            TK_.append(T_K); TmK_.append(T_mK)

    if len(d_) == 0:
        raise RuntimeError("No usable data points found in '%s'." % SRC_BOOK)

    return dict(
        P         = np.array(P_,  dtype=float),
        ph        = np.array(ph_,  dtype=float),
        freq      = np.array(fr_,  dtype=float),
        delta     = np.array(d_,   dtype=float),
        delta_err = np.array(de_,  dtype=float),
        T_K       = np.array(TK_,  dtype=float),
        T_mK      = np.array(TmK_, dtype=float),
    )


# ===========================================================================
# Fitting strategies
# ===========================================================================
class StopFitException(Exception):
    """Raised to immediately halt the Scipy optimizer."""
    pass

def fit_diff_evolution(data_mask_dict, ranges, n_polish=200,
                       de_maxiter=300, de_popsize=15, stop_flag=None, progress_cb=None):
    bounds = [ranges[n] for n in PARAM_NAMES]

    eval_count = [0]
    def objective(p):
        if stop_flag is not None and stop_flag.is_set():
            raise StopFitException()
            
        # Micro-yield to prevent Tkinter freezing (GIL Starvation)
        eval_count[0] += 1
        if eval_count[0] % 100 == 0:
            time.sleep(0.001) 
            
        yfit = model_total(p, data_mask_dict['freq'],
                           data_mask_dict['ph'],
                           data_mask_dict['T_K'])
        return weighted_mse(data_mask_dict['delta'], yfit,
                            data_mask_dict['delta_err'])
                            
    iters = [0]
    def de_callback(xk, *args, **kwargs):
        if stop_flag is not None and stop_flag.is_set():
            raise StopFitException()
        iters[0] += 1
        if progress_cb:
            progress_cb(iters[0], de_maxiter)

    try:
        de = differential_evolution(
            objective, bounds=bounds, popsize=de_popsize,
            maxiter=de_maxiter, tol=1e-3, 
            workers=1, polish=False, disp=False, callback=de_callback)
        best_p, best_mse = de.x, de.fun
    except StopFitException:
        raise

    if stop_flag is None or not stop_flag.is_set():
        if progress_cb:
            progress_cb(-1, -1) # Signal indeterminate for local polish
        try:
            loc = minimize(objective, best_p, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': n_polish, 'disp': False})
            if loc.fun < best_mse:
                best_p, best_mse = loc.x, loc.fun
        except StopFitException:
            raise
        except Exception:
            pass
            
    return best_p, best_mse


def fit_random_restarts(data_mask_dict, ranges, n_attempts=1000,
                        seed_params=None, stop_flag=None, progress_cb=None):
    bounds = [ranges[n] for n in PARAM_NAMES]
    best_p = list(seed_params) if seed_params is not None else None
    best_mse = np.inf

    eval_count = [0]
    def objective(p):
        if stop_flag is not None and stop_flag.is_set():
            raise StopFitException()
            
        eval_count[0] += 1
        if eval_count[0] % 100 == 0:
            time.sleep(0.001)
            
        yfit = model_total(p, data_mask_dict['freq'],
                           data_mask_dict['ph'], data_mask_dict['T_K'])
        return weighted_mse(data_mask_dict['delta'], yfit, data_mask_dict['delta_err'])

    if best_p is not None:
        try:
            best_mse = objective(best_p)
        except StopFitException:
            raise

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
                best_p   = list(loc.x)
        except StopFitException:
            raise
        except Exception:
            continue
            
    return best_p, best_mse


def fit_local_only(data_mask_dict, ranges, seed_params, stop_flag=None, progress_cb=None):
    bounds = [ranges[n] for n in PARAM_NAMES]
    p0 = list(seed_params)
    
    eval_count = [0]
    def objective(p):
        if stop_flag is not None and stop_flag.is_set():
            raise StopFitException()
            
        eval_count[0] += 1
        if eval_count[0] % 100 == 0:
            time.sleep(0.001)
            
        yfit = model_total(p, data_mask_dict['freq'],
                           data_mask_dict['ph'], data_mask_dict['T_K'])
        return weighted_mse(data_mask_dict['delta'], yfit, data_mask_dict['delta_err'])

    iters = [0]
    def loc_callback(xk, *args, **kwargs):
        if stop_flag is not None and stop_flag.is_set():
            raise StopFitException()
        iters[0] += 1
        if progress_cb:
            progress_cb(iters[0], 500) # Assuming maxiter 500

    try:
        loc = minimize(objective, p0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500}, callback=loc_callback)
        return list(loc.x), float(loc.fun)
    except StopFitException:
        raise
    except Exception:
        try:
            return p0, float(objective(p0))
        except StopFitException:
            raise


# ===========================================================================
# Origin write helpers
# ===========================================================================
def get_or_make_sheet(book, name):
    ws = op.find_sheet('w', '[%s]%s' % (book.lt_range().lstrip('[').rstrip(']'), name))
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
        ws.from_df 
    except AttributeError:
        pass
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


def write_outputs_to_origin(book, params, mse, r2, sse, sst,
                            P_arr, T_arr, freq_arr, photon_arr,
                            delta_arr, delta_err_arr,
                            param_ranges, fit_strategy, n_attempts):
    # ---- 1) FitParams ----
    p_ws = get_or_make_sheet(book, OUT_PARAMS_SHEET)
    headers = PARAM_NAMES + ['MSE', 'R2', 'SSE', 'SST',
                             'N_points', 'Strategy', 'N_attempts']
    clear_sheet(p_ws, len(headers))
    for i, hd in enumerate(headers):
        p_ws.set_label(i, hd, type='L')
    values = list(params) + [mse, r2, sse, sst, len(delta_arr)]
    for i, v in enumerate(values):
        p_ws.from_list(i, [v])
    p_ws.from_list(len(values),     [str(fit_strategy)])
    p_ws.from_list(len(values) + 1, [int(n_attempts)])

    # ---- 2) Loss_Raw ----
    Ts = sorted(set(T_arr.tolist()))
    Ps = sorted(set(P_arr.tolist()))
    raw_cols = 1 + 2 * len(Ps)
    raw_ws = get_or_make_sheet(book, OUT_RAW_SHEET)
    clear_sheet(raw_ws, raw_cols)
    raw_ws.set_label(0, 'T', type='L')
    raw_ws.set_label(0, 'K', type='U')
    raw_ws.from_list(0, Ts)
    for j, p in enumerate(Ps):
        cb = 1 + 2 * j
        raw_ws.set_label(cb,     'δ',     type='L')
        raw_ws.set_label(cb,     'ppm',   type='U')
        raw_ws.set_label(cb,     '%g dBm' % p, type='C')
        raw_ws.set_label(cb + 1, 'δ_err', type='L')
        raw_ws.set_label(cb + 1, 'ppm',   type='U')
        raw_ws.set_label(cb + 1, '%g dBm' % p, type='C')
        d_col, e_col = [], []
        for T in Ts:
            mask = (T_arr == T) & (P_arr == p)
            if np.any(mask):
                idx = int(np.where(mask)[0][0])
                d_col.append(float(delta_arr[idx])     * 1e6)
                e_col.append(float(delta_err_arr[idx]) * 1e6)
            else:
                d_col.append(np.nan)
                e_col.append(np.nan)
        raw_ws.from_list(cb,     d_col)
        raw_ws.from_list(cb + 1, e_col)

    # ---- 3) Loss_Fitted ----
    T_min, T_max = float(min(Ts)), float(max(Ts))
    DENSE = 500
    if T_max <= 0.1:
        T_dense = np.linspace(max(T_min, 1e-3), T_max, DENSE)
    elif T_min >= 0.1:
        T_dense = np.linspace(T_min, T_max, DENSE)
    else:
        T_dense = np.concatenate([
            np.linspace(max(T_min, 1e-3), 0.1, DENSE // 2),
            np.linspace(0.1, T_max, DENSE - DENSE // 2)[1:],
        ])
    fit_cols = 1 + len(Ps)
    fit_ws = get_or_make_sheet(book, OUT_FIT_SHEET)
    clear_sheet(fit_ws, fit_cols)
    fit_ws.set_label(0, 'T', type='L')
    fit_ws.set_label(0, 'K', type='U')
    fit_ws.from_list(0, T_dense.tolist())

    for j, p in enumerate(Ps):
        col = 1 + j
        fit_ws.set_label(col, 'δ_fit', type='L')
        fit_ws.set_label(col, 'ppm',   type='U')
        fit_ws.set_label(col, '%g dBm' % p, type='C')
        sel = np.where(P_arr == p)[0]
        if sel.size == 0:
            fit_ws.from_list(col, [np.nan] * len(T_dense))
            continue
        T_s = T_arr[sel];   f_s = freq_arr[sel];   n_s = photon_arr[sel]
        order = np.argsort(T_s)
        T_s, f_s, n_s = T_s[order], f_s[order], n_s[order]
        if T_s.size >= 4:
            f_fun = interp1d(T_s, f_s, kind='cubic',
                             bounds_error=False, fill_value='extrapolate')
            n_fun = interp1d(T_s, n_s, kind='cubic',
                             bounds_error=False, fill_value='extrapolate')
            f_d = f_fun(T_dense);   n_d = np.maximum(n_fun(T_dense), 0)
        elif T_s.size >= 2:
            f_fun = interp1d(T_s, f_s, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
            n_fun = interp1d(T_s, n_s, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
            f_d = f_fun(T_dense);   n_d = np.maximum(n_fun(T_dense), 0)
        else:
            f_d = np.full_like(T_dense, f_s[0])
            n_d = np.full_like(T_dense, n_s[0])
        y = model_total(params, f_d, n_d, T_dense) * 1e6
        fit_ws.from_list(col, y.tolist())

    # ---- 4) Loss_Breakdown ----
    br_cols = 1 + 4 * len(Ps)
    br_ws = get_or_make_sheet(book, OUT_BREAK_SHEET)
    clear_sheet(br_ws, br_cols)
    br_ws.set_label(0, 'T', type='L')
    br_ws.set_label(0, 'K', type='U')
    br_ws.from_list(0, T_dense.tolist())
    for j, p in enumerate(Ps):
        sel = np.where(P_arr == p)[0]
        if sel.size == 0:
            for k, lab in enumerate(['TLS', 'QP', 'Offset', 'Total']):
                col = 1 + 4 * j + k
                br_ws.set_label(col, lab, type='L')
                br_ws.set_label(col, 'ppm', type='U')
                br_ws.set_label(col, '%g dBm' % p, type='C')
                br_ws.from_list(col, [np.nan] * len(T_dense))
            continue
        T_s = T_arr[sel];   f_s = freq_arr[sel];   n_s = photon_arr[sel]
        order = np.argsort(T_s)
        T_s, f_s, n_s = T_s[order], f_s[order], n_s[order]
        if T_s.size >= 4:
            kind = 'cubic'
        elif T_s.size >= 2:
            kind = 'linear'
        else:
            kind = None
        if kind is not None:
            f_d = interp1d(T_s, f_s, kind=kind, bounds_error=False,
                           fill_value='extrapolate')(T_dense)
            n_d = interp1d(T_s, n_s, kind=kind, bounds_error=False,
                           fill_value='extrapolate')(T_dense)
            n_d = np.maximum(n_d, 0)
        else:
            f_d = np.full_like(T_dense, f_s[0])
            n_d = np.full_like(T_dense, n_s[0])
        tls, qp, off, tot = model_breakdown(params, f_d, n_d, T_dense)
        for k, arr in enumerate([tls, qp, off, tot]):
            col = 1 + 4 * j + k
            lab = ['TLS', 'QP', 'Offset', 'Total'][k]
            br_ws.set_label(col, lab, type='L')
            br_ws.set_label(col, 'ppm', type='U')
            br_ws.set_label(col, '%g dBm' % p, type='C')
            br_ws.from_list(col, (arr * 1e6).tolist())

    # ---- 5) Graph ----
    gp = op.find_graph(OUT_GRAPH_NAME)
    if gp is None:
        try:
            gp = op.new_graph(template=OUT_GRAPH_TEMPLATE,
                              lname=OUT_GRAPH_NAME)
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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
    for j, p in enumerate(Ps):
        col_raw = 1 + 2 * j
        col_err = col_raw + 1
        try:
            sc = ly.add_plot(raw_ws, col_raw, 0, type='s', colyerr=col_err)
            sc.color     = colors[j % len(colors)]
            try:    sc.fillcolor = colors[j % len(colors)]
            except Exception: pass
            try:    sc.name = 'P=%g dBm' % p
            except Exception: pass
        except Exception:
            pass
        try:
            ln = ly.add_plot(fit_ws, 1 + j, 0, type='l')
            ln.color = colors[j % len(colors)]
            try:    ln.name = 'P=%g dBm Fit' % p
            except Exception: pass
        except Exception:
            pass
    try:    ly.title = 'Loss Model Fit'
    except Exception: pass
    try:    ly.rescale()
    except Exception: pass


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


class LossFitApp(tk.Toplevel):
    def __init__(self, master, raw):
        super().__init__(master)
        self.title("Loss Fit  —  source: [%s]" % SRC_BOOK)
        self.geometry("1280x880")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.raw = raw   
        self.param_ranges = dict(DEFAULT_RANGES)
        self.best_params  = None
        self.best_mse     = None
        self.fit_thread   = None
        self.stop_flag    = threading.Event()
        
        # UI Message Queue to prevent cross-thread freezing
        self.gui_queue    = queue.Queue()

        self.power_list       = []   
        self.temp_list        = []   
        self.power_checks     = {}   
        self.temp_checks      = {}   
        self.power_idx        = {}   
        self.temp_idx         = {}   
        self.scatter_artists  = {}   
        self.fit_line_artists = {}   
        self.resid_artists    = {}   

        self._build_widgets()
        self._apply_filter()         
        self._init_plots()
        
        # Start checking the queue immediately
        self._poll_queue()

    def _poll_queue(self):
        """Safely processes updates from the background thread onto the main UI thread."""
        try:
            while True:
                msg = self.gui_queue.get_nowait()
                cmd = msg[0]
                if cmd == 'progress':
                    self._update_progress(msg[1], msg[2])
                elif cmd == 'finish':
                    self._fit_finished(msg[1], msg[2], msg[3])
        except queue.Empty:
            pass
        # Only keep polling if the GUI is running
        if self.winfo_exists():
            self.after(50, self._poll_queue)

    def _build_widgets(self):
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=8, pady=8)

        self.tab_setup = ttk.Frame(nb);  nb.add(self.tab_setup, text='Setup')
        self.tab_fit   = ttk.Frame(nb);  nb.add(self.tab_fit,   text='Fit & Plot')
        self.tab_model = ttk.Frame(nb);  nb.add(self.tab_model, text='Model Equations')

        self._build_setup_tab(self.tab_setup)
        self._build_fit_tab(self.tab_fit)
        self._build_model_tab(self.tab_model)

        self.notebook = nb

    def _build_setup_tab(self, parent):
        flt = ttk.LabelFrame(parent, text='Pre-fit Filter', padding=10)
        flt.pack(fill='x', padx=10, pady=10)

        Pmin, Pmax = float(np.min(self.raw['P'])),  float(np.max(self.raw['P']))
        Tmin, Tmax = float(np.min(self.raw['T_mK'])), float(np.max(self.raw['T_mK']))

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

        rng = ttk.LabelFrame(parent, text='Parameter Ranges', padding=10)
        rng.pack(fill='x', padx=10, pady=10)
        ttk.Label(rng, text='Parameter', font=('Arial', 10, 'bold')
                  ).grid(row=0, column=0, padx=4, pady=4)
        ttk.Label(rng, text='Min', font=('Arial', 10, 'bold')
                  ).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(rng, text='Max', font=('Arial', 10, 'bold')
                  ).grid(row=0, column=2, padx=4, pady=4)
        ttk.Label(rng, text='Parameter', font=('Arial', 10, 'bold')
                  ).grid(row=0, column=4, padx=4, pady=4)
        ttk.Label(rng, text='Min', font=('Arial', 10, 'bold')
                  ).grid(row=0, column=5, padx=4, pady=4)
        ttk.Label(rng, text='Max', font=('Arial', 10, 'bold')
                  ).grid(row=0, column=6, padx=4, pady=4)

        self.range_vars = {}
        per_col = math.ceil(len(PARAM_NAMES) / 2)
        for i, nm in enumerate(PARAM_NAMES):
            col = (i // per_col) * 4
            row = i % per_col + 1
            lo, hi = self.param_ranges[nm]
            vmin = tk.DoubleVar(self, value=lo)
            vmax = tk.DoubleVar(self, value=hi)
            self.range_vars[nm] = (vmin, vmax)
            ttk.Label(rng, text=nm).grid(row=row, column=col,   sticky='e', padx=4, pady=2)
            ttk.Entry(rng, textvariable=vmin, width=14).grid(row=row, column=col+1, padx=4, pady=2)
            ttk.Entry(rng, textvariable=vmax, width=14).grid(row=row, column=col+2, padx=4, pady=2)

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
        self.var_attempts = tk.IntVar(self, value=1000)
        ttk.Entry(st, textvariable=self.var_attempts, width=10
                  ).grid(row=0, column=2, padx=4, sticky='w')

        sb = ttk.LabelFrame(parent, text='Filtered Dataset Summary', padding=10)
        sb.pack(fill='x', padx=10, pady=10)
        self.lbl_summary = ttk.Label(sb, text='(no filter applied yet)',
                                     font=('Consolas', 10), justify='left')
        self.lbl_summary.pack(anchor='w')

    def _build_fit_tab(self, parent):
        top = ttk.Frame(parent);  top.pack(fill='x', padx=10, pady=6)
        self.btn_fit  = ttk.Button(top, text='Run Fit', command=self._on_run_fit)
        self.btn_fit.pack(side='left', padx=4)
        self.btn_stop = ttk.Button(top, text='Stop', command=self._on_stop_fit,
                                   state='disabled')
        self.btn_stop.pack(side='left', padx=4)
        ttk.Button(top, text='Reset Sliders to Range Mid',
                   command=self._reset_sliders_mid).pack(side='left', padx=4)
        ttk.Button(top, text='Export to Origin',
                   command=self._on_export).pack(side='left', padx=12)
                   
        # ---- PROGRESS BAR UI ----
        self.progress_var = tk.DoubleVar(self)
        self.progress = ttk.Progressbar(top, variable=self.progress_var, maximum=100, length=200)
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
        controls_window = ctrl_canvas.create_window(
            (0, 0), window=controls, anchor='nw')
        
        def _ctrl_configure(_event):
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

        self.slider_vars   = {}
        self.slider_labels = {}
        self.slider_widgets = {}
        per_col = math.ceil(len(PARAM_NAMES) / 2)
        for i, nm in enumerate(PARAM_NAMES):
            col_block = (i // per_col) * 4
            row       = i % per_col
            lo, hi = self.param_ranges[nm]
            ttk.Label(sl, text=nm + ':', font=('Arial', 9, 'bold')
                      ).grid(row=row, column=col_block,
                             padx=(8, 4), pady=4, sticky='e')
            dv = tk.DoubleVar(self)
            self.slider_vars[nm] = dv
            if nm in LOG_PARAMS and lo > 0 and hi > 0:
                lo_, hi_ = np.log10(lo), np.log10(hi)
                dv.set((lo_ + hi_) / 2)
            else:
                dv.set((lo + hi) / 2)
            sld = ttk.Scale(sl, variable=dv,
                            from_=(np.log10(lo) if nm in LOG_PARAMS else lo),
                            to=(np.log10(hi)    if nm in LOG_PARAMS else hi),
                            orient='horizontal', length=240,
                            command=lambda v, name=nm: self._on_slider(name, v))
            sld.grid(row=row, column=col_block+1, padx=4, pady=4)
            self.slider_widgets[nm] = sld
            vlbl = ttk.Label(sl, text='', font=('Consolas', 9), width=14)
            vlbl.grid(row=row, column=col_block+2, padx=(4, 14), pady=4,
                      sticky='w')
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
        self.fig = None;   self.canvas = None
        self.ax_main = None;  self.ax_resid = None

        def _set_initial_sash(_event=None):
            try:
                total_h = paned.winfo_height()
                if total_h > 200:
                    paned.sashpos(0, min(320, total_h // 3))
            except Exception:
                pass
        paned.bind('<Map>', _set_initial_sash)
        self.after(200, _set_initial_sash)

    def _build_model_tab(self, parent):
        # Create a clean figure with no axes
        fig_eq = plt.Figure(figsize=(8, 6), facecolor='white')
        ax_eq = fig_eq.add_subplot(111)
        ax_eq.axis('off')

        # Define the LaTeX strings for the mathtext engine
        eq1 = r"$\frac{1}{Q_{int}} = \frac{1}{Q_{TLS}(\bar{n}, T)} + \frac{1}{Q_{QP}(T)} + \frac{1}{Q_{other}}$"
        eq2 = r"$Q_{TLS}(\bar{n}, T) = Q_{TLS,0} \frac{\sqrt{1 + \left( \frac{\bar{n}^{\beta_2}}{D T^{\beta_1}} \right) \tanh\left( \frac{\hbar \omega}{2 k_B T} \right)}}{\tanh\left( \frac{\hbar \omega}{2 k_B T} \right)}$"
        eq3 = r"$Q_{QP}(T) = A_{QP} \frac{e^{\Delta_0 / k_B T}}{\sinh\left( \frac{\hbar \omega}{2 k_B T} \right) K_0\left( \frac{\hbar \omega}{2 k_B T} \right)}$"

        # Render Equation 1
        ax_eq.text(0.5, 0.85, "Total Internal Loss:", fontsize=14, ha='center', va='center', fontweight='bold')
        ax_eq.text(0.5, 0.70, eq1, fontsize=24, ha='center', va='center')

        # Render Equation 2
        ax_eq.text(0.5, 0.55, "Two-Level System (TLS) Loss:", fontsize=14, ha='center', va='center', fontweight='bold')
        ax_eq.text(0.5, 0.40, eq2, fontsize=24, ha='center', va='center')

        # Render Equation 3
        ax_eq.text(0.5, 0.25, "Quasiparticle (QP) Loss:", fontsize=14, ha='center', va='center', fontweight='bold')
        ax_eq.text(0.5, 0.10, eq3, fontsize=24, ha='center', va='center')

        # Adjust subplot margins so the text isn't cut off
        fig_eq.subplots_adjust(top=0.9, bottom=0.1)

        # Embed it into the Tkinter tab
        canvas_eq = FigureCanvasTkAgg(fig_eq, master=parent)
        canvas_eq.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
        canvas_eq.draw()

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
        Pmin = self.var_pmin.get();  Pmax = self.var_pmax.get()
        Tmin = self.var_tmin.get();  Tmax = self.var_tmax.get()
        if Pmin > Pmax: Pmin, Pmax = Pmax, Pmin
        if Tmin > Tmax: Tmin, Tmax = Tmax, Tmin

        m = ((self.raw['P']    >= Pmin) & (self.raw['P']    <= Pmax) &
             (self.raw['T_mK'] >= Tmin) & (self.raw['T_mK'] <= Tmax))

        f = {k: v[m] for k, v in self.raw.items()}
        self.filt = f

        self.power_list = sorted(set(f['P'].tolist()))
        self.temp_list  = sorted(set(f['T_mK'].tolist()))

        self.power_idx = {p: np.where(f['P']    == p)[0] for p in self.power_list}
        self.temp_idx  = {t: np.where(f['T_mK'] == t)[0] for t in self.temp_list}

        n_pts = len(f['delta'])
        self.lbl_summary.config(text=(
            'Points: %d   |   distinct powers: %d   |   '
            'distinct temperatures: %d\n'
            'Power range kept: [%g, %g] dBm     Temperature range kept: '
            '[%g, %g] mK'
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

        pw_btns = ttk.Frame(pw_box);  pw_btns.grid(row=0, column=0, columnspan=8,
                                                   sticky='w', pady=(0, 4))
        ttk.Button(pw_btns, text='All',  width=5,
                   command=lambda: self._set_all_powers(True)
                   ).pack(side='left', padx=2)
        ttk.Button(pw_btns, text='None', width=5,
                   command=lambda: self._set_all_powers(False)
                   ).pack(side='left', padx=2)
        tp_btns = ttk.Frame(tp_box);  tp_btns.grid(row=0, column=0, columnspan=8,
                                                   sticky='w', pady=(0, 4))
        ttk.Button(tp_btns, text='All',  width=5,
                   command=lambda: self._set_all_temps(True)
                   ).pack(side='left', padx=2)
        ttk.Button(tp_btns, text='None', width=5,
                   command=lambda: self._set_all_temps(False)
                   ).pack(side='left', padx=2)

        cols_pw = 4
        for i, p in enumerate(self.power_list):
            v = tk.BooleanVar(self, value=True)
            self.power_checks[p] = v
            ttk.Checkbutton(pw_box, text='%g' % p, variable=v,
                            command=self._on_dataset_toggle
                            ).grid(row=1 + i // cols_pw, column=i % cols_pw,
                                   sticky='w', padx=6, pady=1)

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
                if self.power_checks.get(p, tk.BooleanVar(self, value=True)).get()]

    def _checked_temps(self):
        return [t for t in self.temp_list
                if self.temp_checks.get(t, tk.BooleanVar(self, value=True)).get()]

    def _checked_indices(self):
        powers_on = self._checked_powers()
        temps_on  = self._checked_temps()
        if not powers_on or not temps_on:
            return np.array([], dtype=int)
        f = self.filt
        mask = np.isin(f['P'], powers_on) & np.isin(f['T_mK'], temps_on)
        return np.where(mask)[0]

    def _on_dataset_toggle(self):
        if self.ax_main is None:
            return
        powers_on = set(self._checked_powers())
        temps_on  = set(self._checked_temps())
        for p, container in self.scatter_artists.items():
            vis_series = (p in powers_on)
            container[0].set_visible(vis_series)
            for ln in container[1]: ln.set_visible(vis_series)
            for ln in container[2]: ln.set_visible(vis_series)
            if p in self.fit_line_artists:
                self.fit_line_artists[p].set_visible(vis_series)
            if p in self.resid_artists:
                self.resid_artists[p].set_visible(vis_series)
        self._refresh_plot()

    def _slider_value_to_param(self, nm):
        v = self.slider_vars[nm].get()
        if nm in LOG_PARAMS:
            return 10 ** float(v)
        return float(v)

    def _set_param_to_slider(self, nm, value):
        if nm in LOG_PARAMS and value > 0:
            self.slider_vars[nm].set(np.log10(value))
        else:
            self.slider_vars[nm].set(value)

    def _refresh_slider_label(self, nm):
        v = self._slider_value_to_param(nm)
        if nm in LOG_PARAMS:
            txt = eng(v)
        else:
            txt = '%.4f' % v
        self.slider_labels[nm].config(text=txt)

    def _on_slider(self, nm, _val):
        self._refresh_slider_label(nm)
        self._refresh_plot()

    def _reset_sliders_mid(self):
        for nm in PARAM_NAMES:
            lo, hi = self.range_vars[nm][0].get(), self.range_vars[nm][1].get()
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

    def _init_plots(self):
        for w in self.plot_container.winfo_children():
            w.destroy()

        self.fig, (self.ax_main, self.ax_resid) = plt.subplots(
            2, 1, figsize=(9, 5),
            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        self.fig.patch.set_facecolor('white')
        self.ax_main.set_yscale('log')
        self.ax_main.set_ylabel('δ  (log)')
        self.ax_resid.set_ylabel('weighted residual')
        self.ax_resid.set_xlabel('Temperature (K)')
        self.ax_main.grid(True, which='both', alpha=0.3)
        self.ax_resid.grid(True, alpha=0.3)
        self.ax_resid.axhline(0, color='k', lw=0.7)

        self.scatter_artists.clear()
        self.fit_line_artists.clear()
        self.resid_artists.clear()

        cmap = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for p in self.power_list:
            color = next(cmap)
            idx   = self.power_idx[p]
            T_K   = self.filt['T_K'][idx]
            δ     = self.filt['delta'][idx]
            σ     = self.filt['delta_err'][idx]
            order = np.argsort(T_K)
            T_K, δ, σ = T_K[order], δ[order], σ[order]
            cont = self.ax_main.errorbar(
                T_K, δ, yerr=σ, fmt='o', ms=4, alpha=0.85, color=color,
                capsize=2, label='%g dBm' % p)
            self.scatter_artists[p] = cont
            ln, = self.ax_main.plot([], [], lw=1.8, color=color)
            self.fit_line_artists[p] = ln
            rln, = self.ax_resid.plot([], [], 'o-', ms=3, lw=0.8, color=color)
            self.resid_artists[p] = rln

        n_pow = len(self.power_list)
        if 0 < n_pow <= 24:
            ncol = 1 if n_pow <= 8 else 2
            self.ax_main.legend(fontsize=8, loc='upper left',
                                bbox_to_anchor=(1.01, 1.0), ncol=ncol,
                                handletextpad=0.4, columnspacing=0.8,
                                borderaxespad=0.2)

        right_margin = 0.78 if (n_pow > 8) else 0.83
        self.fig.subplots_adjust(
            left=0.10, right=right_margin,
            top=0.92, bottom=0.12, hspace=0.10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_container)
        toolbar.update()

        self._refresh_plot()

    def _refresh_plot(self):
        if self.ax_main is None:
            return
        params    = self._current_params()
        powers_on = set(self._checked_powers())
        temps_on  = set(self._checked_temps())

        used_T = []
        all_resid = []

        for p in self.power_list:
            if p not in powers_on:
                self.fit_line_artists[p].set_data([], [])
                self.resid_artists[p].set_data([], [])
                continue

            idx_all = self.power_idx[p]
            T_mK_all = self.filt['T_mK'][idx_all]
            keep = np.array([t in temps_on for t in T_mK_all], dtype=bool)
            idx = idx_all[keep]
            if idx.size == 0:
                self.fit_line_artists[p].set_data([], [])
                self.resid_artists[p].set_data([], [])
                container = self.scatter_artists[p]
                container[0].set_data([], [])
                continue

            T   = self.filt['T_K'][idx]
            f0  = self.filt['freq'][idx]
            n0  = self.filt['ph'][idx]
            δ   = self.filt['delta'][idx]
            σ   = self.filt['delta_err'][idx]
            order = np.argsort(T)
            T_s, f_s, n_s, δ_s, σ_s = (T[order], f0[order], n0[order],
                                       δ[order], σ[order])

            container = self.scatter_artists[p]
            container[0].set_data(T_s, δ_s)

            yfit = model_total(params, f_s, n_s, T_s)
            self.fit_line_artists[p].set_data(T_s, yfit)
            resid = (δ_s - yfit) / np.maximum(σ_s, 1e-30)
            self.resid_artists[p].set_data(T_s, resid)
            used_T.append(T_s)
            all_resid.append(resid)

        idx_on = self._checked_indices()
        if idx_on.size:
            mse = self._mse_for_indices(idx_on, params)
        else:
            mse = float('nan')

        title = 'Manual fit overlay   |   weighted MSE = %.3e' % mse
        if self.best_mse is not None:
            title += '   |   best fit MSE = %.3e' % self.best_mse
        self.ax_main.set_title(title)

        if used_T:
            allT = np.concatenate(used_T)
            self.ax_main.set_xlim(allT.min() * 0.9, allT.max() * 1.1)
        if all_resid:
            r = np.concatenate(all_resid)
            r = r[np.isfinite(r)]
            if r.size:
                m = max(3, np.percentile(np.abs(r), 99))
                self.ax_resid.set_ylim(-m * 1.1, m * 1.1)

        self.ax_main.relim();  self.ax_main.autoscale_view(scalex=False)
        self.canvas.draw_idle()

    def _mse_for_indices(self, idx, params):
        f = self.filt
        yfit = model_total(params, f['freq'][idx], f['ph'][idx], f['T_K'][idx])
        return float(weighted_mse(f['delta'][idx], yfit, f['delta_err'][idx]))

    def _checked_data_dict(self):
        idx = self._checked_indices()
        if idx.size == 0:
            return None, None
        f = self.filt
        return {
            'freq':      f['freq'][idx],
            'ph':        f['ph'][idx],
            'T_K':       f['T_K'][idx],
            'delta':     f['delta'][idx],
            'delta_err': f['delta_err'][idx],
        }, idx

    def _update_progress(self, current, total):
        """Called safely from main GUI thread via queue"""
        if total <= 0:
            self.progress.configure(mode='indeterminate')
            self.progress.start(10)
        else:
            self.progress.configure(mode='determinate')
            self.progress.stop()
            val = max(0, min(100, (current / total) * 100))
            self.progress_var.set(val)

    def _on_run_fit(self):
        data, _ = self._checked_data_dict()
        if data is None or len(data['delta']) < 3:
            messagebox.showwarning(
                "No data", "Select at least a few datasets first.", parent=self)
            return

        ranges = {}
        for nm, (vmin, vmax) in self.range_vars.items():
            lo, hi = float(vmin.get()), float(vmax.get())
            if lo >= hi:
                messagebox.showerror(
                    "Bad range", "Parameter %s: min must be < max." % nm,
                    parent=self)
                return
            ranges[nm] = (lo, hi)
            self.param_ranges[nm] = (lo, hi)
            if nm in LOG_PARAMS and lo > 0 and hi > 0:
                self.slider_widgets[nm].configure(
                    from_=np.log10(lo), to=np.log10(hi))
            else:
                self.slider_widgets[nm].configure(from_=lo, to=hi)

        strategy   = self.var_strategy.get()
        n_attempts = int(self.var_attempts.get())

        # IMPORTANT: read every Tk variable HERE, on the main GUI thread.
        # Tk DoubleVar/IntVar/StringVar are NOT thread-safe; reading them
        # from a worker thread raises "main thread is not in main loop".
        # This is why local/random failed and de_local worked: only DE
        # doesn't need a seed from the sliders.
        seed_params = self._current_params()

        # Clamp the seed into the (possibly just-edited) bounds so
        # L-BFGS-B doesn't reject it with "x0 violates bound constraints".
        clamped_seed = []
        for nm, v in zip(PARAM_NAMES, seed_params):
            lo, hi = ranges[nm]
            clamped_seed.append(min(max(float(v), lo), hi))

        self.stop_flag.clear()
        self.btn_fit.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.lbl_status.config(text='fitting...')
        self.progress_var.set(0)
        self.progress.configure(mode='determinate')

        def worker():
            try:
                def thread_safe_progress(curr, tot):
                    self.gui_queue.put(('progress', curr, tot))

                if strategy == 'de_local':
                    p, mse = fit_diff_evolution(
                        data, ranges, n_polish=max(50, n_attempts // 5),
                        de_maxiter=300, de_popsize=15,
                        stop_flag=self.stop_flag,
                        progress_cb=thread_safe_progress)
                elif strategy == 'random':
                    p, mse = fit_random_restarts(
                        data, ranges, n_attempts=n_attempts,
                        seed_params=clamped_seed,
                        stop_flag=self.stop_flag,
                        progress_cb=thread_safe_progress)
                else:
                    p, mse = fit_local_only(
                        data, ranges, clamped_seed,
                        stop_flag=self.stop_flag,
                        progress_cb=thread_safe_progress)

                self.gui_queue.put(('finish', p, mse, None))

            except StopFitException:
                self.gui_queue.put(('finish', None, None, None))
            except Exception:
                tb = traceback.format_exc()
                tagged = "Strategy: %s\n\n%s" % (strategy, tb)
                self.gui_queue.put(('finish', None, None, tagged))

        self.fit_thread = threading.Thread(target=worker, daemon=True)
        self.fit_thread.start()

    def _on_stop_fit(self):
        self.stop_flag.set()
        self.lbl_status.config(text='stopping...')

    def _fit_finished(self, p, mse, err):
        self.btn_fit.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.progress.stop()
        self.progress.configure(mode='determinate')
        self.progress_var.set(100 if p is not None else 0)
        
        if err is not None:
            self.lbl_status.config(text='fit error')
            messagebox.showerror("Fit error", err, parent=self)
            return
        if p is None:
            self.lbl_status.config(text='fit cancelled')
            return
        self.best_params = list(p)
        self.best_mse    = float(mse)
        for nm, val in zip(PARAM_NAMES, p):
            self._set_param_to_slider(nm, float(val))
            self._refresh_slider_label(nm)
        self.lbl_status.config(text='fit done   MSE=%.3e' % self.best_mse)
        self._refresh_plot()
        self.notebook.select(self.tab_fit)

    def _on_export(self):
        params = self._current_params()
        data, idx = self._checked_data_dict()
        if data is None:
            messagebox.showwarning("No data", "Select datasets first.",
                                   parent=self)
            return
        yfit = model_total(params, data['freq'], data['ph'], data['T_K'])
        residuals = data['delta'] - yfit
        sse = float(np.sum(residuals ** 2))
        sst = float(np.sum((data['delta'] - np.mean(data['delta'])) ** 2))
        r2  = 1.0 - sse / sst if sst > 0 else float('nan')
        mse = float(weighted_mse(data['delta'], yfit, data['delta_err']))

        f = self.filt
        P_arr     = f['P'][idx]
        T_arr_K   = f['T_K'][idx]
        freq_arr  = f['freq'][idx]
        photon    = f['ph'][idx]
        delta     = f['delta'][idx]
        delta_err = f['delta_err'][idx]

        try:
            book = op.find_book('w', SRC_BOOK)
            if book is None:
                raise RuntimeError("Workbook '%s' not found." % SRC_BOOK)
            self.lbl_status.config(text='exporting to Origin...')
            self.update_idletasks()
            write_outputs_to_origin(
                book, params, mse, r2, sse, sst,
                P_arr, T_arr_K, freq_arr, photon,
                delta, delta_err,
                self.param_ranges, self.var_strategy.get(),
                int(self.var_attempts.get()))
            self.lbl_status.config(
                text='exported   MSE=%.3e  R²=%.4f' % (mse, r2))
            messagebox.showinfo(
                "Exported",
                "Results written to workbook '%s'.\n"
                "Sheets: %s, %s, %s, %s\nGraph: %s" % (
                    SRC_BOOK, OUT_PARAMS_SHEET, OUT_RAW_SHEET,
                    OUT_FIT_SHEET, OUT_BREAK_SHEET, OUT_GRAPH_NAME),
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
        root = tk.Tk();  root.withdraw()
        messagebox.showerror("Cannot read source", str(e))
        root.destroy()
        return

    root = tk.Tk()
    root.withdraw()
    app = LossFitApp(root, raw)
    root.wait_window(app)
    try:
        root.destroy()
    except Exception:
        pass


if __name__ == '__main__':
    main()