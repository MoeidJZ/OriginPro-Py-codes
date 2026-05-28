import numpy as np
import originpro as op
from scipy.optimize import curve_fit
import re

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTANT: Sheet Naming Convention
# ══════════════════════════════════════════════════════════════════════════════
# This script expects sheets to be named in the format: "Res{number}_{temperature} mK"
# Example: "Res1_50 mK", "Res2_75 mK"
# If your sheet names differ, please rename them to match this format.
# ══════════════════════════════════════════════════════════════════════════════

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
SRC_BOOK      = 'TaN7p0R2'
RESONATOR_NUM = 'Res5'   # e.g., 'Res1', 'Res2', etc.
NUM_ATTEMPTS  = 50
PARAM_RANGES  = {
    'FdeltaTLS0': (0.5e-6, 20e-6),  # = 1/Q_TLS at low power & T
    'nc':         (1, 5000),          # critical photon number
    'beta':       (0.4, 1),          # saturation exponent
    'Q_other':    (0.1e6, 100e6),          # photon-independent quality factor
}
DENSE_POINTS  = 1000
EXPORT_POINTS = 200
# ──────────────────────────────────────────────────────────────────────────────

# Physical constants
HBAR = 1.0546e-34   # J·s
KB   = 1.381e-23    # J/K

# Column mapping in source sheets
COLUMN_MAP = {
    'photon_numbers':   1,   # B → index 1
    'freq_values':      5,   # F → index 5
    'freq_err_values': 11,   # L → index 11
    'delta':           18,   # S → index 18
    'delta_error':     19,   # T → index 19
}

# ── 1) Find source book and locate resonator sheet ────────────────────────────
src_book = op.find_book('w', SRC_BOOK)
if src_book is None:
    raise RuntimeError(f"Source book '{SRC_BOOK}' not found")

pattern   = re.compile(rf'^{re.escape(RESONATOR_NUM)}_.*mK', re.IGNORECASE)
res_sheet     = None
temperature_mK = None

for wks in src_book:
    if pattern.match(wks.name):
        res_sheet = wks
        temp_match = re.search(r'(\d+\.?\d*)\s*mK', wks.name, re.IGNORECASE)
        if temp_match:
            temperature_mK = float(temp_match.group(1))
        break

if res_sheet is None:
    raise RuntimeError(f"No sheet found matching pattern '{RESONATOR_NUM}_* mK'")
if temperature_mK is None:
    raise RuntimeError(f"Could not extract temperature from sheet name '{res_sheet.name}'")

print(f"Processing sheet : {res_sheet.name}")
print(f"Temperature      : {temperature_mK} mK")

# ── 2) Read data ───────────────────────────────────────────────────────────────
def read_column(wks, col_idx):
    return wks.to_list(col_idx)

raw_ph = read_column(res_sheet, COLUMN_MAP['photon_numbers'])
raw_fr = read_column(res_sheet, COLUMN_MAP['freq_values'])
raw_d  = read_column(res_sheet, COLUMN_MAP['delta'])
raw_de = read_column(res_sheet, COLUMN_MAP['delta_error'])

valid = [
    i for i, (p, f, d, er)
    in enumerate(zip(raw_ph, raw_fr, raw_d, raw_de))
    if p  not in ('', None)
    and f not in ('', None)
    and d not in ('', None)
    and er not in ('', None)
]

if not valid:
    raise RuntimeError(f"No valid data in sheet '{res_sheet.name}'")

photon    = np.array([float(raw_ph[i]) for i in valid])
freq      = np.array([float(raw_fr[i]) for i in valid])
delta     = np.array([float(raw_d[i])  for i in valid])
delta_err = np.array([float(raw_de[i]) for i in valid])

# ── 3) Sort by photon number ──────────────────────────────────────────────────
order = np.argsort(photon)
photon, freq, delta, delta_err = [arr[order] for arr in (photon, freq, delta, delta_err)]

# ── 4) Model ──────────────────────────────────────────────────────────────────
T_K = temperature_mK / 1e3

def model_power(x, FdeltaTLS0, nc, beta, Q_other):
    """
    Power-dependent TLS loss model:

        delta = 1/Q_other
                + FdeltaTLS0 * tanh(hbar*omega / (2*kB*T))
                  / sqrt(1 + (n/nc)^beta)

    Parameters
    ----------
    FdeltaTLS0 : filling-factor × TLS loss tangent  (= 1/Q_TLS at low power/T)
    nc         : critical photon number for TLS saturation
    beta       : saturation exponent (typically 0.3–0.7)
    Q_other    : photon-number-independent loss channel
    """
    f, n = x
    # freq column stores f (Hz); hbar*omega = hbar * 2*pi*f = h*f
    tt = np.tanh(HBAR * 2.0 * np.pi * f / (2.0 * KB * T_K))
    return FdeltaTLS0 * tt /(np.sqrt(1.0 + (n / nc) ** beta)) + 1.0 / Q_other

# ── 5) Fit ────────────────────────────────────────────────────────────────────
param_keys = ('FdeltaTLS0', 'nc', 'beta', 'Q_other')
lower = np.array([PARAM_RANGES[k][0] for k in param_keys])
upper = np.array([PARAM_RANGES[k][1] for k in param_keys])

def run_fit():
    best_mse, best_p, best_cov = np.inf, None, None
    xdata = (freq, photon)

    for attempt in range(NUM_ATTEMPTS):
        # Random starting point drawn uniformly within bounds
        p0 = [np.random.uniform(lo, hi) for lo, hi in zip(lower, upper)]

        try:
            popt, pcov = curve_fit(
                model_power,
                xdata, delta,
                sigma=delta_err,        # error bars drive inverse-variance weighting
                absolute_sigma=True,
                p0=p0,
                maxfev=20000,
                bounds=(lower, upper),  # PARAM_RANGES enforced as hard bounds
            )
        except Exception:
            continue

        yfit = model_power(xdata, *popt)
        # Reduced chi-squared — no extra ad-hoc power weighting
        mse  = np.mean(((delta - yfit) / np.maximum(delta_err, 1e-10)) ** 2)

        if mse < best_mse:
            best_mse, best_p, best_cov = mse, popt, pcov
            print(f"  Attempt {attempt + 1:>3d}: reduced χ² = {mse:.4e}")

    return best_p, best_cov, best_mse

print(f"\nRunning fit with {NUM_ATTEMPTS} random initialisations...")
popt, pcov, mse = run_fit()

if popt is None:
    raise RuntimeError("Fit failed on all attempts — try widening PARAM_RANGES")

FdeltaTLS0, nc, beta, Q_other = popt
FdeltaTLS0_err, nc_err, beta_err, Q_other_err = np.sqrt(np.diag(pcov))

# Q_TLS = 1/FdeltaTLS0;  propagated error: dQ_TLS = dFdeltaTLS0 / FdeltaTLS0²
Q_TLS     = 1.0 / FdeltaTLS0
Q_TLS_err = FdeltaTLS0_err / FdeltaTLS0 ** 2

print(f"\nFit Results:")
print(f"  FdeltaTLS0 = {FdeltaTLS0:.4e} ± {FdeltaTLS0_err:.4e}")
print(f"  nc         = {nc:.4e}   ± {nc_err:.4e}")
print(f"  beta       = {beta:.4f}   ± {beta_err:.4f}")
print(f"  Q_other    = {Q_other:.4e}   ± {Q_other_err:.4e}")
print(f"  Q_TLS      = {Q_TLS:.4e}   ± {Q_TLS_err:.4e}")
print(f"  Reduced χ² = {mse:.4e}")

# ── 6) Save fit parameters to 'Fit Parameters' sheet ─────────────────────────
params_name = 'Fit Parameters'
params_wks  = op.find_sheet('w', f'[{SRC_BOOK}]{params_name}')

if params_wks is None:
    params_wks = src_book.add_sheet(params_name)
    params_wks.cols = 8
    params_wks.from_list(0, [], lname='Resonator')
    params_wks.from_list(1, [], lname='Fδ\-(TLS0)')
    params_wks.from_list(2, [], lname='n\-(c)')
    params_wks.from_list(3, [], lname='β')
    params_wks.from_list(4, [], lname='Q\-(other)')
    params_wks.from_list(5, [], lname='Q\-(TLS)')
    params_wks.from_list(6, [], lname='Q\-(TLS) err')
    params_wks.from_list(7, [], lname='Reduced χ²')

existing_res = [str(x) for x in params_wks.to_list(0) if x not in (None, '')]

new_row = [RESONATOR_NUM, FdeltaTLS0, nc, beta, Q_other, Q_TLS, Q_TLS_err, mse]

if RESONATOR_NUM in existing_res:
    row_idx  = existing_res.index(RESONATOR_NUM)
    print(f"\nUpdating existing row for {RESONATOR_NUM} (row {row_idx})")
    col_data = [params_wks.to_list(i) for i in range(8)]
    for col_i, val in enumerate(new_row):
        col_data[col_i][row_idx] = val
    for col_i, data in enumerate(col_data):
        params_wks.from_list(col_i, data)
else:
    print(f"\nAppending new row for {RESONATOR_NUM}")
    def _ext(col):
        return [float(x) for x in params_wks.to_list(col) if x not in (None, '')]
    params_wks.from_list(0, existing_res + [RESONATOR_NUM],  lname='Resonator')
    params_wks.from_list(1, _ext(1) + [FdeltaTLS0],          lname='Fδ\-(TLS0)')
    params_wks.from_list(2, _ext(2) + [nc],                  lname='n\-(c)')
    params_wks.from_list(3, _ext(3) + [beta],                lname='β')
    params_wks.from_list(4, _ext(4) + [Q_other],             lname='Q\-(other)')
    params_wks.from_list(5, _ext(5) + [Q_TLS],               lname='Q\-(TLS)')
    params_wks.from_list(6, _ext(6) + [Q_TLS_err],           lname='Q\-(TLS) err')
    params_wks.from_list(7, _ext(7) + [mse],                 lname='Reduced χ²')

# ── 7) Generate fitted curve for export ───────────────────────────────────────
f0        = float(np.median(freq))
n_dense   = np.logspace(np.log10(photon.min()), np.log10(photon.max()), DENSE_POINTS)
tt        = np.tanh(HBAR * 2.0 * np.pi * f0 / (2.0 * KB * T_K))
fit_dense = FdeltaTLS0 * tt / np.sqrt(1.0 + (n_dense / nc) ** beta) + 1.0 / Q_other

idx        = np.round(np.linspace(0, DENSE_POINTS - 1, EXPORT_POINTS)).astype(int)
n_export   = n_dense[idx]
fit_export = fit_dense[idx]

# Column comment for fitted delta — carries Q_TLS and shows up as legend in OriginPro
qtls_comment = f"{RESONATOR_NUM} Q\-(TLS)={Q_TLS:.2e}±{Q_TLS_err:.2e}"

# ── 8) Export raw data + fitted curve to 'Power Loss Fit' sheet ───────────────
out_name = 'Power Loss Fit'
out_wks  = op.find_sheet('w', f'[{SRC_BOOK}]{out_name}')

if out_wks is None:
    out_wks  = src_book.add_sheet(out_name)
    out_wks.cols = 5
    base_col = 0
    print(f"\nCreated new sheet '{out_name}'")
else:
    # Strip any Q_TLS suffix when matching to find existing resonator columns
    def _base_comment(c):
        return (out_wks.get_label(c, type='C') or '').split(' Q_TLS=')[0]

    existing_resonators = []
    for c in range(out_wks.cols):
        bc = _base_comment(c)
        if bc and bc not in existing_resonators:
            existing_resonators.append(bc)

    if RESONATOR_NUM in existing_resonators:
        print(f"\nReplacing existing columns for {RESONATOR_NUM}")
        base_col = None
        for c in range(out_wks.cols):
            if _base_comment(c) == RESONATOR_NUM:
                base_col = c
                break
    else:
        base_col = out_wks.cols + 1   # one blank column as visual separator
        out_wks.cols = base_col + 5
        print(f"\nAdding new columns for {RESONATOR_NUM} starting at column {base_col}")

if base_col is not None:
    out_wks.from_list(base_col + 0, photon.tolist(),
                      lname='<n>',      comments=RESONATOR_NUM)
    out_wks.from_list(base_col + 1, (delta     * 1e6).tolist(),
                      lname='δ',        units='ppm', comments=RESONATOR_NUM)
    out_wks.from_list(base_col + 2, (delta_err * 1e6).tolist(),
                      lname='δ err',    units='ppm', comments=RESONATOR_NUM)
    out_wks.from_list(base_col + 3, n_export.tolist(),
                      lname='<n>(fit)', comments=RESONATOR_NUM)
    # Fitted delta: Q_TLS in comment → appears as legend entry in OriginPro
    out_wks.from_list(base_col + 4, (fit_export * 1e6).tolist(),
                      lname='Fit δ',    units='ppm', comments=qtls_comment)
    print(f"Exported to columns {base_col} – {base_col + 4}")

# ── 9) Create graph ───────────────────────────────────────────────────────────
try:
    gp = op.new_graph(template='PowerDepLossFit')
    print("\nCreated graph using 'PowerDepLossFit' template")
except Exception:
    gp = op.new_graph()
    print("\nCreated graph using default template (PowerDepLossFit not found)")

ly = gp[0]

# Scatter plot — raw data with error bars
sc = ly.add_plot(out_wks, coly=base_col + 1, colx=base_col + 0,
                 type='s', colyerr=base_col + 2)
sc.name  = RESONATOR_NUM
sc.color = '#167BB2'

# Line plot — fitted curve (legend picks up Q_TLS from column comment)
ln = ly.add_plot(out_wks, coly=base_col + 4, colx=base_col + 3, type='l')
ln.name  = f"Q\\-(TLS)={Q_TLS:.2e}±{Q_TLS_err:.2e}"
ln.color = '#D62728'

ly.xscale = 2   # log10 x-axis
ly.set_xlim(photon.min(), photon.max(), 2)
ly.rescale()

print(f"\nGraph created successfully for {RESONATOR_NUM}")
print(f"Q_TLS = {Q_TLS:.4e} ± {Q_TLS_err:.4e}")
print("Script completed!")