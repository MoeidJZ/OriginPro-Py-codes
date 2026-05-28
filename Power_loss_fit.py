import numpy as np
import originpro as op
from scipy.optimize import curve_fit

# ── USER SETTINGS ────────────────────────────────────────────────────────────
SRC_BOOK       = 'LossMatW5Res8'
temperature_mK = 50    # which column-comment to pick
NUM_ATTEMPTS   = 500
PARAM_RANGES   = {
    'F_delta_TLS0': (7e-6, 15e-6),
    'b':            (0.1, 1000.0),
    'beta2':        (0.1, 1.0),
    'Q_other':      (6e5, 8e5)
}
DENSE_POINTS  = 1000
EXPORT_POINTS = 200
# ─────────────────────────────────────────────────────────────────────────────
# 1) Open sheets
book   = op.find_book('w', SRC_BOOK)
pn_wks = op.find_sheet('w', f'[{SRC_BOOK}]photon_numbers')
fr_wks = op.find_sheet('w', f'[{SRC_BOOK}]freq_values')
d_wks  = op.find_sheet('w', f'[{SRC_BOOK}]delta')
de_wks = op.find_sheet('w', f'[{SRC_BOOK}]delta_error')
for ws in (pn_wks, fr_wks, d_wks, de_wks):
    if ws is None:
        raise RuntimeError("Missing one of the required sheets")

# 2) Locate the target column by its comment
target_col = None
for c in range(1, pn_wks.cols):
    lbl = pn_wks.get_label(c, type='C') or ''
    try:
        if int(float(lbl)) == temperature_mK:
            target_col = c
            break
    except ValueError:
        continue
if target_col is None:
    raise RuntimeError(f"No column commented “{temperature_mK}”")

# 3) Read & filter data
raw_ph = pn_wks .to_list(target_col)
raw_fr = fr_wks .to_list(target_col)
raw_d  = d_wks  .to_list(target_col)
raw_de = de_wks .to_list(target_col)
valid = [i for i,(p,f,d,er) in enumerate(zip(raw_ph,raw_fr,raw_d,raw_de))
         if p not in ('',None) and f not in ('',None)
            and d not in ('',None) and er not in ('',None)]
if not valid:
    raise RuntimeError("No valid data in that column")

photon    = np.array([float(raw_ph[i])   for i in valid])
freq      = np.array([float(raw_fr[i])   for i in valid])
delta     = np.array([float(raw_d[i])    for i in valid])
delta_err = np.array([float(raw_de[i])   for i in valid])

# 4) Sort by photon
order = np.argsort(photon)
photon, freq, delta, delta_err = [arr[order] for arr in (photon, freq, delta, delta_err)]

# 5) Prepare fit
T_K = temperature_mK / 1e3
def model_power(x, F0, b, beta2, Q_other):
    f, n = x
    tt = np.tanh(6.634e-34 * f / (2 * 1.38e-23 * T_K))
    return F0*(tt/np.sqrt(1 + (n**beta2)/b * tt)) + 1.0/Q_other

def weighted_mse(y, yfit, err, photon):
    w  = 1.0/np.maximum(err**2, 1e-10)
    pw = np.ones_like(photon); pw[photon>1] = 3.0
    W  = w * pw
    return np.sum((y - yfit)**2 * W)/np.sum(W)

def run_fit():
    best_mse, best_p = np.inf, None
    xdata = (freq, photon)
    for _ in range(NUM_ATTEMPTS):
        p0 = [np.random.uniform(*PARAM_RANGES[k])
              for k in ('F_delta_TLS0','b','beta2','Q_other')]
        bounds = (
                        [0, 0, 0, 0],
                        [np.inf, np.inf, np.inf, np.inf]
                )
        try:
            popt, _ = curve_fit(
                lambda x,F0,b,b2,Q: model_power(x, F0,b,b2,Q),
                xdata, delta, sigma=delta_err,
                absolute_sigma=True, p0=p0, maxfev=20000
            )
        except:
            continue
        yfit = model_power(xdata, *popt)
        mse  = weighted_mse(delta, yfit, delta_err, photon)
        if mse < best_mse:
            best_mse, best_p = mse, popt
    return best_p, best_mse

# 6) Run the fit
popt, mse = run_fit()
if popt is None:
    raise RuntimeError("Fit failed")
F0, b, beta2, Q_other = popt

# ── 7) Export Fit Parameters ─────────────────────────────────────────────────
params_name = 'Fit Parameters'
params_wks  = op.find_sheet('w', f'[{SRC_BOOK}]{params_name}')
if params_wks is None:
    params_wks = book.add_sheet(params_name)
    params_wks.cols = 6  # Temp, F0, b, beta2, Q_other, MSE

# read existing temperatures (col 0)
existing = params_wks.to_list(0)
# filter out blanks & convert
existing_t = [int(float(x)) for x in existing if x not in (None,'')]
if temperature_mK not in existing_t:
    # build new full columns
    new_t   = existing_t + [temperature_mK]
    new_F0  = [float(x) for x in params_wks.to_list(1)] + [F0]
    new_b   = [float(x) for x in params_wks.to_list(2)] + [b]
    new_b2  = [float(x) for x in params_wks.to_list(3)] + [beta2]
    new_Qo  = [float(x) for x in params_wks.to_list(4)] + [Q_other]
    new_mse = [float(x) for x in params_wks.to_list(5)] + [mse]

    params_wks.from_list(0, new_t,   lname='Temp', units = 'mK')
    params_wks.from_list(1, new_F0,  lname='Fδ\-(TLS0)')
    params_wks.from_list(2, new_b,   lname='b')
    params_wks.from_list(3, new_b2,  lname='\g(b)\-(2)')
    params_wks.from_list(4, new_Qo,  lname='Q\-(other)')
    params_wks.from_list(5, new_mse, lname='Weighted MSE')
else:
    print(f"Fit Parameters for {temperature_mK} mK already present; skipping append.")

# ── 8) Build dense fit & down-sample ─────────────────────────────────────────
f0        = float(np.median(freq))
n_dense   = np.logspace(np.log10(photon.min()),
                        np.log10(photon.max()), DENSE_POINTS)
tt        = np.tanh(6.634e-34 * f0 / (2*1.38e-23*T_K))
den_den   = np.sqrt(1 + (n_dense**beta2)/b * tt)
fit_dense = F0*(tt/den_den) + 1.0/Q_other

idx       = np.round(np.linspace(0, DENSE_POINTS-1, EXPORT_POINTS)).astype(int)
n_export  = n_dense[idx]
fit_export= fit_dense[idx]

# ── 9) Export Power dependent Fit ────────────────────────────────────────────
out_name = 'Power dependent Fit'
out      = op.find_sheet('w', f'[{SRC_BOOK}]{out_name}')
if out is None:
    out        = book.add_sheet(out_name)
    base_col   = 0
    out.cols   = 9
else:
    # check for existing columns commented with this temperature
    exists = any(f"{temperature_mK} mK" in (out.get_label(c, type='C') or '')
                 for c in range(out.cols))
    if exists:
        print(f"Power dependent Fit for {temperature_mK} mK already present; skipping export.")
        base_col = None
    else:
        base_col      = out.cols
        out.cols     += 9

if base_col is not None:
    # original δ data + fit (unitless)
    out.from_list(base_col + 0, photon.tolist(),
                  lname='<n>',       comments=f'{temperature_mK} mK')
    out.from_list(base_col + 1, delta.tolist(),
                  lname='δ',         comments=f'{temperature_mK} mK, Fit: Fδ\-(TLS0)={F0:.2e}, Q\-(other)={Q_other:.2e}')
    out.from_list(base_col + 2, delta_err.tolist(),
                  lname='δ err',     comments=f'{temperature_mK} mK')
    out.from_list(base_col + 3, n_export.tolist(),
                  lname='<n>(fit)',  comments=f'{temperature_mK} mK')
    out.from_list(base_col + 4, fit_export.tolist(),
                  lname='Fit δ',      comments=f'{temperature_mK} mK')

    # ppm versions
    out.from_list(base_col + 6, (delta*1e6).tolist(),
                  lname='δ', units='ppm',
                  comments=f'{temperature_mK} mK, Fit: Fδ\-(TLS0)={F0:.2e}, Q\-(other)={Q_other:.2e}')
    out.from_list(base_col + 7, (delta_err*1e6).tolist(),
                  lname='δ err', units='ppm',
                  comments=f'{temperature_mK} mK')
    out.from_list(base_col + 8, (fit_export*1e6).tolist(),
                  lname='Fit', units='ppm',
                  comments=f'{temperature_mK} mK')

    # 10) Create the graph
    try:
        gp = op.new_graph(template='PowerDepLossFit')
    except:
        gp = op.new_graph(template='Line')
    ly = gp[0]

    # scatter: raw data (Y‐col base_col+6) vs raw photon (X‐col base_col+0)
    sc = ly.add_plot(out, base_col+6, base_col+0, type='s', colyerr=base_col+7)
    sc.name  = f"{temperature_mK} mK"
    sc.color = '#167BB2'

    # smooth fit curve
    ln = ly.add_plot(out, base_col+8, base_col+3, type='l')
    ln.name = f"Fit: Fδ\-(TLS0)={F0:.2e}, Q\-(other)={Q_other:.2e}"

    ly.xscale = 2        
    ly.set_xlim(photon.min(), photon.max(), 2)
    ly.rescale()
