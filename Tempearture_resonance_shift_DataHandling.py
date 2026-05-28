import re, math
import originpro as op

# --- Configuration ----------------------------------------------------------
SRC     = 'LossMat36Res1'
TD_NAME = 'Temperature_dependent'

# --- 1) Open source book ----------------------------------------------------
src_book = op.find_book('w', SRC)
if src_book is None:
    raise RuntimeError(f"Book '{SRC}' not found")

# --- 2) Get or create the Temperature_dependent sheet -----------------------
td_wks = op.find_sheet('w', f'[{SRC}]{TD_NAME}')
if td_wks is None:
    td_wks = src_book.add_sheet(TD_NAME)

# --- 3) Locate the source worksheets ---------------------------------------
fv_wks = op.find_sheet('w', f'[{SRC}]freq_values')
fe_wks = op.find_sheet('w', f'[{SRC}]freq_err_values')
pn_wks = op.find_sheet('w', f'[{SRC}]photon_numbers')
for ws, name in [(fv_wks,'freq_values'), (fe_wks,'freq_err_values'), (pn_wks,'photon_numbers')]:
    if ws is None:
        raise RuntimeError(f"Sheet '{name}' not found in '{SRC}'")

# --- 4) Temperatures (mK→K) from Comments of fv columns B→end --------------
n_temps = fv_wks.cols - 1
temps_K = []
for col in range(1, n_temps+1):
    com = fv_wks.get_label(col, type='C') or '0'
    try:
        mK = float(com)
    except:
        mK = 0.0
    temps_K.append(mK / 1e3)

# --- 5) Power list from photon_numbers Col A -------------------------------
powers   = pn_wks.to_list(0)
n_powers = len(powers)

# --- 6) Robust reader: turn every cell into float or None ------------------
def read_rows(wks, start_col):
    rows = []
    for row_idx in range(n_powers):
        r = []
        for c in range(start_col, start_col + n_temps):
            try:
                raw = wks.to_list(c)[row_idx]
                val = float(raw)
                if math.isnan(val):
                    val = None
            except:
                val = None
            r.append(val)
        rows.append(r)
    return rows

fv_rows = read_rows(fv_wks, 1)
fe_rows = read_rows(fe_wks, 1)

# --- 7) Compute Δfr: subtract highest-temp value in each row --------------
dfr_rows = []
for row in fv_rows:
    highest = row[0]
    dfr_rows.append([
        None if (v is None or highest is None) else v - highest
        for v in row
    ])

# --- 8) Resize the target sheet -------------------------------------------
td_wks.cols = 1 + 3 * n_powers

# --- 9) Fill Column A: Temperature (K) --------------------------------------
td_wks.from_list(0, temps_K, lname='Temperature', units='K')

# --- 10) Fill FR columns (B→…) with comments = power -----------------------
for i, row in enumerate(fv_rows):
    col_idx = 1 + i
    td_wks.from_list(col_idx, row or [], lname=f'fr_P{i+1}', units='Hz')
    td_wks.set_label(col_idx, str(powers[i]), type='C')

# --- 11) Fill ΔFR columns (next block) -------------------------------------
for i, row in enumerate(dfr_rows):
    col_idx = 1 + n_powers + i
    td_wks.from_list(col_idx, row or [], lname=f'Δfr_P{i+1}', units='Hz')
    td_wks.set_label(col_idx, str(powers[i]), type='C')

# --- 12) Fill ERR columns (final block) -----------------------------------
for i, row in enumerate(fe_rows):
    col_idx = 1 + 2*n_powers + i
    td_wks.from_list(col_idx, row or [], lname=f'err_P{i+1}', units='Hz')
    td_wks.set_label(col_idx, str(powers[i]), type='C')

print("Temperature_dependent sheet updated with power comments.")
