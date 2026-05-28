import re
import numpy as np
import originpro as op

# --- Configuration ----------------------------------------------------------
SRC   = 'S21data'   # Workbook short name
TEMP  = '50'              # Temperature in mK
POWER = '-130'             # Power in dBm
Res   =  '2'
# ----------------------------------------------------------------------------

# --- 1) Find source book ----------------------------------------------------
src_book = op.find_book('w', SRC)
if src_book is None:
    raise RuntimeError(f"Source book '{SRC}' not found")

# --- 2) List worksheets ------------------------------------------------------
sheets = list(src_book)
if not sheets:
    raise RuntimeError(f"No worksheets in source book '{SRC}'")

# --- 3) Locate target sheet ------------------------------------------------
pattern = rf"^R\s*{re.escape(Res)}\s*+,\s*{re.escape(POWER)}\s*dBm,\s*{re.escape(TEMP)}\s*mK$"
target_sheet = next((w for w in sheets if re.match(pattern, w.name)), None)
if target_sheet is None:
    raise RuntimeError(f"No sheet matching pattern '{pattern}' in '{SRC}'")

# --- 4) Read and compute data -----------------------------------------------
def to_float_array(wks, col_name):
    vals = wks.to_list(col_name)
    arr = []
    for v in vals:
        try:
            arr.append(float(v))
        except (ValueError, TypeError):
            arr.append(np.nan)
    return np.array(arr)

mag_raw   = to_float_array(target_sheet, 'Mag S21 raw')
phase_raw = to_float_array(target_sheet, 'Phase S21 raw')
mag_fit   = to_float_array(target_sheet, 'Mag S21 fitted')
phase_fit = to_float_array(target_sheet, 'Phase S21 fitted')
freq_raw  = to_float_array(target_sheet, 'Frequency raw')
freq_fit  = to_float_array(target_sheet, 'Frequency Filtered')

# Reconstruct complex S21
s21_raw = mag_raw * np.exp(1j * phase_raw)
s21_fit = mag_fit * np.exp(1j * phase_fit)

# Derived quantities
Re_raw      = np.real(s21_raw)
Im_raw      = np.imag(s21_raw)
Re_fitted   = np.real(s21_fit)
Im_fitted   = np.imag(s21_fit)
f_raw_GHz   = freq_raw / 1e9
f_fit_GHz   = freq_fit / 1e9

S21raw_dB   = 20 * np.log10(np.abs(s21_raw))
phase_raw_d = np.degrees(np.angle(s21_raw))
S21fit_dB   = 20 * np.log10(np.abs(s21_fit))
phase_fit_d = np.degrees(np.angle(s21_fit))

# --- 5) Write/overwrite derived columns with proper units --------------------
base_col = 6  # first 6 are original inputs
target_sheet.cols = base_col + 10
cols = [Re_raw, Im_raw, Re_fitted, Im_fitted,
        f_raw_GHz, f_fit_GHz,
        S21raw_dB, phase_raw_d,
        S21fit_dB, phase_fit_d]
col_names = ['Re{S\-(21,raw)}', 'Imag{S\-(21,raw)}', 'Re{S\-(21,fitted)}', 'Imag{S\-(21,fitted)}',
             'f-(raw)', 'f-(fit)',
             '|S\-(21,raw)|', 'phase{S\-(21,raw)}',
             '|S\-(21,fitted)|', 'phase{S\-(21,fitted)}']
col_units = ['', '', '', '',           # unitless real/imag
             'GHz', 'GHz',       # frequency
             'dB', 'deg',        # magnitude & phase raw
             'dB', 'deg']       # magnitude & phase fit

for i, (data, lname, unit) in enumerate(zip(cols, col_names, col_units)):
    target_sheet.from_list(
        base_col + i,
        data.tolist(),
        lname=lname,
        units=unit,
        comments=f'{POWER} dBm, {TEMP} mK'
    )

# --- 6) Create three separate graphs using 's21fit' template if available ----
def new_s21_graph(name):
    gp = op.new_graph(template='Line')
    gp.name = name
    return gp

# Graph 1: |S21| vs Frequency
gp1 = new_s21_graph('Graph1')
ly1 = gp1[0]
p1 = ly1.add_plot(target_sheet, base_col+6, base_col+4, type='s')
p1.name = '|S21 raw|'
l1 = ly1.add_plot(target_sheet, base_col+8, base_col+5, type='l')
l1.name = '|S21 fitted|'
ly1.rescale()

# Graph 2: Phase vs Frequency
gp2 = new_s21_graph('Graph2')
ly2 = gp2[0]
p2 = ly2.add_plot(target_sheet, base_col+7, base_col+4, type='s')
p2.name = 'Phase raw'
l2 = ly2.add_plot(target_sheet, base_col+9, base_col+5, type='l')
l2.name = 'Phase fitted'
ly2.rescale()

# Graph 3: Imag vs Real
gp3 = new_s21_graph('Graph3')
ly3 = gp3[0]
p3 = ly3.add_plot(target_sheet, base_col+1, base_col+0, type='s')
p3.name = 'Imag vs Real raw'
l3 = ly3.add_plot(target_sheet, base_col+3, base_col+2, type='l')
l3.name = 'Imag vs Real fitted'
ly3.rescale()

# --- 7) Merge only the new graphs ------------------------------------------
graph_names = [gp1.name, gp2.name, gp3.name]
graphs_str = "\n".join(graph_names)
ltStr = f'merge_graph option:=specified option:=specified graphs:="{graphs_str}" row:=1 col:=3 ogp:="MergedGraph";'
op.lt_exec(ltStr)

# --- 8) Apply 's21fit' template to the merged graph -------------------------
merged = op.find_graph('MergedGraph')
if merged is None:
    raise RuntimeError("MergedGraph not found after merge; check merge_graph parameters.")
merged.name = 'MergedGraph'

print(f"Done: merged graphs {graph_names} into one window with 's21fit' template applied.")