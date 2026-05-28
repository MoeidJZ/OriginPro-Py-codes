"""
Import resonator CSV data into OriginPro.

Run this from Origin's embedded Python editor (Code Builder / Script Window).

For each CSV file listed below:
  - Each unique value in column 'chip_res' becomes a separate WORKBOOK.
      e.g.  'TaCN2.0_R1_Res2'  -> short name 'TaCN20R1Res2'
                               -> long  name 'TaCN2.0_R1_Res2'
  - For each workbook, each unique value in 'temperature_mK' becomes a SHEET.
      e.g.  60 mK with resonator Res2 -> sheet 'Res2_60 mK'
  - Filters applied BEFORE inserting rows:
        1e-4  <= photon_count <= 1e12
        1000  <= Qi           <= 1e8
  - Origin sheet layout (columns A..M filled from CSV, N empty,
    O..V are calculated from A..L):

        A  good_power     (dBm)              <- power_dBm
        B  ñ                                  <- photon_count
        C  Qi                                 <- Qi
        D  Qc                                 <- Qc
        E  Qr                                 <- Ql
        F  fr              (Hz)               <- fr
        G  phi                                <- phi
        H  tau                                <- tau
        I  Qi_err                             <- Qi_err
        J  Qc_err                             <- Qc_err
        K  Q_total_err                        <- Ql_err
        L  fr_err                             <- fr_err
        M  phi_err                            <- (empty; no source column)
        N  (blank spacer)
        O  Qi              x10^5    Res#, X.XX GHz   = C / 1e5
        P  Qi_err          x10^5                     = I / 1e5
        Q  Qc              x10^4                     = D / 1e4
        R  Qc_err          x10^4                     = J / 1e4
        S  delta                                     = 1 / C
        T  delta_err                                 = I / C^2
        U  delta           ppm                       = 1e6 / C
        V  delta_err       ppm                       = 1e6 * I / C^2
"""

import os
import re
import csv
import originpro as op

# ---------------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------------
# Folder containing the CSV files and the file names to import.
CSV_FOLDER = r'D:\Moeid_HPC_computer\Probst_Circuit'
CSV_FILES  = [
    'NYU_Resonators_TaCN2_R1_R5NT.csv',
    # add more file names here
]

# Filter limits
PHOTON_MIN, PHOTON_MAX = 1e-4, 1e12
QI_MIN,     QI_MAX     = 1e3,  1e8

# ---------------------------------------------------------------------------
# CSV column names (must match the header row in the CSV exactly)
# ---------------------------------------------------------------------------
COL_CHIP_RES   = 'chip_res'
COL_TEMP_MK    = 'temperature_mK'
COL_POWER      = 'power_dBm'
COL_PHOTON     = 'photon_count'
COL_QI         = 'Qi'
COL_QC         = 'Qc'
COL_QL         = 'Ql'
COL_FR         = 'fr'
COL_PHI        = 'phi'
COL_TAU        = 'tau'
COL_QI_ERR     = 'Qi_err'
COL_QC_ERR     = 'Qc_err'
COL_QL_ERR     = 'Ql_err'
COL_FR_ERR     = 'fr_err'

# ---------------------------------------------------------------------------
# Origin sheet layout  (column index, long name, units, comments)
#   - long name  -> goes to "Long Name" header row
#   - units      -> goes to "Units" header row
#   - comments   -> goes to "Comments" header row (only set where needed)
# Order of tuples = order of columns A, B, C, ...
# ---------------------------------------------------------------------------
ORIGIN_COLUMNS = [
    # (long name, units, comments, source CSV column or None)
    ('good_power',  'dBm', '', COL_POWER),     # A
    (u'\u00f1',     '',    '', COL_PHOTON),    # B  (ñ)
    ('Qi',          '',    '', COL_QI),        # C
    ('Qc',          '',    '', COL_QC),        # D
    ('Qr',          '',    '', COL_QL),        # E
    ('fr',          'Hz',  '', COL_FR),        # F
    ('phi',         '',    '', COL_PHI),       # G
    ('tau',         '',    '', COL_TAU),       # H
    ('Qi_err',      '',    '', COL_QI_ERR),    # I
    ('Qc_err',      '',    '', COL_QC_ERR),    # J
    ('Q_total_err', '',    '', COL_QL_ERR),    # K
    ('fr_err',      '',    '', COL_FR_ERR),    # L
    ('phi_err',     '',    '', None),          # M (no CSV source)
]
N_INPUT_COLS = len(ORIGIN_COLUMNS)             # 13  (A..M)

# Calculated columns start at index 14 (column O), leaving N (index 13) blank.
CALC_START_IDX = 14   # O

# Each tuple: (long name, units, formula expression using A..M references)
#   We'll fill these in by computing values in Python (not Origin formulas)
#   so filtering happens cleanly. Long name + units + comment are still set.
CALC_COLUMNS = [
    ('Qi',        'x10^5', '__FR_COMMENT__'),  # O
    ('Qi_err',    'x10^5', ''),                # P
    ('Qc',        'x10^4', ''),                # Q
    ('Qc_err',    'x10^4', ''),                # R
    (u'\u03b4',   '',      ''),                # S  (delta)
    (u'\u03b4_err','',     ''),                # T
    (u'\u03b4',   'ppm',   ''),                # U
    (u'\u03b4_err','ppm',  ''),                # V
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_chip_res(chip_res_str):
    """
    Split a 'chip_res' value like 'TaCN2.0_R1_Res2' into (chip, res_label, res_num).
        chip      = 'TaCN2.0_R1'
        res_label = 'Res2'
        res_num   = 2
    """
    m = re.search(r'_(Res\d+)\s*$', chip_res_str)
    if not m:
        # Fall back: use the whole string as chip, no resonator parsed
        return chip_res_str, 'Res?', None
    res_label = m.group(1)
    res_num   = int(re.search(r'\d+', res_label).group(0))
    chip      = chip_res_str[:m.start()]
    return chip, res_label, res_num


def make_book_short_name(chip_res_str):
    """
    'TaCN2.0_R1_Res2'  ->  'TaCN20R1Res2'
    'TaCN2.0_R5(NT)_Res1' -> 'TaCN20R5NTRes1'
    Strips '.', '_', '(', ')' and spaces.
    Origin short names must be <= 17 chars; we trim if needed.
    """
    s = re.sub(r'[._()\s]', '', chip_res_str)
    return s[:17]


def fmt_temp(t):
    """ 60.0 -> '60'   59.5 -> '59.5' """
    try:
        f = float(t)
    except (TypeError, ValueError):
        return str(t)
    if f.is_integer():
        return str(int(f))
    return ('%g' % f)


def to_float(s):
    """ Convert a CSV cell to float; return None if blank or unparseable. """
    if s is None:
        return None
    s = s.strip()
    if s == '':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def median(values):
    """ Median of a non-empty list of floats. """
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------------------
# Read + filter CSV rows, grouped by (chip_res, temperature)
# ---------------------------------------------------------------------------
def read_grouped_rows(csv_path):
    """
    Returns dict:
        { chip_res_str: { temp_value: [row_dict, ...] } }
    Rows that fail the photon_count or Qi filter are dropped.
    """
    groups = {}
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            qi  = to_float(row.get(COL_QI))
            ph  = to_float(row.get(COL_PHOTON))
            # Filters
            if qi is None or qi < QI_MIN or qi > QI_MAX:
                continue
            if ph is None or ph < PHOTON_MIN or ph > PHOTON_MAX:
                continue
            chip_res = (row.get(COL_CHIP_RES) or '').strip()
            if not chip_res:
                continue
            temp = to_float(row.get(COL_TEMP_MK))
            if temp is None:
                continue
            groups.setdefault(chip_res, {}).setdefault(temp, []).append(row)
    return groups


# ---------------------------------------------------------------------------
# Origin write helpers
# ---------------------------------------------------------------------------
def get_or_create_book(short_name, long_name):
    """ Find a workbook by short name; create if absent. """
    wb = op.find_book('w', short_name)
    if wb is not None:
        return wb
    wb = op.new_book(type='w', lname=long_name, hidden=False)
    # Origin auto-assigns a short name; rename it to what we want.
    try:
        wb.short_name = short_name
    except Exception:
        # Some originpro versions use .name for short name
        wb.name = short_name
    try:
        wb.lname = long_name
    except Exception:
        pass
    # New books come with one default sheet ('Sheet1') — we'll reuse or remove
    # it later as needed.
    return wb


def get_or_create_sheet(wb, sheet_name):
    """ Return the sheet with the given short name, creating it if needed. """
    # Try direct lookup
    for s in wb:
        try:
            if s.name == sheet_name:
                return s
        except Exception:
            pass
    # Reuse the default 'Sheet1' if it exists and is empty
    sheets = list(wb)
    if len(sheets) == 1:
        s0 = sheets[0]
        try:
            if s0.name in ('Sheet1', 'Sheet'):
                s0.name = sheet_name
                return s0
        except Exception:
            pass
    # Otherwise add a new sheet
    wb.add_sheet(name=sheet_name, active=True)
    for s in wb:
        if s.name == sheet_name:
            return s
    # Last resort: return the most recently added sheet
    return list(wb)[-1]


def set_header(ws, col_idx, long_name=None, units=None, comments=None):
    """ Set Long Name / Units / Comments for one column. """
    if long_name is not None:
        ws.set_label(col_idx, long_name, type='L')
    if units is not None:
        ws.set_label(col_idx, units, type='U')
    if comments is not None:
        ws.set_label(col_idx, comments, type='C')


def write_sheet(ws, rows, res_num):
    """
    Fill `ws` with the input columns A..M from `rows`, then compute O..V.
    """
    n_rows = len(rows)
    if n_rows == 0:
        return

    # Make sure there are enough columns. We need indices up to V = 21.
    needed_cols = CALC_START_IDX + len(CALC_COLUMNS)   # 14 + 8 = 22
    if ws.cols < needed_cols:
        ws.cols = needed_cols

    # ---- Build column data for A..M from the CSV rows ----
    for col_i, (lname, units, comments, csv_key) in enumerate(ORIGIN_COLUMNS):
        set_header(ws, col_i, long_name=lname, units=units, comments=comments)
        if csv_key is None:
            # leave column empty
            continue
        col_data = [to_float(r.get(csv_key)) for r in rows]
        ws.from_list(col_i, col_data)

    # Column N (index 13): leave blank, no header
    set_header(ws, 13, long_name='', units='', comments='')

    # ---- Pull Qi and Qi_err back as Python lists for calculations ----
    qi_list      = [to_float(r.get(COL_QI))     for r in rows]
    qi_err_list  = [to_float(r.get(COL_QI_ERR)) for r in rows]
    qc_list      = [to_float(r.get(COL_QC))     for r in rows]
    qc_err_list  = [to_float(r.get(COL_QC_ERR)) for r in rows]
    fr_list      = [to_float(r.get(COL_FR))     for r in rows]

    def safe(fn, *args):
        try:
            for a in args:
                if a is None:
                    return None
            v = fn(*args)
            return v
        except (ZeroDivisionError, TypeError, ValueError):
            return None

    o_qi_scaled    = [safe(lambda q: q / 1e5, qi) for qi in qi_list]
    p_qi_err_scl   = [safe(lambda e: e / 1e5, e)  for e  in qi_err_list]
    q_qc_scaled    = [safe(lambda q: q / 1e4, qc) for qc in qc_list]
    r_qc_err_scl   = [safe(lambda e: e / 1e4, e)  for e  in qc_err_list]
    s_delta        = [safe(lambda q: 1.0 / q, qi) for qi in qi_list]
    t_delta_err    = [safe(lambda e, q: e / (q * q), e, q)
                      for e, q in zip(qi_err_list, qi_list)]
    u_delta_ppm    = [safe(lambda q: 1e6 / q, qi) for qi in qi_list]
    v_dlt_err_ppm  = [safe(lambda e, q: 1e6 * e / (q * q), e, q)
                      for e, q in zip(qi_err_list, qi_list)]

    calc_data = [o_qi_scaled, p_qi_err_scl,
                 q_qc_scaled, r_qc_err_scl,
                 s_delta,     t_delta_err,
                 u_delta_ppm, v_dlt_err_ppm]

    # ---- Comment for column O: 'Res#, X.XX GHz' (X.XX = median fr in GHz) ----
    fr_clean = [f for f in fr_list if f is not None]
    if fr_clean:
        fr_ghz = median(fr_clean) / 1e9
        fr_comment = 'Res%d, %.2f GHz' % (res_num if res_num is not None else 0,
                                          fr_ghz)
    else:
        fr_comment = ''

    # ---- Write calculated columns O..V ----
    for j, (lname, units, comment) in enumerate(CALC_COLUMNS):
        col_i = CALC_START_IDX + j
        if comment == '__FR_COMMENT__':
            comment = fr_comment
        set_header(ws, col_i, long_name=lname, units=units, comments=comment)
        ws.from_list(col_i, calc_data[j])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    for fname in CSV_FILES:
        path = os.path.join(CSV_FOLDER, fname)
        if not os.path.isfile(path):
            print('SKIP (not found):', path)
            continue

        print('Reading:', path)
        groups = read_grouped_rows(path)
        if not groups:
            print('  No rows passed the filters.')
            continue

        for chip_res_str, temp_dict in groups.items():
            chip, res_label, res_num = parse_chip_res(chip_res_str)
            short = make_book_short_name(chip_res_str)
            wb = get_or_create_book(short_name=short, long_name=chip_res_str)
            print('  Workbook: [%s]  (%s)' % (short, chip_res_str))

            # Sort temperatures ascending so sheets appear in a sensible order
            for temp in sorted(temp_dict.keys()):
                rows = temp_dict[temp]
                sheet_name = '%s_%s mK' % (res_label, fmt_temp(temp))
                ws = get_or_create_sheet(wb, sheet_name)
                write_sheet(ws, rows, res_num)
                print('    Sheet: %s   (%d rows)' % (sheet_name, len(rows)))

    print('Done.')


if __name__ == '__main__':
    main()