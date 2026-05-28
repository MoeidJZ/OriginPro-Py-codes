import re
import originpro as op

# --- Configuration ----------------------------------------------------------
SRC = 'Res1AllTemp'
TGT = 'LossRes1'
mapping = {
    'delta':           18,  # S → index 18
    'delta_error':     19,  # T → index 19
    'freq_values':      5,  # F → index 5
    'freq_err_values': 11,  # L → index 11
    'photon_numbers':   1,  # B → index 1
}

def idx_to_col_letter(idx):
    """
    Zero-based idx → Excel-style column letter(s):
    0->A, 1->B, …, 25->Z, 26->AA, etc.
    """
    s = ''
    while True:
        s = chr(ord('A') + (idx % 26)) + s
        idx //= 26
        if idx == 0:
            break
        idx -= 1
    return s

# --- 1) Find source book and its first sheet -------------------------------
src_book = op.find_book('w', SRC)
if src_book is None:
    raise RuntimeError(f"Source book '{SRC}' not found")

sheets = list(src_book)
if not sheets:
    raise RuntimeError(f"No worksheets in source book '{SRC}'")
first_wks = sheets[0]

# --- 2) Find or create the target book -------------------------------------
tgt_book = op.find_book('w', TGT)
if tgt_book is None:
    op.new_sheet(type='w', lname=TGT)
    tgt_book = op.find_book('w', TGT)

# --- 3) Process each mapped sheet ------------------------------------------
for sheet_name, src_col_idx in mapping.items():
    # 3a) Find or create the worksheet
    tgt_wks = op.find_sheet('w', f'[{TGT}]{sheet_name}')
    if tgt_wks is None:
        tgt_book.add_sheet()
        new_ws = list(tgt_book)[-1]
        new_ws.name = sheet_name
        tgt_wks = new_ws

    # 3b) Column A: relation back to FIRST sheet’s Col A
    letterA = idx_to_col_letter(0)  # "A"
    fA = f'[{SRC}]"{first_wks.name}"!{letterA}'
    tgt_wks.set_formula(0, fA)      # writes into F(x)= row :contentReference[oaicite:0]{index=0}

    # 3c) Gather & sort temperature sheets
    temp_sheets = []
    for w in src_book:
        m = re.search(r'(\d+\.?\d*)\s*mK', w.name)
        if m:
            temp_sheets.append((float(m.group(1)), w))
    temp_sheets.sort(key=lambda x: x[0])

    # 3d) Columns B→…: relation back to each source column; label temp in Comments
    for i, (tval, src_wks) in enumerate(temp_sheets):
        col_idx = i + 1
        if tgt_wks.cols <= col_idx:
            tgt_wks.cols = col_idx + 1

        # build formula using the source column's letter
        src_letter = idx_to_col_letter(src_col_idx)
        formula = f'[{SRC}]"{src_wks.name}"!{src_letter}'
        tgt_wks.set_formula(col_idx, formula)

        # Comments = numeric temperature (no "mK")
        label = str(int(tval)) if tval.is_integer() else str(tval)
        tgt_wks.set_label(col_idx, label, type='C')  # Comments row :contentReference[oaicite:1]{index=1}
