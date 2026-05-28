import originpro as op
import re

# 1) Find the workbook by its short tab name
book_name = 'TaN1p0R8Res1'
book = op.find_book('w', book_name)
if book is None:
    raise RuntimeError(f"Workbook {book_name} not found")

# 2) Parse Sample, Chip, Resonator from the book’s long name
parts = book.lname.split('_')
if len(parts) < 4:
    raise ValueError(f"Book long name '{book.lname}' must have at least 4 underscore-delimited pieces")
_, sample, chip, resonator = parts[:4]

# 3) Loop through every worksheet (each temperature) in the book
for wks in book:  
    m = re.search(r'\d+\.?\d*\s*mK', wks.name)
    temp = m.group(0) if m else wks.name
    comment = f"{sample},{chip},{resonator},{temp}"

    # 4a) Clear old comment in column O (zero-based index 14)
    wks.set_label(14, "", type='C')

    # 4b) Write the new comment
    wks.set_label(14, comment, type='C')
    wks.set_label(16, f"{temp}", type='C')
    
print("Set all the O columns' comments!")