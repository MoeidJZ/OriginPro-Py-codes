# OriginPro Integration: Superconducting Resonator Analysis & Fitting

This repository contains a suite of Python scripts designed exclusively to run inside **OriginPro's embedded Python environment**. These tools bridge the gap between automated data extraction (from the QRES fitting suite) and Origin's powerful graphing and data management capabilities.

With these scripts, you can seamlessly import wideband $S_{21}$ sweeps, dynamically organize data into multidimensional matrices, and utilize advanced GUI-driven fitting algorithms for Two-Level System (TLS) loss, power-dependent loss, and quasiparticle resonance shifts.

## 📥 Sample Data
A sample OriginPro project file, **`Sample_OriginPro_ResonatorData.opju`**, is included in this repository. It contains formatted data demonstrating the expected layout for all the Python scripts described below.

---

## ⚙️ OriginPro Python Setup

To use these scripts, you must run them from within OriginPro (Version 2021 or later is recommended, as it includes a fully embedded Python interpreter).

### How to Open Python in OriginPro
1. **Script Window:** Go to `Connectivity` > `Open untitled.py` to make a new file or press `Open` to open a certain python file.
2. **Code Builder (Recommended):** Go to `View` > `Code Builder`. This opens a full IDE where you can open, edit, and run the `.py` scripts provided in this repository. Ensure the language drop-down at the top of the Code Builder is set to `Python`.

### Installing Required Libraries
These scripts require external scientific libraries that are not packaged with Origin's default Python installation. 
1. Go to `Connectivity` > `Python Packages...` in the top Origin menu.
2. An installation window will appear. Click `Install` and type the following package names to add them:
   * `numpy`
   * `scipy`
   * `pandas`
*(Note: The `originpro` library is built-in and does not need to be installed).*

### Acknowledgments
The integration of Python within Origin relies heavily on the `originpro` API. The architecture of these scripts was guided by the following resources:
* [OriginLab originpro GitHub](https://github.com/originlab/originpro)
* [OriginLab Python-Samples](https://github.com/originlab/Python-Samples)
* [swharden/originpro](https://github.com/swharden/originpro)

---

## 🛠️ Script Workflow & Documentation

The 9 scripts in this repository are divided into three workflow categories: **Data Import & Formatting**, **Loss Extraction & Fitting**, and **S21 Visualization**.

### Phase 1: Data Import & Formatting
These scripts structure the raw CSV outputs from Part 3 into organized Origin workbooks.

**1. `Import_Res_Data.py`**
* **Function:** Imports CSV files exported from the Probst or QRES GUI and builds a structured Origin workbook for each resonator.
* **Structure:** Creates a new workbook per resonator. Inside the workbook, it creates a new worksheet for each temperature. Different measurement powers are arranged in rows. It also adds new columsn to calculate loss and loss errors from Qi and Qi_err columns.
* **User Inputs:** `CSV_FOLDER` (path to data) and `CSV_FILES` (list of files to import).

**2. `Comment_set_for_Qi.py` (Optional Utility)**
* **Function:** A helper script to batch-set comments on specific columns (e.g., Column O) to label metadata like sample, chip, resonator, and temperature.
* **User Inputs:** `book_name` (e.g., `'TaN1p0R8Res1'`).

**3. `Loss_extraction_Origin_Script.py`**
* **Function:** Extracts data from the multi-sheet temperature workbook (generated in Step 1) and converts it into unified matrices. This is **essential** for the multi-variable GUI fits.
* **Structure:** Creates a target workbook with dedicated sheets for: `photon_numbers`, `freq_values`, `delta`, `delta_error`. In these sheets, rows represent powers and columns represent temperatures.
* **User Inputs:** `SRC` (the source workbook name) and `TGT` (the desired output workbook name).

**4. `Tempearture_resonance_shift_DataHandling.py`**
* **Function:** Prepares data for resonance shift fitting. It calculates $\Delta f_r$ by subtracting the resonance frequency of the highest-temperature baseline from the rest of the data.
* **Structure:** Adds a `Temperature_dependent` sheet to the matrix workbook.
* **User Inputs:** `SRC` (workbook name) and `TD_NAME` (name of the target sheet).

---

### Phase 2: Loss & Resonance Shift Fitting
These scripts apply physical models to the organized Origin data.

**5. `BaseTemp_PowerLossFit.py`**
* **Function:** Fits power-dependent loss (TLS + power-independent losses) at a *single* base temperature. This script bypasses the matrix extraction requirement, allowing for flexible, quick-look fitting directly on the imported sheets.
* **User Inputs:** `SRC_BOOK` (e.g., 'TaN7p0R2'), `RESONATOR_NUM` (e.g., 'Res5'), and `PARAM_RANGES` (bounds for $F\delta_{TLS,0}$, $n_c$, $\beta$, and $Q_{other}$).
* **Structure:** Expects sheet names exactly in the format `Res{number}_{temperature} mK` (e.g., `Res1_50 mK`).

**6. `Power_loss_fit.py`**
* **Function:** Operates identically to `BaseTemp_PowerLossFit.py`, but points to the matrix-formatted workbook generated by `Loss_extraction_Origin_Script.py`.
* **User Inputs:** `SRC_BOOK` and `temperature_mK` (the specific temperature column to fit).

**7. `Total_loss_fit_V2.py`**
* **Function:** An advanced, interactive Tkinter GUI that fits loss data across *both* temperature and power simultaneously (7 fitting variables).
* **User Inputs:** `SRC_BOOK`. The user trims power/temperature ranges, selects optimization strategies (Differential Evolution vs. local polish), and adjusts sliders live before exporting the final parameters back into Origin.

**8. `ResonanceShift fit GUI.py`**
* **Function:** A Tkinter GUI dedicated to fitting fractional resonance shifts ($\Delta f_r / f_r$) against temperature. 
* **User Inputs:** Reads the `Temperature_dependent` sheet. Allows the user to toggle Quasiparticle models (BCS conductivity vs. dirty limit) and TLS models, adjust $T_c$, $\alpha$, and $F\delta_{TLS,0}$, and export the fitted curves back to Origin.

---

### Phase 3: S21 Visualization

**9. `s21_data_plotting.py`**
* **Function:** Takes raw and fitted $S_{21}$ complex data exported directly from the Probst GUI and generates three standardized plots: Magnitude vs. Frequency, Phase vs. Frequency, and Imaginary vs. Real (Smith Chart layout).
* **User Inputs:** `SRC` (workbook), `TEMP` (in mK), `POWER` (in dBm), and `Res` (resonator number).
* **Crucial Structure Requirements:** To prevent script failure, your Origin sheets MUST follow this exact naming convention: `R {Res}, {POWER} dBm, {TEMP} mK` (e.g., `R 2, -130 dBm, 50 mK`). 
    * **Column Layout:** Col 0 (Real Raw), Col 1 (Imag Raw), Col 4 (Freq), Col 5 (Freq Fit), Col 6 (|S21| Raw), Col 7 (Phase Raw), Col 8 (|S21| Fit), Col 9 (Phase Fit).

---

## 📚 References & Literature
The physical models for TLS saturation, quasiparticle density shifts, and power-dependent loss utilized in these fitting scripts are heavily based on the following works:

* **Example Data & Model Application:** [Jamalzadeh et al., Applied Physics Letters 127.9 (2025): 092601](https://pubs.aip.org/aip/apl/article/127/9/092601/3361167)
* **Kinetic Inductance & Loss Fitting:** [Applied Physics Letters 127.19 (2025): 192603](https://pubs.aip.org/aip/apl/article/127/19/192603/3372272)
* **TLS Dynamics in Superconducting Circuits:** [Burnett et al., Physical Review X 4 (2014): 041005](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.041005)
* **Dissertation Reference:** [ProQuest Theses & Dissertations](https://www.proquest.com/docview/1080814168?pq-origsite=gscholar&fromopenview=true&sourcetype=Dissertations%20&%20Theses)
