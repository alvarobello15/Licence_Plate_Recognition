# Spanish License Plate OCR — Char‑Level Segmentation + EasyOCR + Metrics

Robust OCR pipeline to read **Spanish license plates (format: 4 digits + 3 letters)** from already-cropped plate images.  
It segments characters, classifies each one with **EasyOCR**, and (optionally) evaluates predictions against a ground truth file, producing a confusion matrix, ROC/AUC, and a summary report.

> Script: `OCR_Def_with_metrics.py`

---

## Features

- **End‑to‑end pipeline (per image):**
  1) Robust binarization (handles uneven lighting).
  2) **Contour-based character segmentation** → up to 7 ordered boxes (left→right). Removes the blue EU band when present.
  3) **Character crop normalization** to `64×64` (with padding, aspect preserved).
  4) **Per-position allowlist** for EasyOCR: digits only in positions 0..3, Spanish capital consonants in 4..6 (`BCDFGHJKLMNPRSTVWXYZ`).
  5) **Light disambiguation** of common confusions (e.g., `0/O`, `1/I/L`, `5/S`, `6/G`, `8/B`, `7/T`), while still forcing the final `4+3` pattern.
  6) **Outputs**: per-image overlay, character crops, per-image TXT, and a global `results.csv` with per‑char confidences.
- **Optional evaluation** with ground truth (CSV/XLSX):
  - Char‑level accuracy, precision/recall/F1 (micro & macro).
  - **Confusion matrix** (CSV + heatmap PNG).
  - **ROC curve** for the binary criterion “character correct vs. EasyOCR confidence” with **AUC**.
- Recursive scan for common image formats (`jpg, png, bmp, webp, tif`, etc.).

---

## Requirements

- Python 3.8+
- Runtime packages:  
  `easyocr`, `opencv-python-headless`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`  
  + **PyTorch** (CPU or GPU) as EasyOCR dependency

```bash
# Minimal install (CPU)
pip install easyocr opencv-python-headless numpy pandas scikit-learn matplotlib

# If you want GPU acceleration (recommended when available)
# Install PyTorch following https://pytorch.org/get-started/locally/ for your CUDA version,
# then install EasyOCR and the rest.
```

---

## Quickstart

### 1) Process images only
```bash
python OCR_Def_with_metrics.py   --in_dir "./outputs_engi/plates"   --out_dir "./out/ocr_chars_easyocr_local"
```

### 2) Process **and** evaluate with ground truth (CSV or XLSX)
```bash
python OCR_Def_with_metrics.py   --in_dir "./outputs_engi/plates"   --out_dir "./out/ocr_chars_easyocr_local"   --gt_path "./groundtruth.csv"   --gt_image_col "image"   --gt_plate_col "plate"   --gpu
```

> **Windows example**
> ```powershell
> python OCR_Def_with_metrics.py --in_dir "C:\path	o\plates" --out_dir "C:\path	o\out" --gt_path "C:\path	o\gt.xlsx" --gt_image_col "foto" --gt_plate_col "matricula" --gt_sheet "Sheet1"
> ```

**CLI arguments (most used):**
- `--in_dir`: folder with the cropped plate images (searches recursively).
- `--out_dir`: output folder.
- `--gpu`: enable GPU in EasyOCR if available.
- `--gt_path`: path to **CSV/XLSX** with ground truth.
- `--gt_image_col`: name of the image identifier column in GT (default `image`).
- `--gt_plate_col`: name of the license plate column in GT (default `plate`).
- `--gt_sheet`: sheet name when using Excel.

---

## Outputs

```
out/
├─ results.csv                  # One row per image: predicted plate + per‑char predictions & confidences
├─ all.txt                      # All predicted plates (one per line)
├─ unique.txt                   # Unique predicted plates (sorted)
├─ overlays/
│  └─ <image>.jpg               # Original image with green boxes + full plate printed
├─ crops/
│  └─ <image>/
│     └─ 0_<char>.png ...       # Normalized per‑char crops (64×64)
├─ <image>.txt                  # One-line plate string per image
└─ metrics/                     # (Only if --gt_path is set)
   ├─ metrics.txt               # Summary: exact-match accuracy + char-level metrics
   ├─ confusion_matrix.csv      # Char-level confusion counts
   ├─ cm.png                    # Confusion matrix heatmap
   └─ roc_auc.png               # ROC curve (char correct vs. confidence)
```

### `results.csv` columns
- `image`, `plate`
- For each position `i=0..6`:
  - `p{i}`: predicted character at position `i` (empty if unread).
  - `c{i}`: **EasyOCR confidence** for the chosen char at position `i`.
  - `raw{i}`: raw char read before positional disambiguation.

---

## Ground Truth Formats

You can use **either**:
1) **Two columns** (recommended):
   - `image`: image identifier (basename with or **without** extension)
   - `plate`: target plate string (any non-alphanumerics are stripped; uppercased)

   **CSV example:**
   ```csv
   image,plate
   1000052160,1234BCD
   1000052161,5678FGH
   ```

2) **Single column** with `image,plate` in each row (auto‑split).  
   Example row contents: `1000052160,1234BCD`

> During evaluation, the script **normalizes** the image ID to the **lowercased basename without extension** and filters GT rows to exact length 7 after cleaning. This helps match `results.csv` rows with your GT even if extensions or paths differ.

---

## Metrics (when `--gt_path` is provided)

- **Exact‑match accuracy** (full plate equality).
- **Char‑level metrics**: accuracy, precision/recall/F1 (micro & macro), full classification report.
- **Confusion matrix** (`confusion_matrix.csv` + `cm.png`).  
- **ROC** for the binary task “char is correct” using per‑char confidence (`c0..c6`) → `roc_auc.png` with AUC.

Open `out/metrics/metrics.txt` after a run to see the numeric summary.

---

## Tips & Notes

- Language: EasyOCR initialized with `['en']` is sufficient for digits and capital Latin letters.
- Allowlist by position:
  - Positions **0–3**: digits `0–9` only.
  - Positions **4–6**: letters `BCDFGHJKLMNPRSTVWXYZ` (Spanish plate consonants; no vowels, no `Q/Ñ`).  
- The pipeline enforces the final **4+3** pattern after per‑char reads.
- The segmenter removes the left EU band if it contains significant “ink” and includes a fallback that splits overly wide boxes when fewer than 7 chars are found.
- Supported image extensions (recursive search): `jpg, jpeg, png, bmp, webp, tif, tiff` (upper/lowercase).

---

## Troubleshooting

- **“No images found”**: check `--in_dir` path and that your images use one of the supported extensions. The search is recursive in subfolders.
- **“No intersection between results.csv and GT on ‘image’”**: ensure that the GT **image identifiers** match the **basenames** written to `results.csv`. The evaluator lowercases and strips extensions. If your output names are like `plate00`, your GT must reference that same basename (not a numeric ID). You can also remap/rename either side for consistency.
- **Low recall on certain characters**: verify plate crops are well framed; adjust upstream detection/cropping if necessary. Consider enabling `--gpu` for faster, potentially more stable inference.
- **Very wide or fused characters**: the fallback box‑splitter helps, but extremely tight crops may still need loosening the original crop or pre‑processing parameters.

---

## Example Repo Structure

```
your-repo/
├─ OCR_Def_with_metrics.py
├─ README.md
├─ outputs_engi/
│  └─ plates/               # Your input images (any nested folders)
└─ out/
   └─ ...                   # Created after running the script
```

---

## Acknowledgements

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) and its PyTorch backend.

---

## 📜 License

Choose a license for your repository (e.g., MIT, Apache‑2.0). Add it as `LICENSE` at the repo root.
