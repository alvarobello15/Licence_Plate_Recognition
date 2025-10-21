
# License Plate OCR (offline, character by character)

This repository provides an offline pipeline to read modern Spanish liscence plates. The project combines a deterministic segmentation and per-character OCR approach with EasyOCR and optional evaluation against ground truth, and includes complementary YOLOv8 scripts to detect and crop license plates from full images before running OCR. Everything runs locally without cloud dependencies and produces reproducible artifacts for inspection and reporting.

# What this repository contains

This repository contains a main OCR script, two evaluation variants, and two YOLOv8 utilities for detection and cropping. The OCR scripts handle binarization, contour-based segmentation into seven left-to-right characters, normalization of each character to 64×64, and recognition with position-aware allowlists. The evaluation variants add metrics, plots, and K-Fold analysis when a CSV or Excel ground truth is available. The YOLOv8 scripts cover training a plate detector and running inference to crop plate regions that can be fed into the OCR stage.

# How it works

The OCR stage starts by converting the plate crop to grayscale and applying background normalization so that the characters become white on black with stable contrast under uneven lighting. The method removes the left EU band when it contains significant ink, finds external contours, filters them with simple geometric rules, and enforces a strict left-to-right order. When fewer than seven boxes appear, the algorithm attempts to split overly wide boxes and when more appear, it keeps the seven most height-consistent boxes. Each character is padded and resized to 64×64 while preserving aspect ratio, and then passed to EasyOCR one by one. Positions zero to three accept only digits and positions four to six accept only the Spanish consonants used in modern plates, which lowers substitution errors. A soft correction step maps common confusions such as O with 0 or I with 1 without altering the raw OCR outputs. The final string is forced to the pattern of four digits followed by three letters and is written with overlays and structured files that make debugging and evaluation straightforward.

# Requirements and installation

The OCR scripts require Python 3.8 or newer and depend on EasyOCR with PyTorch, OpenCV, NumPy, Pandas, scikit-learn, and Matplotlib. Install the dependencies with the following command and make sure that PyTorch is available in CPU or GPU form depending on your system.

```bash
pip install easyocr opencv-python-headless numpy pandas scikit-learn matplotlib
````

# Running the OCR

The typical execution only requires the input directory with plate crops and the output directory. The script scans subfolders recursively and processes common raster formats.

```bash
python OCR_Def.py --in_dir "./outputs_engi/plates" --out_dir "./out/ocr_chars_easyocr_local"
```

If your machine supports GPU acceleration through PyTorch and EasyOCR you can enable it.

```bash
python OCR_Def.py --in_dir "./my_plates" --out_dir "./out/ocr" --gpu
```

# Using the YOLOv8 detector

The repository includes a minimal training script for YOLOv8 and a testing script that runs inference, saves annotated predictions, and crops detected plates to feed the OCR stage. The usual workflow is to train or adopt a YOLOv8 model to detect license plates on full images, run the testing utility to generate plate crops, and then point the OCR script to the folder of generated crops.

```python
# Training_Yolov8.py (simplified)
model = YOLO('yolov8n.pt')
model.train(data=DATA_YAML, epochs=15, patience=15, imgsz=640, device=0, workers=2, batch=16, project=SAVE_ROOT, name="plates_v8n", exist_ok=True)
```

```bash
# Testing_Yolov8n.py (inference and crops)
python Testing_Yolov8n.py --model "/path/to/best.pt" --source "/path/to/images_or_folder" --outdir "/path/to/outputs" --conf 0.50 --imgsz 1280
```

# Ground truth and evaluation

When you provide a CSV or Excel ground truth, the OCR scripts merge results by the image base name and normalize the target plate to seven uppercase alphanumeric characters. The ground truth can either contain two explicit columns for the image name and the plate or a single column with the pattern image,plate per row, which the script will split automatically. After the merge, only rows that result in a seven-character plate are kept for metrics. The evaluation reports exact match for the full plate, character-level accuracy together with precision, recall, and F1 under micro and macro averaging, a confusion matrix saved as CSV and heatmap, and a ROC curve with its AUC computed from the per-character OCR confidences.

# Repository files explained

`OCR_Def.py` is the main OCR pipeline that performs binarization, segmentation, normalization, and per-character recognition with EasyOCR, saving overlays, crops, a per-image text file, a CSV with predictions and confidences, and list files of all and unique predictions. It optionally evaluates against ground truth when you pass a CSV or XLSX file and writes a metrics folder with summaries and plots.

`OCR_Def_with_metrics.py` is a variant of the main OCR script focused on explicit metric reporting. It keeps the same pipeline and outputs but emphasizes the generation of confusion matrices, character-level reports, and ROC/AUC graphics inside an out_dir/metrics folder, which is convenient when the primary goal is auditing and reporting.

`OCR_Def_kfold.py` extends the evaluation to K-Fold analysis using the intersection between results and ground truth. It computes exact-match and character accuracy per fold, writes a textual summary with mean and standard deviation, and produces simple bar plots that visualize per-fold performance to understand stability across splits.

`Training_Yolov8.py` defines a concise training setup for YOLOv8 using a nano backbone by default. It reads a dataset configuration, trains for a configurable number of epochs with basic hyperparameters, and stores runs under the chosen project and name so that the resulting weights can be consumed by the testing script.

`Testing_Yolov8n.py` runs inference with a YOLOv8 model over images or folders, writes annotated previews for quick inspection, and extracts detected license plate regions to a structured output directory. The resulting crops are organized per input image to make downstream OCR execution simple by pointing the OCR script to that folder structure.

# Inputs and outputs

The OCR scripts expect a folder of plate crops as input in common formats such as jpg, jpeg, png, bmp, webp, tif, or tiff. The output directory contains a CSV with one row per image that includes the predicted plate string, the seven characters at positions p0 through p6, the seven EasyOCR confidences at positions c0 through c6, and the seven raw characters before soft correction at positions raw0 through raw6. The scripts also write a per-image text file with the predicted plate, a text file with all predictions, a text file with the unique set of predictions, a folder of overlays with the original plate annotated with boxes and final text, and a folder of 64×64 crops for visual debugging. When evaluation is enabled, an additional metrics folder is created with textual summaries and figures.

# Options and configuration

The `--in_dir` argument selects the input folder and is scanned recursively. The `--out_dir` argument selects the destination for CSVs, overlays, crops, lists, and optional metrics. The `--gpu` flag enables GPU usage when supported by your environment. The `--gt_path` argument activates evaluation and accepts CSV or XLSX files. The `--gt_image_col` and `--gt_plate_col` arguments adapt the script to custom column names, and the `--gt_sheet` argument selects a specific sheet in Excel files. Defaults expect the column names image and plate, and the merge lowercases and strips file extensions on the image side to avoid mismatches.

# Practical notes

Results are best when the plate region is tightly cropped, sharp, and not heavily skewed. The method tolerates illumination gradients and removes the left EU band when necessary, but strong motion blur or decorative fonts will reduce segmentation quality. The seven boxes are ordered left to right and the prediction is constrained to four digits followed by three letters, which matches modern Spanish plates and prevents vowels, Q, or Ñ from appearing in letter positions. When no images are found the script still creates an empty CSV and list files so that downstream steps remain robust.

# Example commands

The first example runs OCR and writes results and artifacts to the output directory.

```bash
python OCR_Def.py --in_dir "./data/plates" --out_dir "./out/ocr"
```

The second example runs OCR and evaluation using a CSV ground truth with custom column names and writes a metrics summary, a confusion matrix in CSV and PNG, and the ROC/AUC plot.

```bash
python OCR_Def.py --in_dir "./data/plates" --out_dir "./out/ocr_eval" --gt_path "./groundtruth.csv" --gt_image_col "foto" --gt_plate_col "matricula"
```

The third example shows how to run YOLOv8 inference to crop plates before OCR.

```bash
python Testing_Yolov8n.py --model "/path/to/best.pt" --source "/path/to/images" --outdir "/path/to/outputs" --conf 0.50 --imgsz 1280
```

---

# Evaluation protocol (5-Fold)

We evaluate the pipeline on **35 annotated plate images** using **5-Fold cross-validation**. For each fold we report:

* **Plate-level exact match** (all 7 characters correct).
* **Character-level metrics** (Accuracy, Precision, Recall, F1 under micro and macro averaging).
* **Confusion matrix** (CSV + heatmap) and **ROC/AUC** from EasyOCR confidences.

Ground truth rows are normalized to **7 uppercase characters** and only valid targets are kept. Results and figures are written under `out_dir/metrics/` for full traceability.

# Results

**K-Fold — plate-level (exact match)**

* **Mean:** **60.0%**
* **Per fold (≈):** 43%, 58%, **86%**, 58%, 58%

**K-Fold — character-level (accuracy)**

* **Mean:** **79.6%**
* **Per fold (≈):** 80%, 71%, **95%**, 84%, 67%

**Aggregate character-level metrics**

* **Micro:** Accuracy = **79.6%**, Precision = **79.6%**, Recall = **79.6%**, F1 = **79.6%**
* **Macro:** Precision = **83.1%**, Recall = **75.0%**, F1 = **78.0%**

**ROC / confidence usefulness**

* Character-correct vs EasyOCR-confidence **AUC = 0.736**, showing the confidence score is informative and can be thresholded to trade precision/recall.

**Per-class behavior**

* Most digits/letters achieve **P/R/F1 ≥ 0.80**.
* Lower recall concentrates on **rare letters** (e.g., **P, V, Y**), as shown by the support histogram (some classes have <5 samples).
* Typical confusions: **0↔O**, **1↔I**, **5↔S**.

# Discussion

* The deterministic **contour-based segmentation** is robust under moderate illumination changes; plate-level failures usually stem from **one wrong character**, not from a complete pipeline breakdown.
* **Data imbalance** across letters is the main driver of macro-recall degradation.
* The **position-aware allowlists** (digits at 0–3, consonants at 4–6) effectively prevent invalid outputs and reduce substitution errors.

# Limitations and future work

* Sensitive to **strong blur**, **heavy skew**, and **specular reflections**.
* Imbalanced character distribution limits recall for rare letters.
* Next steps:

  * Grow and **rebalance** the dataset.
  * **Real-time streaming** + **parallelization** for speed.
  * Weather/lighting augmentation (rain, fog, glare).
  * Optional lightweight **CNN classifier per character** using the generated 64×64 crops.
  * Confidence-aware post-processing (e.g., human review when `min(c0..c6) < τ`).

# Reproducibility artifacts

Running the scripts produces:

* `predictions.csv` with `plate`, `p0..p6`, `c0..c6`, and `raw0..raw6`.
* `overlays/` (boxes + final text) and `crops64/` per character.
* `lists_all.txt` and `lists_unique.txt`.
* `metrics/` folder (classification report, confusion matrix, ROC/AUC, K-Fold summaries and plots).

All artifacts are versionable, enabling inspection and reporting.

# Authors

* **Marc Cases**
* **Álvaro Bello**
* **Adrián Fuster**
* **Namanmahi Kumar**

Universitat Autònoma de Barcelona — 2025

# Acknowledgements

* **Ultralytics YOLOv8** for detection
* **EasyOCR** for recognition
* **Roboflow** dataset for detector training

# How to cite

Suggested citation:  
Cases, M., Bello, Á., Fuster, A., & Kumar, N. (2025). *License Plate OCR (offline, character by character)*. GitHub repository. https://github.com/alvarobello15/Licence_Plate_Recognition



```
```

