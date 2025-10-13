# License Plate OCR (offline, character by character)

This repository contains a compact offline tool to read modern Spanish and EU license plates in the DDDDLLL pattern from already cropped plate images. The script performs robust binarization, segments the plate into seven characters from left to right, normalizes each character to a fixed size, and recognizes them individually using EasyOCR with position-aware constraints. It saves clear artifacts for inspection and, when a ground truth file is supplied, it reports reproducible metrics such as exact match for the full plate, character-level scores, a confusion matrix, and ROC/AUC derived from per-character confidences.

# How it works

The pipeline begins by converting the plate region to grayscale and applying background normalization so that the text becomes white on black even under uneven lighting. The method removes the typical EU blue strip at the left when it contains significant ink, finds external contours, filters them using simple geometric constraints, and orders the resulting boxes from left to right. When fewer than seven boxes are found, the method attempts to split overly wide boxes, and when more are found, it keeps the seven most consistent in height. Each character is padded and resized to 64×64 pixels while preserving aspect ratio, then passed to EasyOCR one by one. Positions zero to three accept only digits and positions four to six accept only the consonants used in Spanish plates, which reduces substitutions. A soft correction step maps common confusions such as O with 0 or I with 1 without altering the raw OCR outputs. The final string is forced to four digits followed by three letters and is written alongside overlays and structured files for later analysis.

# Requirements and installation

The tool targets Python 3.8 or newer. It relies on EasyOCR with PyTorch, OpenCV, NumPy, Pandas, scikit-learn, and Matplotlib. Install the Python dependencies with the following command and ensure that PyTorch is available in CPU or GPU form according to your system.

```bash
pip install easyocr opencv-python-headless numpy pandas scikit-learn matplotlib
```

# Running the script

A typical run only requires an input directory with plate images and an output directory. The script scans subfolders recursively and processes common raster formats.

```bash
python OCR_Def.py --in_dir "./outputs_engi/plates" --out_dir "./out/ocr_chars_easyocr_local"
```

If your machine supports GPU acceleration through PyTorch and EasyOCR, you can enable it.

```bash
python OCR_Def.py --in_dir "./my_plates" --out_dir "./out/ocr" --gpu
```

# Inputs and outputs

The input is a folder with cropped plate images in jpg, jpeg, png, bmp, webp, tif, or tiff, with any letter casing in file extensions. The output directory contains a CSV file with one row per image that includes the predicted plate string, the seven characters at positions p0 through p6, the seven EasyOCR confidences at positions c0 through c6, and the seven raw characters before soft correction at positions raw0 through raw6. The script writes a per-image text file with the predicted plate, a file with all predictions, a file with the unique set of predictions, an overlays folder that shows the original image annotated with boxes and the final text, and a crops folder with normalized 64×64 character images that simplify visual debugging.

# Ground truth and evaluation

When a CSV or Excel ground truth file is provided, the script merges results by the base name of the image and normalizes the target plate to seven uppercase alphanumeric characters. The file can contain two explicit columns for the image name and the plate, or a single column with the pattern image,plate per row that the script will split automatically. After the merge, only rows that result in a seven-character plate are considered for metrics. The evaluation reports exact match for the full plate, character-level accuracy together with precision, recall, and F1 under micro and macro averaging, a confusion matrix in CSV and as a heatmap image, and a ROC curve with its AUC computed from the per-character OCR confidence as the score for the event character predicted equals character true.

# Options that matter

The in_dir argument selects the input folder and is scanned recursively. The out_dir argument selects the destination for CSV files, overlays, crops, lists, and optional metrics. The gpu flag enables GPU usage when supported by your environment. The gt_path argument activates evaluation and accepts CSV or XLSX files. The gt_image_col and gt_plate_col arguments let you adapt to your column names, and the gt_sheet argument targets a specific sheet in Excel files. By default the script expects image and plate as column names and will lowercase and strip file extensions on the image side during the merge.

# Practical notes

Results are best when the plate region is tightly cropped, sharp, and not heavily skewed. The method tolerates illumination gradients and attempts to ignore the left EU band when necessary, but strong motion blur or decorative fonts degrade segmentation quality. The seven boxes are ordered from left to right and the prediction is constrained to four digits followed by three letters, which mirrors modern Spanish plates and prevents vowels, Q, or Ñ from appearing in the letter positions. If no images are found, the script still creates an empty CSV and the list files to keep downstream steps robust.

# Known limitations

The current implementation assumes the DDDDLLL pattern and a restricted letter set that excludes vowels, Q, and Ñ. It does not correct heavy perspective distortion and does not search for a plate inside a larger scene because it expects an already cropped plate. Plates with severe occlusion, weathering, or custom fonts may require tuning segmentation thresholds or adding an external deskew step before processing.

# Example commands

The following example runs the OCR and writes all artifacts in the chosen output directory.

```bash
python OCR_Def.py --in_dir "./data/plates" --out_dir "./out/ocr"
```

The next example runs OCR and evaluation using a CSV ground truth with custom column names and writes the metrics summary, the confusion matrix in CSV and PNG, and the ROC/AUC plot.

```bash
python OCR_Def.py --in_dir "./data/plates" --out_dir "./out/ocr_eval" --gt_path "./groundtruth.csv" --gt_image_col "foto" --gt_plate_col "matricula"
```

# File conventions

The results.csv file includes the columns image and plate, the per-position predictions p0 to p6, the per-position confidences c0 to c6, and the raw EasyOCR outputs raw0 to raw6. The overlays directory contains one image per input with detected boxes and the final text rendered on top. The crops directory contains normalized character images named with their position and the chosen character, which helps manual audits and quick error analysis.

# Roadmap

Future work may include optional deskew and geometric normalization, a simple HTML report for batch browsing, configurable character sets for other national formats, and lightweight ensembling across multiple OCR passes to stabilize low-confidence positions.

# License

Include the license that best fits your project. MIT, Apache-2.0, or GPL-3.0 are common choices for utilities of this kind.
