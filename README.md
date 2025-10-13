License Plate OCR (offline, character by character)

This repository contains a compact offline tool to read modern Spanish and EU license plates in the DDDDLLL pattern from already cropped plate images. The script performs robust binarization, segments the plate into seven characters from left to right, normalizes each character to a fixed size, and recognizes them individually using EasyOCR with position-aware constraints. It saves clear artifacts for inspection and, when a ground truth file is supplied, it reports reproducible metrics such as exact match for the full plate, character-level scores, a confusion matrix, and ROC/AUC derived from per-character confidences.

How it works

The pipeline begins by converting the plate region to grayscale and applying background normalization so that the text becomes white on black even under uneven lighting. The method removes the typical EU blue strip at the left when it contains significant ink, finds external contours, filters them using simple geometric constraints, and orders the resulting boxes from left to right. When fewer than seven boxes are found, the method attempts to split overly wide boxes, and when more are found, it keeps the seven most consistent in height. Each character is padded and resized to 64×64 pixels while preserving aspect ratio, then passed to EasyOCR one by one. Positions zero to three accept only digits and positions four to six accept only the consonants used in Spanish plates, which reduces substitutions. A soft correction step maps common confusions such as O with 0 or I with 1 without altering the raw OCR outputs. The final string is forced to four digits followed by three letters and is written alongside overlays and structured files for later analysis.

Requirements and installation

The tool targets Python 3.8 or newer. It relies on EasyOCR with PyTorch, OpenCV, NumPy, Pandas, scikit-learn, and Matplotlib. Install the Python dependencies with the following command and ensure that PyTorch is available in CPU or GPU form according to your system.
