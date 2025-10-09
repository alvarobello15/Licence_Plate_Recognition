# -*- coding: utf-8 -*-
"""
OCR_Def.py — Segmentación carácter a carácter + EasyOCR (local, sin cloud)
+ Evaluación con ground truth (CSV/XLSX): matriz de confusión, F1, accuracy, precision, recall y curva ROC/AUC.

Pipeline:
  1) Binarización (texto blanco sobre negro) robusta a iluminación.
  2) Segmentación por contornos → 7 caracteres ordenados (izq→der).
  3) Normalización por recorte (64x64) manteniendo aspecto.
  4) Clasificación EasyOCR por carácter con allowlist posicional:
       - Posiciones 0..3: dígitos '0-9'
       - Posiciones 4..6: letras válidas ES 'BCDFGHJKLMNPRSTVWXYZ'
     (guardamos también la CONFIDENCIA por carácter)
  5) Salidas OCR:
       - results.csv (predicciones; incluye p0..p6 y c0..c6)
       - all.txt / unique.txt
       - overlays/*.jpg (placa con cajas y texto)
       - crops/<img>/*.png (recortes normalizados)
       - <img>.txt (matrícula detectada)
  6) Si se pasa ground truth (CSV/XLSX), se generan métricas en out_dir/metrics:
       - metrics.txt (resumen)
       - confusion_matrix.csv
       - cm.png (heatmap)
       - roc_auc.png (curva ROC del criterio “carácter correcto” vs. confianza de EasyOCR)

Uso:
  python OCR_Def.py --in_dir "./outputs_engi/plates" --out_dir "./out/ocr_chars_easyocr_local" \
                    [--gpu] \
                    [--gt_path "./groundtruth.csv" --gt_image_col "foto" --gt_plate_col "matricula" --gt_sheet "Sheet1"]

Requisitos:
  pip install easyocr opencv-python-headless numpy pandas scikit-learn matplotlib
  + PyTorch (CPU o GPU) para EasyOCR
"""
import os, re, cv2, glob, time, argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

import easyocr

# Métricas y plots
import matplotlib
matplotlib.use("Agg")  # backend sin GUI para guardar PNGs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score, roc_curve, auc

# ----------------------------
# Parámetros y defaults
# ----------------------------
IN_DIR_DEFAULT = "./outputs_engi/plates"
OUT_DIR_DEFAULT = "./out/ocr_chars_easyocr_local"

IMG_GLOBS = (
    "*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff",
    "*.JPG","*.JPEG","*.PNG","*.BMP","*.WEBP","*.TIF","*.TIFF"
)

DIGITOS = "0123456789"
LETRAS  = "BCDFGHJKLMNPRSTVWXYZ"  # ES sin vocales ni Q/Ñ
CHAR_H, CHAR_W = 64, 64

# Correcciones suaves habituales
MAPA_DIG = str.maketrans({'D':'0','Q':'0','O':'0','U':'0','L':'1','I':'1','T':'7','Z':'2','S':'5','B':'8','G':'6'})
MAPA_LET = str.maketrans({'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B','7':'T'})

# ----------------------------
# Utilidades de E/S
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def draw_overlay(img: np.ndarray, boxes: List[Tuple[int,int,int,int]], text: str) -> np.ndarray:
    out = img.copy()
    for i, (x,y,w,h) in enumerate(boxes):
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        if i < len(text) and text[i]:
            cv2.putText(out, text[i], (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
    if text:
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)
    return out

# ----------------------------
# 1) Binarización
# ----------------------------
def binarizar(placa_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(placa_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=60, sigmaSpace=60)
    bg = cv2.GaussianBlur(g, (35,35), 0)
    norm = cv2.normalize((g.astype(np.float32)+1.0)/(bg.astype(np.float32)+1.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 3)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    return bw  # texto blanco (255), fondo negro (0)

# ----------------------------
# 2) Segmentación por contornos → cajas de chars
# ----------------------------
def segmentar_chars(bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    H, W = bw.shape

    # Posible franja azul "EU" a la izquierda: si hay mucha tinta, bórrala
    left_band = int(0.10 * W)
    if left_band > 0 and (bw[:, :left_band] > 0).sum() > 0.03 * H * left_band:
        bw[:, :left_band] = 0

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-6)
        area = w * h
        if h < 0.30*H or h > 0.98*H:    continue
        if w < 0.012*W or w > 0.50*W:   continue
        if ar < 0.12 or ar > 1.35:      continue
        if area < 0.004*H*W:            continue
        if x+w > 0.99*W and w > 0.03*W: continue  # borde derecho
        boxes.append((x,y,w,h))

    # Fallback ligero si no detecta nada
    if not boxes:
        bw2 = cv2.dilate(bw, np.ones((3,2), np.uint8), 1)
        cnts, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h + 1e-6)
            area = w * h
            if h < 0.30*H or h > 0.98*H:    continue
            if w < 0.012*W or w > 0.50*W:   continue
            if ar < 0.12 or ar > 1.35:      continue
            if area < 0.004*H*W:            continue
            boxes.append((x,y,w,h))

    if not boxes:
        return []

    boxes.sort(key=lambda b: b[0])

    # Si hay más de 7, quedarnos con los 7 más "coherentes" en altura
    if len(boxes) > 7:
        hs = np.array([h for (_,_,_,h) in boxes], dtype=np.float32)
        med = float(np.median(hs))
        scores = [ -abs(h - med) for (_,_,_,h) in boxes ]
        idx = np.argsort(scores)[-7:]
        boxes = [boxes[i] for i in sorted(idx, key=lambda k: boxes[k][0])]

    # Si hay menos de 7, intentar partir cajas anchas (dobles)
    if len(boxes) < 7:
        boxes = _split_cajas_anchas(bw, boxes)

    if len(boxes) > 7:
        boxes = boxes[:7]

    return boxes

def _split_cajas_anchas(bw: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    out = []
    for (x,y,w,h) in sorted(boxes, key=lambda b:b[0]):
        if w > 1.5*h:
            roi = bw[y:y+h, x:x+w]
            col_sum = (roi > 0).sum(axis=0)
            if col_sum.size < 6:
                out.append((x,y,w,h)); continue
            m = h//6
            inner = col_sum[m:-m] if (w-2*m)>3 else col_sum
            cut = int(np.argmin(inner)) + (m if (w-2*m)>3 else 0)
            cut = max(2, min(w-2, cut))
            out.append((x,y,cut,h))
            out.append((x+cut,y,w-cut,h))
        else:
            out.append((x,y,w,h))
    out.sort(key=lambda b:b[0])
    return out

# ----------------------------
# 3) Normalización de recortes
# ----------------------------
def recorte_normalizado(bw: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,w,h = box
    roi = bw[y:y+h, x:x+w]  # blanco=texto
    p = max(1, int(0.08*max(w,h)))  # padding proporcional
    roi = cv2.copyMakeBorder(roi, p,p,p,p, cv2.BORDER_CONSTANT, value=0)
    H, W = roi.shape
    scale = min(CHAR_H/float(H), CHAR_W/float(W))
    nh, nw = max(1, int(round(H*scale))), max(1, int(round(W*scale)))
    rs = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CHAR_H, CHAR_W), dtype=np.uint8)
    y0 = (CHAR_H - nh)//2
    x0 = (CHAR_W - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = rs
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)  # EasyOCR espera BGR

# ----------------------------
# 4) Clasificación EasyOCR carácter a carácter (con confidencias)
# ----------------------------
def clasificar_chars(reader: easyocr.Reader, char_imgs: List[np.ndarray]) -> Tuple[str, List[float], List[str]]:
    """
    Devuelve:
      - pred_str: string con hasta 7 caracteres tras correcciones suaves posicionales
      - confs: lista de confidencias (len <= 7), una por carácter elegido
      - raw_chars: lista de caracteres crudos (sin corrección) detectados por EasyOCR
    """
    raw_chars: List[str] = []
    confs: List[float] = []

    for i, chimg in enumerate(char_imgs[:7]):
        allow = DIGITOS if i < 4 else LETRAS
        # detail=1 ⇒ [(bbox, text, conf), ...]
        res = reader.readtext(
            chimg,
            detail=1,
            paragraph=False,
            allowlist=allow,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.3,
            mag_ratio=2.0,
            add_margin=0.1,
            width_ths=0.8,
            decoder='greedy'
        )

        if res:
            # Escogemos la hipótesis con mayor confianza
            best = max(res, key=lambda r: (r[2] if len(r) > 2 and r[2] is not None else 0.0))
            txt = (best[1] or "").strip().upper()
            conf = float(best[2]) if len(best) > 2 and best[2] is not None else 0.0
            ch = txt[:1] if txt else ""
        else:
            ch, conf = "", 0.0

        raw_chars.append(ch)
        confs.append(conf)

    # Relleno hasta 7 para evitar IndexError aguas abajo
    if len(raw_chars) < 7:
        raw_chars += [""] * (7 - len(raw_chars))
        confs     += [0.0] * (7 - len(confs))

    # Correcciones posicionales suaves (sin tocar raw_chars)
    pred_chars: List[str] = []
    for i, ch in enumerate(raw_chars[:7]):
        if not ch:
            pred_chars.append("")
            continue
        if i < 4:
            pred_chars.append(ch.translate(MAPA_DIG))
        else:
            pred_chars.append(ch.translate(MAPA_LET))

    pred_str = "".join(pred_chars[:7])
    return pred_str, confs[:7], raw_chars[:7]


# ----------------------------
# 5) Proceso por imagen
# ----------------------------
def process_image(reader, img_path: str, out_dir: str) -> Tuple[str, str, List[float], List[str]]:
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        return name, "", [0.0]*7, [""]*7

    bw = binarizar(img)
    boxes = segmentar_chars(bw)

    if len(boxes) < 6:
        matricula, confs, raw_chars = "", [0.0]*7, [""]*7
        char_imgs = []
    else:
        boxes = sorted(boxes, key=lambda b:b[0])
        if len(boxes) > 7:
            boxes = boxes[:7]
        char_imgs = [recorte_normalizado(bw, b) for b in boxes]
        raw, confs, raw_chars = clasificar_chars(reader, char_imgs)

        # Forzar patrón 4+3
        digs = ''.join(ch for ch in raw[:4] if ch in DIGITOS).ljust(4, '?')
        lets = ''.join(ch for ch in raw[4:7] if ch in LETRAS).ljust(3, '?')
        matricula = digs + lets

    # Guardar crops
    crop_dir = os.path.join(out_dir, "crops", name)
    ensure_dir(crop_dir)
    for i, ch in enumerate(char_imgs):
        cv2.imwrite(os.path.join(crop_dir, f"{i}_{(matricula[i] if i < len(matricula) else '?')}.png"), ch)

    # Guardar overlay y txt
    ov = draw_overlay(img, boxes, matricula if matricula else "NO_READ")
    ensure_dir(os.path.join(out_dir, "overlays"))
    cv2.imwrite(os.path.join(out_dir, "overlays", f"{name}.jpg"), ov)
    with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write((matricula or "") + "\n")

    return name, matricula, confs, raw_chars

# ----------------------------
# 6) Evaluación con ground truth (CSV/XLSX)
# ----------------------------
def _read_groundtruth(gt_path: str, gt_sheet: str = None) -> pd.DataFrame:
    ext = os.path.splitext(gt_path)[1].lower()
    if ext == ".csv":
        # Lee CSV con separador coma por defecto; ajusta si fuera ; con sep=";"
        return pd.read_csv(gt_path)
    else:
        # XLSX / XLS
        try:
            return pd.read_excel(gt_path, sheet_name=gt_sheet) if gt_sheet else pd.read_excel(gt_path)
        except Exception as e:
            raise RuntimeError(f"No se pudo leer el ground truth '{gt_path}': {e}")

def evaluar_con_groundtruth(out_dir: str,
                            results_df: pd.DataFrame,
                            gt_path: str,
                            gt_image_col: str = "image",
                            gt_plate_col: str = "plate",
                            gt_sheet: str = None):
    """
    Lee un CSV/XLSX de ground truth y genera métricas en out_dir/metrics:
      - metrics.txt con accuracy exact-match (placa), y métricas char-level (accuracy, P/R/F1 micro y macro)
      - confusion_matrix.csv y cm.png (char-level)
      - roc_auc.png (ROC de 'carácter correcto' usando la confianza de EasyOCR por carácter)

    Acepta GT en dos formatos:
      (A) Dos columnas separadas (p.ej. 'foto' y 'matricula'), indicadas con --gt_image_col y --gt_plate_col
      (B) Una única columna con el texto "foto,matricula" en cada fila (auto-split)
    Además, se limpia la 'foto' para quedarse con el basename sin extensión (y se pasa a minúsculas).
    """
    metrics_dir = os.path.join(out_dir, "metrics")
    ensure_dir(metrics_dir)

    # --- Leer GT según extensión ---
    gt = _read_groundtruth(gt_path, gt_sheet)

    # --- Renombrar columnas si existen con los nombres proporcionados ---
    rename_map = {}
    if gt_image_col in gt.columns:
        rename_map[gt_image_col] = "image"
    if gt_plate_col in gt.columns:
        rename_map[gt_plate_col] = "gt_plate"
    if rename_map:
        gt = gt.rename(columns=rename_map)

    # --- Normalizaciones y casos especiales ---
    # 1) Si NO tenemos ambas columnas, intentar auto-split si hay una sola con "foto,matricula"
    if not {"image", "gt_plate"}.issubset(gt.columns):
        if gt.shape[1] == 1:
            unico = gt.columns[0]
            # split por primera coma
            spl = gt[unico].astype(str).str.split(",", n=1, expand=True)
            if spl.shape[1] == 2:
                gt["image"] = spl[0].astype(str)
                gt["gt_plate"] = spl[1].astype(str)
            else:
                raise ValueError(
                    "No se encontraron columnas 'image' y 'gt_plate' ni se pudo dividir por coma una única columna."
                )
        else:
            # Intento adicional: buscar alguna columna con coma en la mayoría de filas y dividirla
            found = False
            for col in gt.columns:
                series = gt[col].astype(str)
                if series.str.contains(",").mean() > 0.5:  # heurística
                    spl = series.str.split(",", n=1, expand=True)
                    if spl.shape[1] == 2:
                        gt["image"] = spl[0]
                        gt["gt_plate"] = spl[1]
                        found = True
                        break
            if not found:
                raise ValueError(
                    "No se pudo identificar ground truth. Especifica --gt_image_col y --gt_plate_col "
                    "o proporciona una sola columna con formato 'foto,matricula'."
                )

    # 2) Normalizar strings, quitar comillas, rutas y extensiones
    gt["image"] = (
        gt["image"]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .apply(lambda s: os.path.splitext(os.path.basename(s))[0])
        .str.lower()
    )
    # Normaliza matrícula: quitar no alfanuméricos y pasar a mayúsculas
    gt["gt_plate"] = (
        gt["gt_plate"]
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    # 3) Filtrar filas con GT de longitud 7
    gt = gt[gt["gt_plate"].str.len() == 7].copy()
    if len(gt) == 0:
        with open(os.path.join(metrics_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write("No hay filas válidas en el ground truth (placas de longitud 7 tras normalización).\n")
        return

    # --- Fusionar con resultados (por basename en minúsculas) ---
    df = results_df.copy()
    df["image"] = df["image"].astype(str).str.lower()
    df = df.merge(gt[["image", "gt_plate"]], on="image", how="inner")

    if len(df) == 0:
        with open(os.path.join(metrics_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write("No hay intersección entre 'results.csv' y el ground truth por el campo 'image'.\n")
        return

    # --- Métricas de placa completa (exact match) ---
    df["pred_plate"] = df["plate"].fillna("").astype(str)
    exact_acc = (df["pred_plate"] == df["gt_plate"]).mean()

    # --- Preparación para métricas char-level ---
    y_true_chars, y_pred_chars = [], []
    y_score_conf = []  # confidencias por carácter (c0..c6) de EasyOCR

    for _, row in df.iterrows():
        gt_plate = (row["gt_plate"] + "???????")[:7]
        pred_plate = (row["pred_plate"] + "???????")[:7]

        confs = []
        for i in range(7):
            key = f"c{i}"
            confs.append(float(row[key]) if key in row and pd.notnull(row[key]) else 0.0)

        for i in range(7):
            y_true_chars.append(gt_plate[i])
            y_pred_chars.append(pred_plate[i])
            y_score_conf.append(confs[i])

    # --- Matriz de confusión char-level ---
    clases = sorted(list(set(list("".join(y_true_chars + y_pred_chars)))))
    cm = confusion_matrix(y_true_chars, y_pred_chars, labels=clases)
    cm_df = pd.DataFrame(cm, index=clases, columns=clases)
    cm_df.to_csv(os.path.join(metrics_dir, "confusion_matrix.csv"), encoding="utf-8")

    # --- Métricas char-level ---
    acc_char = accuracy_score(y_true_chars, y_pred_chars)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_chars, y_pred_chars, average='micro', zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_chars, y_pred_chars, average='macro', zero_division=0
    )
    cls_report = classification_report(y_true_chars, y_pred_chars, zero_division=0)

    # --- ROC & AUC (binario: 'carácter correcto' usando confianza del char elegido) ---
    y_bin = [1 if t == p else 0 for t, p in zip(y_true_chars, y_pred_chars)]
    try:
        fpr, tpr, _ = roc_curve(y_bin, y_score_conf)
        auc_val = auc(fpr, tpr)
    except Exception:
        fpr, tpr, auc_val = [0, 1], [0, 1], float('nan')

    # --- Guardar resumen de métricas ---
    with open(os.path.join(metrics_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("=== MÉTRICAS DE OCR DE MATRÍCULAS ===\n\n")
        f.write(f"Total imágenes evaluadas: {len(df)}\n")
        f.write(f"Accuracy exact-match (placa completa): {exact_acc:.4f}\n\n")
        f.write("— Métricas a nivel CARÁCTER —\n")
        f.write(f"Accuracy (char): {acc_char:.4f}\n")
        f.write(f"Precision (micro): {p_micro:.4f}\n")
        f.write(f"Recall    (micro): {r_micro:.4f}\n")
        f.write(f"F1        (micro): {f1_micro:.4f}\n")
        f.write(f"Precision (macro): {p_macro:.4f}\n")
        f.write(f"Recall    (macro): {r_macro:.4f}\n")
        f.write(f"F1        (macro): {f1_macro:.4f}\n\n")
        f.write("— Classification report (char-level) —\n")
        f.write(cls_report + "\n")
        f.write("— ROC/AUC (detección 'carácter correcto' usando la confianza de EasyOCR) —\n")
        f.write(f"AUC: {auc_val:.4f}\n")

    # --- Plot: Matriz de confusión (heatmap simple) ---
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Matriz de confusión (char-level)")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.xticks(ticks=np.arange(len(clases)), labels=clases, rotation=90)
    plt.yticks(ticks=np.arange(len(clases)), labels=clases)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "cm.png"), dpi=200)
    plt.close()

    # --- Plot: ROC ---
    try:
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC — Carácter correcto vs. confianza EasyOCR")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, "roc_auc.png"), dpi=200)
        plt.close()
    except Exception:
        pass

# ----------------------------
# 7) Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default=IN_DIR_DEFAULT, help="Carpeta con imágenes de placas recortadas")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT, help="Carpeta de salida")
    ap.add_argument("--gpu", action="store_true", help="Usar GPU si está disponible")

    # Ground truth (unificado): usa --gt_path y detecta por extensión (.csv/.xlsx)
    ap.add_argument("--gt_path", type=str, default=None, help="Ruta al CSV/XLSX con ground truth")
    # Compatibilidad retro con flags antiguos (opcionales)
    ap.add_argument("--gt_xlsx", type=str, default=None, help="[DEPRECADO] Ruta al Excel con ground truth")
    ap.add_argument("--gt_csv",  type=str, default=None, help="[DEPRECADO] Ruta al CSV con ground truth")
    ap.add_argument("--gt_image_col", type=str, default="image", help="Columna en GT con el identificador de imagen (basename sin extensión o con extensión)")
    ap.add_argument("--gt_plate_col", type=str, default="plate", help="Columna en GT con la matrícula")
    ap.add_argument("--gt_sheet", type=str, default=None, help="Nombre de la hoja (solo si es Excel)")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # Resolver ruta GT priorizando --gt_path; luego --gt_csv / --gt_xlsx
    gt_path = args.gt_path or args.gt_csv or args.gt_xlsx

    # Inicializar EasyOCR (inglés/latin es suficiente para dígitos y mayúsculas)
    reader = easyocr.Reader(['en'], gpu=args.gpu, verbose=False)

    # Buscar imágenes (recursivo en subcarpetas)
    base = Path(args.in_dir)
    paths: List[str] = []
    for pat in IMG_GLOBS:
        paths.extend(str(p) for p in base.rglob(pat))
    paths = sorted(set(paths))

    if not paths:
        print(f"[WARN] No se encontraron imágenes en {base.resolve()}")
        print("Verifica la ruta, extensiones y si las imágenes están en subcarpetas.")
        # Aún así generamos CSV vacío y archivos de lista
        pd.DataFrame([], columns=["image","plate"]).to_csv(os.path.join(args.out_dir, "results.csv"), index=False, encoding="utf-8")
        open(os.path.join(args.out_dir, "all.txt"), "w", encoding="utf-8").close()
        open(os.path.join(args.out_dir, "unique.txt"), "w", encoding="utf-8").close()
        print(f"[OK] Imágenes procesadas: 0")
        print(f"[OK] Resultados: {os.path.abspath(args.out_dir)}")
        print(" - results.csv\n - all.txt / unique.txt\n - overlays/*.jpg\n - crops/<img>/*.png\n - <img>.txt")
        return

    rows = []
    t0 = time.time()
    print(f"[INFO] Procesando {len(paths)} imágenes. Salida: {args.out_dir}")

    for p in paths:
        name, plate, confs, raw_chars = process_image(reader, p, args.out_dir)
        row = {"image": name, "plate": plate}
        # guardar por carácter (pred y conf)
        for i in range(7):
            row[f"p{i}"] = plate[i] if len(plate) == 7 else ""
            row[f"c{i}"] = float(confs[i]) if i < len(confs) else 0.0
            row[f"raw{i}"] = raw_chars[i] if i < len(raw_chars) else ""
        rows.append(row)

    # CSV global + listas
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False, encoding="utf-8")

    all_txt = os.path.join(args.out_dir, "all.txt")
    uniq_txt = os.path.join(args.out_dir, "unique.txt")
    with open(all_txt, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((r["plate"] or "") + "\n")
    uniques = sorted(set(r["plate"] for r in rows if r["plate"]))
    with open(uniq_txt, "w", encoding="utf-8") as f:
        for u in uniques:
            f.write(u + "\n")

    dt = time.time() - t0
    print(f"[OK] Imágenes procesadas: {len(paths)} en {dt:.2f}s")
    print(f"[OK] Resultados: {os.path.abspath(args.out_dir)}")
    print(" - results.csv\n - all.txt / unique.txt\n - overlays/*.jpg\n - crops/<img>/*.png\n - <img>.txt")

    # Evaluación si hay GT
    if gt_path:
        try:
            evaluar_con_groundtruth(
                out_dir=args.out_dir,
                results_df=df,
                gt_path=gt_path,
                gt_image_col=args.gt_image_col,
                gt_plate_col=args.gt_plate_col,
                gt_sheet=args.gt_sheet
            )
            print("[OK] Métricas guardadas en:", os.path.join(args.out_dir, "metrics"))
        except Exception as e:
            print("[WARN] No se pudo evaluar con GT:", e)

if __name__ == "__main__":
    main()
