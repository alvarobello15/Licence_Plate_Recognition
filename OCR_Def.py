# -*- coding: utf-8 -*-
"""
OCR_Def.py — Segmentación carácter a carácter + EasyOCR (local, sin cloud)

Pipeline:
  1) Binarización (texto blanco sobre negro) robusta a iluminación.
  2) Segmentación por contornos → 7 caracteres ordenados (izq→der).
  3) Normalización por recorte (64x64) manteniendo aspecto.
  4) Clasificación EasyOCR por carácter con allowlist posicional:
       - Posiciones 0..3: dígitos '0-9'
       - Posiciones 4..6: letras válidas ES 'BCDFGHJKLMNPRSTVWXYZ'
  5) Salidas:
       - results.csv (todas las lecturas)
       - all.txt / unique.txt
       - overlays/*.jpg (placa con cajas y texto)
       - crops/<img>/*.png (recortes normalizados)
       - <img>.txt (matrícula detectada)

Uso:
  python OCR_Def.py --in_dir "./outputs_engi/plates" --out_dir "./out/ocr_chars_easyocr_local" [--gpu]

Requisitos:
  pip install easyocr opencv-python-headless numpy pandas
  # y PyTorch (CPU o GPU) para EasyOCR
"""

import os
import re
import cv2
import glob
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import easyocr

# ----------------------------
# Parámetros y defaults
# ----------------------------
IN_DIR_DEFAULT  = "./outputs_engi/plates"
OUT_DIR_DEFAULT = "./out/ocr_chars_easyocr_local"

# Extensiones soportadas (recursivo)
IMG_GLOBS = (
    "*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff",
    "*.JPG","*.JPEG","*.PNG","*.BMP","*.WEBP","*.TIF","*.TIFF"
)

# Conjuntos permitidos
DIGITOS = "0123456789"
LETRAS  = "BCDFGHJKLMNPRSTVWXYZ"   # ES sin vocales ni Q/Ñ
CHAR_H, CHAR_W = 64, 64            # tamaño normalización de recortes

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
            cv2.putText(out, text[i], (x, max(0,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
    if text:
        cv2.putText(out, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2, cv2.LINE_AA)
    return out

# ----------------------------
# 1) Binarización
# ----------------------------
def binarizar(placa_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(placa_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=60, sigmaSpace=60)
    bg = cv2.GaussianBlur(g, (35,35), 0)
    norm = cv2.normalize((g.astype(np.float32)+1.0)/(bg.astype(np.float32)+1.0),
                         None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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
        if h < 0.30*H or h > 0.98*H:       continue
        if w < 0.012*W or w > 0.50*W:      continue
        if ar < 0.12   or ar > 1.35:       continue
        if area < 0.004*H*W:               continue
        if x+w > 0.99*W and w > 0.03*W:    continue  # borde derecho
        boxes.append((x,y,w,h))

    # fallback ligero si no detecta nada
    if not boxes:
        bw2 = cv2.dilate(bw, np.ones((3,2), np.uint8), 1)
        cnts, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h + 1e-6)
            area = w * h
            if h < 0.30*H or h > 0.98*H:       continue
            if w < 0.012*W or w > 0.50*W:      continue
            if ar < 0.12   or ar > 1.35:       continue
            if area < 0.004*H*W:               continue
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
    roi = bw[y:y+h, x:x+w]                # blanco=texto
    p = max(1, int(0.08*max(w,h)))        # padding proporcional
    roi = cv2.copyMakeBorder(roi, p,p,p,p, cv2.BORDER_CONSTANT, value=0)
    H, W = roi.shape
    scale = min(CHAR_H/float(H), CHAR_W/float(W))
    nh, nw = max(1, int(round(H*scale))), max(1, int(round(W*scale)))
    rs = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CHAR_H, CHAR_W), dtype=np.uint8)
    y0 = (CHAR_H - nh)//2
    x0 = (CHAR_W - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = rs
    # EasyOCR acepta BGR; convertimos
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

# ----------------------------
# 4) Clasificación EasyOCR carácter a carácter
# ----------------------------
def clasificar_chars(reader: easyocr.Reader, char_imgs: List[np.ndarray]) -> str:
    pred = []
    for i, chimg in enumerate(char_imgs):
        allow = DIGITOS if i < 4 else LETRAS
        # detail=0 → solo texto; restrictivo con allowlist
        txts = reader.readtext(
            chimg, detail=0, paragraph=False, allowlist=allow,
            text_threshold=0.5, low_text=0.3, link_threshold=0.3, mag_ratio=2.0,
            add_margin=0.1, width_ths=0.8, decoder='greedy'
        )
        ch = (txts[0] if len(txts)>0 else "")
        ch = (ch or "").strip().upper()
        ch = ch[:1] if ch else ""
        if not ch:
            ch = allow[0] if allow else "?"
        pred.append(ch or "?")

    # Correcciones suaves por grupo
    for i in range(7):
        if i < 4 and pred[i]:
            pred[i] = pred[i].translate(MAPA_DIG)
        elif i >= 4 and pred[i]:
            pred[i] = pred[i].translate(MAPA_LET)
    return "".join(pred)

# ----------------------------
# 5) Proceso por imagen
# ----------------------------
def process_image(reader, img_path: str, out_dir: str) -> Tuple[str, str]:
    name = os.path.splitext(os.path.basename(img_path))[0]
    img  = cv2.imread(img_path)
    if img is None:
        return name, ""

    bw    = binarizar(img)
    boxes = segmentar_chars(bw)

    if len(boxes) < 6:
        matricula = ""
    else:
        boxes = sorted(boxes, key=lambda b:b[0])
        if len(boxes) > 7:
            boxes = boxes[:7]
        char_imgs = [recorte_normalizado(bw, b) for b in boxes]
        raw = clasificar_chars(reader, char_imgs)

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

    return name, matricula

# ----------------------------
# 6) Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  type=str, default=IN_DIR_DEFAULT, help="Carpeta con imágenes de placas recortadas")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT, help="Carpeta de salida")
    ap.add_argument("--gpu",     action="store_true", help="Usar GPU si está disponible")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

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
    for p in paths:
        name, plate = process_image(reader, p, args.out_dir)
        rows.append({"image": name, "plate": plate})

    # CSV global + listas
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False, encoding="utf-8")
    all_txt  = os.path.join(args.out_dir, "all.txt")
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

if __name__ == "__main__":
    main()
