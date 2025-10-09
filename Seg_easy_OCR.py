# -*- coding: utf-8 -*-
"""
OCR de matrículas españolas (4 dígitos + 3 letras) con EasyOCR
- Robustez: rotaciones leves, postproceso con regex y corrección de confusiones 0/O, 1/I/L, 5/S, 6/G, 8/B, 7/T...
- Sin segmentar caracteres: se reconoce el texto completo y se valida el formato oficial.
- Salidas compatibles: CSV, lista completa y únicas, overlay por imagen, txt por imagen.

Requisitos:
    pip install easyocr opencv-python-headless numpy pandas

Uso típico:
    python ocr_placas_easyocr.py \
        --in_dir ./outputs_engi/plates \
        --out_dir ./out/ocr_chars_eocr \
        --gpu

Notas:
- El conjunto de letras válidas (sin vocales y sin Q/Ñ) se fuerza en el postproceso.
- Si EasyOCR devuelve ruido, el postproceso intenta reconstruir el patrón 4+3.
- Mucho más fiable que prototipos HOG sintéticos dibujados con cv2.putText.
"""

import os, glob, argparse, shutil, time, re
import cv2
import numpy as np
import pandas as pd
from collections import Counter

try:
    import torch
except Exception:
    torch = None

import easyocr

# =========================
# RUTAS (por defecto)
# =========================
IN_DIR     = "./outputs_engi/plates"
OUT_DIR    = "./out/ocr_chars_eocr"

# =========================
# Configuración
# =========================
AN, AL = 44, 64  # no se usan para OCR, solo si deseamos normalizar en algún debug
DIGITS  = "0123456789"
# Letras válidas en matrículas ES modernas (sin vocales, sin Ñ, sin Q)
LETTERS = "BCDFGHJKLMNPRSTVWXYZ"
ALPHANUM_ALLOW = DIGITS + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLATE_REGEX = re.compile(r"(\d{4})([A-Z]{3})")

# Mapas de confusión 
TO_DIGIT = str.maketrans({
    'O':'0','Q':'0','D':'0','U':'0',
    'I':'1','L':'1','T':'7','Z':'2','S':'5','B':'8','G':'6'
})
TO_LETTER = str.maketrans({
    '0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B','7':'T'
})

# =========================
# Utilidades
# =========================

def reset_output_dir(out_dir: str):
    out_dir = os.path.abspath(out_dir)
    forbidden = {"/", os.path.expanduser("~"), "C:\\", "C:\\Windows", "C:\\Users", "C:\\Program Files"}
    if out_dir in forbidden or len(out_dir) < 5:
        raise ValueError(f"[ABORT] Ruta de salida peligrosa: {out_dir}")
    if os.path.isdir(out_dir):
        print(f"[INFO] Limpiando salida: {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def rotate_bound(image, angle):
    # Rotación que conserva todo el contenido
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_plate(bgr: np.ndarray) -> np.ndarray:
    # Limpiado suave: CLAHE + bilateral para reducir reflejos conservando bordes
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=40, sigmaSpace=40)
    return g


def normalize_candidate_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def best_plate_from_string(s: str) -> str:
    """Devuelve el texto en formato 'dddd LLL' si encuentra patrón válido, si no, ''"""
    s = normalize_candidate_text(s)
    if len(s) < 7:
        return ""

    # 1) Intenta regex directa 4+3
    m = PLATE_REGEX.search(s)
    if m:
        dddd, LLL = m.group(1), m.group(2)
        # fuerza que las tres letras pertenezcan al conjunto permitido
        LLL = ''.join([c for c in LLL if c in LETTERS])
        if len(LLL) == 3:
            return f"{dddd} {LLL}"

    # 2) Correcciones por confusión en lado izquierdo y derecho
    #    Probaremos todas las ventanas posibles de longitud 7 en s
    best = ""
    for i in range(0, len(s) - 6):
        chunk = s[i:i+7]
        left = chunk[:4].translate(TO_DIGIT)
        right = chunk[4:].translate(TO_LETTER)
        left = re.sub(r"[^0-9]", "", left)
        right = re.sub(r"[^A-Z]", "", right)
        # Filtra letras no permitidas
        right = ''.join([c for c in right if c in LETTERS])
        if len(left) == 4 and len(right) == 3:
            best = f"{left} {right}"
            break
    return best


def score_candidate(text: str) -> int:
    """Heurístico simple para desempatar (más largo y correcto == mejor)"""
    if not text:
        return -1
    # 7 caracteres buenos = 7, penaliza '?' o vacíos
    return sum(ch.isalnum() for ch in text)


def read_plate_easyocr(bgr: np.ndarray, reader: easyocr.Reader):
    
    # Preprocesado suave
    g = preprocess_plate(bgr)
    # Probamos leve tilt: -3..+3 grados
    angles = [0, -2.0, 2.0, -3.0, 3.0, -1.0, 1.0]

    best_text = ""
    best_score = -1
    best_overlay = bgr.copy()

    for ang in angles:
        img_rot = rotate_bound(bgr, ang) if abs(ang) > 0.1 else bgr
        # EasyOCR puede trabajar con RGB/BGR indistintamente
        res = reader.readtext(img_rot, detail=1, paragraph=True, allowlist=ALPHANUM_ALLOW)
        # Une tokens de izquierda a derecha
        if not res:
            continue
        # Orden por x medio del bbox y concatenar
        def _x_center(b):
            (tl, tr, br, bl) = b
            xs = [p[0] for p in [tl, tr, br, bl]]
            return sum(xs) / 4.0
        res_sorted = sorted(res, key=lambda r: _x_center(r[0]))
        joined = "".join([r[1] for r in res_sorted])
        cand = best_plate_from_string(joined)
        sc = score_candidate(cand)
        if sc > best_score:
            best_text = cand
            best_score = sc
            best_overlay = img_rot.copy()

    return best_text, best_overlay


def save_overlay(img: np.ndarray, text: str):
    out = img.copy()
    h = max(30, int(0.07 * out.shape[0]))
    cv2.rectangle(out, (0,0), (out.shape[1], h+10), (255,255,255), -1)
    cv2.putText(out, text if text else "(sin lectura)", (10, h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
    return out


# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser(description="OCR matrículas ES con EasyOCR + postproceso 4+3")
    ap.add_argument("--in_dir",  default=IN_DIR,  help="Carpeta de entrada (recortes de matrículas)")
    ap.add_argument("--out_dir", default=OUT_DIR, help="Carpeta de salida (txt/overlays)")
    ap.add_argument("--gpu", action="store_true", help="Usar GPU si está disponible")
    ap.add_argument("--keep_failed", action="store_true", help="Guardar archivos aunque no haya lectura")
    args = ap.parse_args()

    # Inicializa EasyOCR una sola vez
    use_gpu = False
    if args.gpu and torch is not None:
        try:
            use_gpu = bool(torch and torch.cuda.is_available())
        except Exception:
            use_gpu = False
    print(f"[INFO] EasyOCR GPU={use_gpu}")
    reader = easyocr.Reader(['en'], gpu=use_gpu)  # 'en' vale para A-Z/0-9

    reset_output_dir(args.out_dir)

    patterns = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff")
    paths = []
    for pat in patterns:
        paths += glob.glob(os.path.join(args.in_dir, "**", pat), recursive=True)

    if not paths:
        print(f"[WARN] No se encontraron imágenes en: {args.in_dir}")
        return

    rows = []
    ok, fail = 0, 0

    for p in sorted(paths):
        img = cv2.imread(p)
        if img is None:
            print("[WARN] No pude leer:", p); fail += 1; continue

        t0 = time.perf_counter()
        text, overlay_base = read_plate_easyocr(img, reader)
        t1 = time.perf_counter()

        # Mirror de carpetas
        rel  = os.path.relpath(os.path.dirname(p), args.in_dir)
        out_sub = os.path.join(args.out_dir, rel)
        os.makedirs(out_sub, exist_ok=True)
        base = os.path.splitext(os.path.basename(p))[0]

        # Guarda txt
        with open(os.path.join(out_sub, base + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)

        # Overlay
        overlay = save_overlay(overlay_base, text)
        cv2.imwrite(os.path.join(out_sub, base + "_ocr.jpg"), overlay)

        # Mantener una imagen binaria mínima para inspección si se desea
        g = preprocess_plate(img)
        cv2.imwrite(os.path.join(out_sub, base + "_gray.png"), g)

        if not text and not args.keep_failed:
            # si no hay lectura, opcionalmente limpiar
            try:
                os.remove(os.path.join(out_sub, base + ".txt"))
                os.remove(os.path.join(out_sub, base + "_ocr.jpg"))
                os.remove(os.path.join(out_sub, base + "_gray.png"))
            except Exception:
                pass
            fail += 1
            print(f"[--] {base}: vacío   [TIME OCR={t1-t0:.3f}s]")
            continue

        rows.append({"path": p, "plate": text})
        ok += 1
        print(f"[OK] {base}: {text}   [TIME OCR={t1-t0:.3f}s]")

    # Resúmenes
    csv_path = os.path.join(args.out_dir, "resultats_ocr.csv")
    txt_all  = os.path.join(args.out_dir, "placas_validas.txt")
    txt_uni  = os.path.join(args.out_dir, "placas_unicas.txt")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(csv_path, index=False, encoding="utf-8")
        with open(txt_all, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['plate']}\t{r['path']}\n")
        cnt = Counter([r["plate"] for r in rows])
        with open(txt_uni, "w", encoding="utf-8") as f:
            for plate, n in cnt.most_common():
                f.write(f"{plate}\t{n}\n")
        print("\nGuardado:")
        print(" -", csv_path)
        print(" -", txt_all)
        print(" -", txt_uni)

    print(f"\nResumen: OK={ok}  FAIL={fail}  Total={ok+fail}")


if __name__ == "__main__":
    main()
