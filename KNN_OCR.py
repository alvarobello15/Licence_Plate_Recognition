# -*- coding: utf-8 -*-
import os, glob, argparse, shutil, time
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Optional

# =========================
# RUTAS (locales)
# =========================
IN_DIR     = "./outputs_engi/plates"
OUT_DIR    = "./out/ocr_chars"
MODELS_DIR = "./ocr_models"

# =========================
# Configuración básica
# =========================
DIGITOS = "0123456789"
LETRAS  = "BCDFGHJKLMNPQRSTVWXYZ"   # ES moderno sin vocales
AN, AL  = 44, 64                    # ancho x alto para normalizar caracteres

# Correcciones típicas (opcional)
MAPA_DIG = str.maketrans({'D':'0','Q':'0','O':'0','L':'1','I':'1','Z':'2','S':'5','B':'8','G':'6'})
MAPA_LET = str.maketrans({'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B'})

# Patrones de búsqueda correctos para glob
IMG_PATTERNS = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff")

# =========================
# Globals para modelos (se cargan en main)
# =========================
KNN_DIG = None
KNN_LET = None
CLS_DIG = DIGITOS
CLS_LET = LETRAS
K_KNN   = 3  # vecinos

# =========================
# 0) Utilidad: limpiar salida con salvaguardas
# =========================

def reset_output_dir(out_dir: str):
    out_dir = os.path.abspath(out_dir)
    forbidden = {
        "/", "C:\\", "C:\\Windows", "C:\\Users", "C:\\Program Files",
        os.path.expanduser("~"),
    }
    if out_dir in forbidden or len(out_dir) < 5:
        raise ValueError(f"[ABORT] Ruta de salida peligrosa: {out_dir}")
    if os.path.isdir(out_dir):
        print(f"[INFO] Limpiando salida: {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

# =========================
# 1) Preproceso (OpenCV)
# =========================

def preprocesar_placa(placa_bgr: np.ndarray,
                      alt_min: int = 120, alt_max: int = 200,
                      kW_div: int = 18, kH_div: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """gris -> resize -> CLAHE -> BlackHat -> Otsu (texto=blanco) -> limpieza suave"""
    g0 = cv2.cvtColor(placa_bgr, cv2.COLOR_BGR2GRAY)

    # resize a altura controlada (manteniendo aspecto)
    h, w = g0.shape
    nh = int(np.clip(h, alt_min, alt_max))
    nw = int(round(w * (nh / max(1.0, h))))
    interp = cv2.INTER_CUBIC if nh > h else cv2.INTER_AREA
    g = cv2.resize(g0, (nw, nh), interpolation=interp)

    # contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)

    # kernel del blackhat (depende del tamaño de la placa)
    H, W = g.shape
    kW = max(3, (W // kW_div) | 1)   # impares
    kH = max(3, (H // kH_div) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kW, kH))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)

    # binarización (texto = blanco)
    enh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    _, bw = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # limpieza ligera para no perder trazos finos
    k = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    return g, bw

# =========================
# 2) Segmentación (dos alternativas)
# =========================

def segmentar_por_contornos(bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    H, W = bw.shape
    contornos, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        ar   = w / float(h + 1e-6)
        area = w * h
        if h < 0.30*H or h > 0.95*H:  continue
        if w < 0.015*W or w > 0.45*W: continue
        if ar < 0.10   or ar > 1.20:  continue
        if area < 0.006*H*W:          continue
        cajas.append((x,y,w,h))
    cajas.sort(key=lambda b: b[0])
    return cajas


def segmentar_por_proyeccion(bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    H, W = bw.shape
    top = int(0.08*H); bot = int(0.92*H)
    bwc = bw[top:bot, :].copy()
    Hc = bwc.shape[0]
    cols = (bwc > 0).astype(np.uint8)
    proj = cols.sum(axis=0).astype(np.float32)
    k = max(5, (W // 40) | 1)
    proj = cv2.GaussianBlur(proj.reshape(1, -1), (k, 1), 0).ravel()
    thr = max(2.0, 0.18 * float(proj.max()))
    mask = (proj > thr).astype(np.uint8)
    intervals = []
    i = 0
    while i < W:
        if mask[i]:
            j = i
            while j < W and mask[j]:
                j += 1
            intervals.append((i, j-1))
            i = j
        else:
            i += 1
    cajas = []
    for (l, r) in intervals:
        w = r - l + 1
        patch = bwc[:, l:r+1]
        ys = np.where(patch.any(axis=1))[0]
        if len(ys) == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        h = y1 - y0 + 1
        if h < 0.35*Hc or h > 0.98*Hc:  continue
        if w < max(2, int(0.01*W)) or w > int(0.48*W): continue
        cajas.append((l, y0+top, w, h))
    cajas.sort(key=lambda b: b[0])
    if len(cajas) > 9:
        idx = np.argsort([-b[3] for b in cajas])[:9]
        cajas = [cajas[i] for i in sorted(idx)]
    return cajas


def top7_por_altura(boxes: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    if len(boxes) <= 7: return boxes
    idx = np.argsort([-b[3] for b in boxes])[:7]
    return [boxes[i] for i in sorted(idx)]


def mayor_gap(boxes: List[Tuple[int,int,int,int]]) -> int:
    if len(boxes) < 2: return 4
    gaps = []
    for i in range(1, len(boxes)):
        xL = boxes[i-1][0] + boxes[i-1][2]
        xR = boxes[i][0]
        gaps.append((xR - xL, i))
    split = max(gaps, key=lambda t: t[0])[1]
    if abs(split - 4) > 2:
        split = 4 if len(boxes) >= 7 else min(4, len(boxes))
    return split

# =========================
# 3) Normalización (44×64)
# =========================

def normalizar_char(placa_bgr: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,w,h = box
    crop = placa_bgr[y:y+h, x:x+w]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = 255 - b  # texto blanco
    esc = min((AN-4)/max(1,w), (AL-4)/max(1,h))
    nw = max(1, int(round(w*esc)))
    nh = max(1, int(round(h*esc)))
    b_res = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_NEAREST)
    lienzo = np.zeros((AL, AN), np.uint8)
    xx = (AN - nw)//2; yy = (AL - nh)//2
    lienzo[yy:yy+nh, xx:xx+nw] = b_res
    return lienzo  # [0..255], 64x44 (AL x AN)

# =========================
# 4) Clasificador k-NN (OpenCV) con HOG
# =========================

def _hog():
    # HOG adaptado a 44x64 (ANxAL): celdas 4x4, bloques 8x8, stride 4x4, 9 bins
    winSize      = (AN, AL)
    blockSize    = (8, 8)
    blockStride  = (4, 4)
    cellSize     = (4, 4)
    nbins        = 9
    return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

_HOG = _hog()


def feat_hog(img44x64: np.ndarray) -> np.ndarray:
    f = _HOG.compute(img44x64)  # (N,1)
    f = f.reshape(1, -1).astype(np.float32)
    # normalización L2 por estabilidad
    n = np.linalg.norm(f)
    if n > 0: f /= n
    return f


def construir_Xy_knn(chars: str, n_por_clase: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i, ch in enumerate(chars):
        for _ in range(n_por_clase):
            img = dibujar_char(ch)
            img = augment_suave(img)
            X.append(feat_hog(img))
            y.append(float(i))  # KNN espera float32
    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.float32).reshape(-1,1)


def entrenar_knn(chars: str, n_por_clase: int = 300, k: int = K_KNN):
    X, y = construir_Xy_knn(chars, n_por_clase)
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(k)
    knn.setIsClassifier(True)
    knn.train(X, cv2.ml.ROW_SAMPLE, y)
    return knn, chars


def cargar_o_entrenar_modelos(models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    dig_path = os.path.join(models_dir, "knn_digits.xml")
    let_path = os.path.join(models_dir, "knn_letters.xml")

    def _load_knn(path):
        try:
            return cv2.ml.KNearest_load(path)
        except Exception:
            return None

    knn_dig = _load_knn(dig_path)
    knn_let = _load_knn(let_path)

    if knn_dig is not None and knn_let is not None:
        print("[INFO] Modelos kNN cargados de", models_dir)
    else:
        print("[INFO] Entrenando kNN dígitos…")
        knn_dig, _ = entrenar_knn(DIGITOS, n_por_clase=300, k=K_KNN)
        print("[INFO] Entrenando kNN letras…")
        knn_let, _ = entrenar_knn(LETRAS,  n_por_clase=300, k=K_KNN)
        # Intentar guardar (algunas builds de OpenCV lo permiten)
        try:
            knn_dig.save(dig_path); knn_let.save(let_path)
            print("[INFO] Modelos kNN guardados en", models_dir)
        except Exception:
            print("[WARN] No se pudo guardar kNN; se reentrenará en futuras ejecuciones")
    return knn_dig, DIGITOS, knn_let, LETRAS


def predecir_char_knn(img44x64: np.ndarray, modelo, clases: str, k: int = K_KNN) -> str:
    f = feat_hog(img44x64)
    modelo.setDefaultK(k)
    _, res, _, _ = modelo.findNearest(f, k)
    idx = int(res[0,0])
    return clases[idx] if 0 <= idx < len(clases) else "?"

# =========================
# 5) OCR de una matrícula recortada (con seg. conmutable)
# =========================

def _segmentar(bw: np.ndarray, metodo: str) -> List[Tuple[int,int,int,int]]:
    metodo = metodo.lower()
    if metodo == "proj":
        return segmentar_por_proyeccion(bw)
    elif metodo == "contours":
        return segmentar_por_contornos(bw)
    else:
        boxes = segmentar_por_proyeccion(bw)
        if len(boxes) < 6:
            boxes2 = segmentar_por_contornos(bw)
            if len(boxes2) > len(boxes):
                return boxes2
            if len(boxes2) == len(boxes) and len(boxes) > 0:
                h1 = np.mean([b[3] for b in boxes])
                h2 = np.mean([b[3] for b in boxes2])
                return boxes2 if h2 > h1 else boxes
        return boxes


def reconocer_desde_recorte(placa_bgr: np.ndarray,
                            seg_method: str = "auto",
                            devolver_debug: bool = False
                           ) -> Tuple[str, List[Tuple[int,int,int,int]], Optional[Dict[str,np.ndarray]]]:
    _, bw = preprocesar_placa(placa_bgr)
    boxes = _segmentar(bw, seg_method)
    boxes = top7_por_altura(boxes)

    if len(boxes) >= 6:
        split = mayor_gap(boxes)
        seq = boxes[:split] + boxes[split:]
        izq = seq[:4]
        der = seq[4:7]
    else:
        izq = boxes[:min(4, len(boxes))]
        der = boxes[min(4, len(boxes)):min(7, len(boxes))]

    txt_izq = "".join(predecir_char_knn(normalizar_char(placa_bgr, b), KNN_DIG, CLS_DIG) for b in izq)
    txt_der = "".join(predecir_char_knn(normalizar_char(placa_bgr, b), KNN_LET, CLS_LET) for b in der)

    txt_izq = txt_izq.translate(MAPA_DIG)
    txt_der = txt_der.translate(MAPA_LET)
    txt_izq = (''.join(c for c in txt_izq if c.isdigit()))[:4].ljust(4,'?')
    txt_der = (''.join(c for c in txt_der if c.isalpha()))[:3].ljust(3,'?')
    texto = f"{txt_izq} {txt_der}"

    if not devolver_debug:
        return texto, boxes, None

    overlay = placa_bgr.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(overlay, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(overlay, texto, (10, max(30, overlay.shape[0]-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    return texto, boxes, {'bw': bw, 'overlay': overlay}

# =========================
# 6) Utilidades: guardar y limpieza
# =========================

def guardar_caracteres_segmentados(out_sub: str, base: str, placa_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]]):
    ch_dir = os.path.join(out_sub, base + "_chars")
    os.makedirs(ch_dir, exist_ok=True)
    for i, box in enumerate(boxes):
        x,y,w,h = box
        crop = placa_bgr[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(ch_dir, f"{base}_c{i:02d}.png"), crop)
        norm = normalizar_char(placa_bgr, box)
        cv2.imwrite(os.path.join(ch_dir, f"{base}_c{i:02d}_norm.png"), norm)


def cleanup_if_empty(text: str, out_sub: str, base: str, keep_failed: bool=False) -> bool:
    if keep_failed:
        return False
    clean = ''.join([c for c in text.upper() if c.isalnum()])
    if len(clean) == 0:
        targets = [
            os.path.join(out_sub, base + ".txt"),
            os.path.join(out_sub, base + "_ocr.jpg"),
            os.path.join(out_sub, base + "_bin.png"),
        ]
        for fp in targets:
            if os.path.exists(fp): os.remove(fp)
        return True
    return False

# =========================
# 7) MAIN: batch con rutas locales
# =========================

def main():
    global KNN_DIG, KNN_LET, CLS_DIG, CLS_LET

    ap = argparse.ArgumentParser(description="OCR matrículas (OpenCV): proyección/contornos + kNN(HOG)")
    ap.add_argument("--in_dir",  default=IN_DIR,  help="Carpeta de entrada (recortes de matrículas)")
    ap.add_argument("--out_dir", default=OUT_DIR, help="Carpeta de salida (txt/overlays/binarios)")
    ap.add_argument("--models_dir", default=MODELS_DIR, help="Carpeta de modelos KNN")
    ap.add_argument("--seg", choices=["proj","contours","auto"], default="auto", help="Método de segmentación")
    ap.add_argument("--show", action="store_true", help="Mostrar overlay por pantalla (bloquea con Enter)")
    ap.add_argument("--keep_failed", action="store_true", help="No borrar outputs aunque no haya texto")
    ap.add_argument("--k", type=int, default=K_KNN, help="Vecinos para kNN (default=3)")
    args = ap.parse_args()

    # Cargar/entrenar modelos kNN
    KNN_DIG, CLS_DIG, KNN_LET, CLS_LET = cargar_o_entrenar_modelos(args.models_dir)

    # Actualizar k dinámicamente si el usuario lo cambia
    if args.k != K_KNN:
        KNN_DIG.setDefaultK(args.k)
        KNN_LET.setDefaultK(args.k)

    reset_output_dir(args.out_dir)

    paths = []
    for pat in IMG_PATTERNS:
        paths += glob.glob(os.path.join(args.in_dir, "**", pat), recursive=True)

    if not paths:
        print(f"[WARN] No se encontraron imágenes en: {args.in_dir}")
    else:
        print(f"[INFO] Encontradas {len(paths)} imágenes")

    rows = []
    ok, fail = 0, 0
    for p in sorted(paths):
        img = cv2.imread(p)
        if img is None:
            print("[WARN] No pude leer:", p); fail += 1; continue

        t0 = time.perf_counter()
        text, boxes, dbg = reconocer_desde_recorte(img, seg_method=args.seg, devolver_debug=True)
        t1 = time.perf_counter()

        rel  = os.path.relpath(os.path.dirname(p), args.in_dir)
        o_sub = os.path.join(args.out_dir, rel)
        os.makedirs(o_sub, exist_ok=True)
        base = os.path.splitext(os.path.basename(p))[0]

        with open(os.path.join(o_sub, base + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)

        if dbg is not None:
            cv2.imwrite(os.path.join(o_sub, base + "_bin.png"), dbg['bw'])
            cv2.imwrite(os.path.join(o_sub, base + "_ocr.jpg"), dbg['overlay'])

        if boxes:
            guardar_caracteres_segmentados(o_sub, base, img, boxes)

        if cleanup_if_empty(text, o_sub, base, keep_failed=args.keep_failed):
            fail += 1
            print(f"[--] {base}: vacío   [TIME OCR={t1-t0:.3f}s]")
            continue

        rows.append({"path": p, "plate": text})
        ok += 1
        print(f"[OK] {base}: {text}   [TIME OCR={t1-t0:.3f}s]   Boxes={len(boxes)}")

        if args.show and dbg is not None:
            cv2.imshow("OCR", dbg['overlay'])
            if cv2.waitKey(0) & 0xFF == 27:
                break

    if args.show:
        cv2.destroyAllWindows()

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
