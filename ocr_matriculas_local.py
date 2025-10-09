# -*- coding: utf-8 -*-
"""
OCR de matrículas españolas (OpenCV puro, sin EasyOCR)
- Preprocesado robusto (CLAHE + BlackHat suave + binarización inteligente)
- Deskew ligero por búsqueda de ángulo (−4..+4º)
- Segmentación híbrida (componentes + proyección + contornos) con merge/split
- Normalización estable (44x64, polaridad consistente, recorte apretado)
- HOG + centroides por clase (dígitos/letras) generados sintéticamente con
  múltiples fuentes Hershey, escalas, grosores y augmentations. Cache en .npz
- Decodificación guiada por patrón 4 dígitos + 3 letras con corrección de confusiones

Uso:
    pip install opencv-python-headless numpy pandas
    python ocr_placas_opencv_puro.py \
        --in_dir ./outputs_engi/plates \
        --out_dir ./out/ocr_chars_opencv \
        --seg auto
"""

import os, glob, argparse, shutil, time, re
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Optional

# =========================
# RUTAS
# =========================
IN_DIR     = "./outputs_engi/plates"
OUT_DIR    = "./out/ocr_chars_opencv"
MODELS_DIR = "./ocr_models"   # cache .npz

# =========================
# Configuración básica
# =========================
DIGITOS = "0123456789"
LETRAS  = "BCDFGHJKLMNPRSTVWXYZ"  # ES moderno (sin vocales, sin Q/Ñ)
AN, AL  = 44, 64                  # ancho x alto del lienzo normalizado

# Correcciones típicas
MAPA_DIG = str.maketrans({'D':'0','Q':'0','O':'0','U':'0','L':'1','I':'1','T':'7','Z':'2','S':'5','B':'8','G':'6'})
MAPA_LET = str.maketrans({'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B','7':'T'})

# Patrones de búsqueda correctos para glob
IMG_PATTERNS = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff")

# =========================
# Prototipos (HOG) + centroides por clase
# =========================
PROT_DIG_X = None  # (N_d, D) float32  (muestras)
PROT_DIG_y = None  # (N_d,)   int
PROT_LET_X = None
PROT_LET_y = None
CENT_DIG   = None  # (10, D) centroides L2-normalizados
CENT_LET   = None  # (len(LETRAS), D)

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

# =========================
# 1) Preproceso + binarización “relleno” (no bordes)
# =========================

def preprocesar_placa(placa_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devolvemos (gris, bw) con texto BLANCO y fondo NEGRO.
    Estrategia:
      1) Gris + bilateral (reduce reflejos manteniendo bordes)
      2) Normalización de iluminación por división (g / blur grande)
      3) Generar 2-3 binarios candidatos (Otsu inv, Adaptive Gauss inv, Sauvola si existe)
      4) Post-proceso uniforme + score y elección del mejor
      5) Relleno de huecos (flood fill) -> glifos sólidos
    """
    g = cv2.cvtColor(placa_bgr, cv2.COLOR_BGR2GRAY)
    # suaviza reflejos sin perder bordes
    g = cv2.bilateralFilter(g, d=7, sigmaColor=60, sigmaSpace=60)

    # normalización de iluminación (division normalization)
    bg = cv2.GaussianBlur(g, (35, 35), 0)
    norm = (g.astype(np.float32) + 1.0) / (bg.astype(np.float32) + 1.0)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- Candidatos de binarización (texto blanco) ---
    # 1) Otsu inverso sobre norm
    _, c1 = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 2) Adaptativa gaussiana inversa
    c2 = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    # 3) Sauvola (si está disponible en ximgproc)
    cands = []
    try:
        import cv2.ximgproc as xip
        c3 = xip.niBlackThreshold(norm, 255, cv2.THRESH_BINARY, 31, k=0.2,
                                  binarizationMethod=xip.BINARIZATION_SAUVOLA)
        # niBlackThreshold devuelve texto NEGRO; lo invertimos a texto BLANCO
        c3 = cv2.bitwise_not(c3)
        cands = [c1, c2, c3]
    except Exception:
        cands = [c1, c2]

    # post-proceso uniforme para todos los candidatos
    cands = [_post_bin(c) for c in cands]

    # elige el mejor por varianza de proyección + ratio de blancos razonable
    best = max(cands, key=_score_bw)

    # rellena huecos para evitar "filigrana" de bordes
    bw = _fill_holes(best)

    return g, bw

def _post_bin(bw: np.ndarray) -> np.ndarray:
    """Suaviza ruido y mejora cohesión de trazos."""
    bw = cv2.medianBlur(bw, 3)
    # abre motas pequeñas y une trazos verticales finos
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    bw = cv2.dilate(bw, np.ones((3,2), np.uint8), iterations=1)
    return bw

def _score_bw(bw: np.ndarray) -> float:
    """Más varianza de proyección vertical y ratio de blancos razonable = mejor."""
    H, W = bw.shape
    p = (bw > 0).sum(axis=0).astype(np.float32)
    var = float(p.var())
    white_ratio = float((bw > 0).mean())
    # penaliza si hay muy pocos o demasiados pixeles blancos
    if white_ratio < 0.05 or white_ratio > 0.45:
        var *= 0.4
    # bonus ligero si el nº de componentes está cerca de 7–20 (texto segmentable)
    num, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    cc = max(0, num - 1)
    bonus = -0.02 * abs(cc - 12)
    return var + bonus

def _fill_holes(bw: np.ndarray) -> np.ndarray:
    """Rellena huecos internos para obtener glifos sólidos."""
    h, w = bw.shape
    inv = cv2.bitwise_not(bw)
    mask = np.zeros((h+2, w+2), np.uint8)
    im_ff = inv.copy()
    cv2.floodFill(im_ff, mask, (0, 0), 255)
    holes = cv2.bitwise_not(im_ff)  # agujeros
    filled = cv2.bitwise_or(bw, holes)
    # un cierre suave consolida cantos
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return filled

# =========================
# 2) Deskew ligero
# =========================

def _rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    cX, cY = w//2, h//2
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))
    M[0,2] += (nW/2) - cX
    M[1,2] += (nH/2) - cY
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _deskew_best(bw: np.ndarray, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    angles = [0,-4,-3,-2,-1,1,2,3,4]
    best_bw, best_bgr, best = bw, bgr, -1.0
    for a in angles:
        rbw = _rotate_bound(bw, a) if a else bw
        p = (rbw>0).sum(axis=0).astype(np.float32)
        score = float(p.var())
        if score > best:
            best = score
            best_bw = rbw
            best_bgr = _rotate_bound(bgr, a) if a else bgr
    return best_bw, best_bgr

# =========================
# 3) Segmentación (components / proyección / contornos) + filtros/merge/split
# =========================

def _filtro_bordes_y_extremos(boxes, H, W):
    out = []
    for (x,y,w,h) in boxes:
        ar = w/float(h+1e-6)
        area = w*h
        # recorta márgenes agresivos (banda EU a la izquierda y marcos)
        if x < 0.10*W and w > 0.03*W:
            continue
        if x+w > 0.98*W and w > 0.02*W:
            continue
        if h < 0.30*H or h > 0.98*H:
            continue
        if ar < 0.16 or ar > 1.35:
            continue
        if area < 0.004*H*W:
            continue
        out.append((x,y,w,h))
    out.sort(key=lambda b: b[0])
    return out


def segmentar_por_componentes(bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    H, W = bw.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cajas = []
    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        if area < 0.003*H*W:
            continue
        ar = w/float(h+1e-6)
        sol = area/float(w*h+1e-6)
        if 0.10<=ar<=1.35 and sol>0.15:
            cajas.append((x,y,w,h))
    cajas = _filtro_bordes_y_extremos(cajas, H, W)
    return cajas


def segmentar_por_proyeccion(bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    H, W = bw.shape
    top = int(0.06*H); bot = int(0.94*H)
    bwc = bw[top:bot, :]
    cols = (bwc > 0).astype(np.uint8)
    proj = cols.sum(axis=0).astype(np.float32)
    k = max(5, (W // 40) | 1)
    proj = cv2.GaussianBlur(proj.reshape(1, -1), (k,1), 0).ravel()
    thr = max(2.0, 0.14 * float(proj.max()))
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
    Hc = bwc.shape[0]
    for (l, r) in intervals:
        w = r - l + 1
        patch = bwc[:, l:r+1]
        ys = np.where(patch.any(axis=1))[0]
        if len(ys) == 0: continue
        y0, y1 = int(ys.min()), int(ys.max())
        h = y1 - y0 + 1
        if h < 0.32*Hc or h > 0.98*Hc:  continue
        if w < max(2, int(0.01*W)) or w > int(0.50*W): continue
        cajas.append((l, y0+top, w, h))
    cajas = _filtro_bordes_y_extremos(cajas, H, W)
    return cajas


def segmentar_por_contornos(bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    H, W = bw.shape
    contornos, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        ar   = w / float(h + 1e-6)
        area = w * h
        if h < 0.28*H or h > 0.98*H:  continue
        if w < 0.012*W or w > 0.48*W: continue
        if ar < 0.08   or ar > 1.35:  continue
        if area < 0.004*H*W:          continue
        cajas.append((x,y,w,h))
    cajas = _filtro_bordes_y_extremos(cajas, H, W)
    return cajas


def _merge_split(boxes: List[Tuple[int,int,int,int]], bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    if not boxes: return boxes
    boxes = sorted(boxes, key=lambda b:b[0])
    # merge por solape fuerte
    merged=[boxes[0]]
    for x,y,w,h in boxes[1:]:
        X,Y,W0,H0 = merged[-1]
        if x <= X + int(0.85*W0):
            nx = min(X,x); ny=min(Y,y)
            rx = max(X+W0, x+w); ry=max(Y+H0, y+h)
            merged[-1]=(nx,ny,rx-nx,ry-ny)
        else:
            merged.append((x,y,w,h))
    # split si muy ancha (usa valle de proyección)
    out=[]
    for x,y,w,h in merged:
        if w > 1.45*h:
            roi = bw[y:y+h, x:x+w]
            p = (roi>0).sum(axis=0)
            cut = int(np.argmin(p[h//6:-h//6])) + h//6
            cut = max(2, min(w-2, cut))
            out.append((x,y,cut,h))
            out.append((x+cut,y,w-cut,h))
        else:
            out.append((x,y,w,h))
    out.sort(key=lambda b:b[0])
    return out


def _join_near_boxes(boxes: List[Tuple[int,int,int,int]], W: int) -> List[Tuple[int,int,int,int]]:
    if not boxes: return boxes
    boxes = sorted(boxes, key=lambda b:b[0])
    out=[boxes[0]]
    for x,y,w,h in boxes[1:]:
        X,Y,W0,H0 = out[-1]
        gap = x - (X + W0)
        vert_overlap = min(Y+H0, y+h) - max(Y, y)
        if gap < 0.035*W and vert_overlap > 0.70*min(H0, h):
            nx=min(X,x); ny=min(Y,y)
            rx=max(X+W0, x+w); ry=max(Y+H0, y+h)
            out[-1] = (nx,ny,rx-nx,ry-ny)
        else:
            out.append((x,y,w,h))
    return out


def top7_por_altura(boxes: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    if len(boxes) <= 7: return boxes
    # prioriza altura ~ mediana y centrado en X
    Hs = np.array([h for (_,_,_,h) in boxes], dtype=np.float32)
    med = float(np.median(Hs))
    xs  = np.array([x+w/2 for (x,_,w,_) in boxes], dtype=np.float32)
    xmed = float(np.median(xs))
    scores = [ -abs(h-med) - 0.001*abs((x+w/2)-xmed) for (x,_,w,h) in boxes ]
    idx = np.argsort(scores)[-7:]
    return [boxes[i] for i in sorted(idx, key=lambda k: boxes[k][0])]


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

# --- Fallback por proyección: fuerza 7 cajas cuando todo se pega en 1-3 blobs ---
def _force7_by_projection(bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """
    Fuerza 7 cajas: 1) recorta banda de texto, 2) elimina banda EU,
    3) busca valles y genera 7 intervalos robustos.
    """
    H, W = bw.shape
    y0, y1 = _find_text_band(bw)
    # recorte vertical a la franja de texto
    strip = bw[y0:y1+1, :]

    # elimina la banda EU (izquierda) y el marco derecho
    xL = int(0.10 * W)  # si alguna placa queda justa, baja a 0.08
    xR = int(0.98 * W)
    xL = min(max(0, xL), strip.shape[1]-3)
    xR = min(strip.shape[1], xR)
    roi = strip[:, xL:xR]
    if roi.shape[1] < 20:
        return []

    # genera 7 cajas a partir de valles
    boxes = _seven_from_valleys(roi, x0_global=xL, y0_global=y0)

    # filtra anchos irreales (evita cortes en el marco)
    if boxes:
        Wtxt = boxes[-1][0] + boxes[-1][2] - boxes[0][0]
        w_min = max(6, int(0.06 * Wtxt))
        w_max = int(0.25 * Wtxt)
        boxes = [b for b in boxes if w_min <= b[2] <= w_max]

    return boxes


def _find_text_band(bw: np.ndarray) -> Tuple[int,int]:
    """Encuentra la franja horizontal donde está el texto (y recorta un poco)."""
    H, W = bw.shape
    p = (bw > 0).sum(axis=1).astype(np.float32)
    k = max(3, (H // 40) | 1)
    p = cv2.GaussianBlur(p.reshape(-1, 1), (1, k), 0).ravel()
    thr = max(3.0, 0.12 * float(p.max()))
    idx = np.where(p > thr)[0]
    if len(idx) == 0:
        return 0, H-1
    y0 = max(0, int(idx[0]) - 2)
    y1 = min(H-1, int(idx[-1]) + 2)
    return y0, y1


def _seven_from_valleys(roi: np.ndarray, x0_global: int, y0_global: int) -> List[Tuple[int,int,int,int]]:
    """
    ROI = texto (y recortado en X: sin banda EU). Busca 6 valles bien espaciados y devuelve 7 cajas.
    """
    H, W = roi.shape
    # Consolidamos trazos para que la proyección tenga valles limpios
    roi2 = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,5)), 1)
    p = (roi2 > 0).sum(axis=0).astype(np.float32)
    k = max(5, (W // 50) | 1)
    p = cv2.GaussianBlur(p.reshape(1, -1), (k, 1), 0).ravel()

    # Queremos 6 valles (mínimos). Usamos "eliminar alrededor" para espaciar.
    min_dist = max(6, int(W / 12))  # ≈ 12 columnas ~ 7-8 valles razonables
    cand = p.copy()
    idxs = []
    for _ in range(6):
        i = int(np.argmin(cand))
        idxs.append(i)
        l = max(0, i - min_dist)
        r = min(W-1, i + min_dist)
        cand[l:r+1] = cand.max()  # “anula” esa zona
    idxs = sorted(idxs)

    # Construimos 7 intervalos a partir de [0, v1, v2, ..., v6, W-1]
    bounds = [0] + idxs + [W-1]
    boxes = []
    for i in range(7):
        l = bounds[i]
        r = bounds[i+1]
        if r - l < 3:  # muy estrecho
            continue
        col = roi[:, l:r]
        ys = np.where(col.any(axis=1))[0]
        if len(ys) == 0:
            continue
        yy0, yy1 = int(ys.min()), int(ys.max())
        h = yy1 - yy0 + 1
        w = r - l
        boxes.append((l + x0_global, yy0 + y0_global, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes


# --- scoring de una propuesta de cajas por verosimilitud dígito/letra ---

def _score_boxes(placa_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> float:
    if not boxes: return -1e9
    boxes = sorted(boxes, key=lambda b:b[0])
    total = 0.0
    for i,b in enumerate(boxes[:7]):
        im = normalizar_char(placa_bgr, b)
        if i<4:
            _,s = _predict_char(im, DIGITOS)
        else:
            _,s = _predict_char(im, LETRAS)
        total += float(s)
    total -= 0.2*abs(len(boxes)-7)  # penaliza no tener 7
    return total

# =========================
# 4) Normalización 44×64 (polaridad consistente + recorte apretado)
# =========================

def normalizar_char(placa_bgr: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,w,h = box
    crop = placa_bgr[y:y+h, x:x+w]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # texto blanco consistente
    if (b==255).sum() < (b==0).sum():
        b = 255 - b
    ys, xs = np.where(b>0)
    if len(ys)>0:
        y0,y1 = ys.min(), ys.max()
        x0,x1 = xs.min(), xs.max()
        b = b[y0:y1+1, x0:x1+1]
    esc = min((AN-4)/max(1,b.shape[1]), (AL-4)/max(1,b.shape[0]))
    nw = max(1, int(round(b.shape[1]*esc)))
    nh = max(1, int(round(b.shape[0]*esc)))
    b_res = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_NEAREST)
    lienzo = np.zeros((AL, AN), np.uint8)
    yy = (AL - nh)//2; xx = (AN - nw)//2
    lienzo[yy:yy+nh, xx:xx+nw] = b_res
    return lienzo

# =========================
# 5) HOG + centroides
# =========================

def _hog():
    winSize=(AN,AL); blockSize=(8,8); blockStride=(4,4); cellSize=(4,4); nbins=9
    return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

_HOG = _hog()


def hog_feat(img44x64: np.ndarray) -> np.ndarray:
    f = _HOG.compute(img44x64).reshape(1,-1).astype(np.float32)
    n = np.linalg.norm(f)
    if n>0: f/=n
    return f


def _draw_char(ch: str, ancho=AN, alto=AL, escala=1.4, grosor=2, font=cv2.FONT_HERSHEY_SIMPLEX) -> np.ndarray:
    img = np.zeros((alto, ancho), np.uint8)
    (tw, th), base = cv2.getTextSize(ch, font, escala, grosor)
    x = (ancho - tw)//2; y = (alto + th)//2 - base
    cv2.putText(img, ch, (x,y), font, escala, 255, grosor, cv2.LINE_AA)
    return img


def _aug(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    ang = np.random.uniform(-4,4)
    M = cv2.getRotationMatrix2D((AN/2,AL/2), ang, 1.0)
    out = cv2.warpAffine(out, M, (AN,AL), flags=cv2.INTER_NEAREST, borderValue=0)
    if np.random.rand()<0.5:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        if np.random.rand()<0.5: out = cv2.erode(out,k,1)
        else: out = cv2.dilate(out,k,1)
    if np.random.rand()<0.3:
        out = cv2.GaussianBlur(out,(3,3),0)
    return out


def _build_prototypes(chars: str, n_per_class: int=260) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    fonts=[cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX,
           cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL]
    X=[]; y=[]
    for i,ch in enumerate(chars):
        feats=[]
        for _ in range(n_per_class):
            font=np.random.choice(fonts)
            esc=np.random.uniform(1.2,1.7)
            gros=np.random.randint(2,4)
            img=_draw_char(ch, escala=esc, grosor=gros, font=font)
            img=_aug(img)
            feats.append(hog_feat(img))
        feats=np.vstack(feats).astype(np.float32)
        X.append(feats); y.extend([i]*feats.shape[0])
    X=np.vstack(X).astype(np.float32)
    y=np.array(y,dtype=np.int32)
    # centroides por clase (L2)
    centroids=[]
    for i in range(len(chars)):
        c = np.mean(X[y==i], axis=0, keepdims=True)
        n = np.linalg.norm(c) + 1e-9
        centroids.append(c/n)
    C=np.vstack(centroids).astype(np.float32)
    return X,y,C


def _cosine_scores(f: np.ndarray, C: np.ndarray) -> np.ndarray:
    return (C @ f.ravel()).astype(np.float32)

# =========================
# 6) OCR por recorte
# =========================

def _segmentar(bw: np.ndarray, metodo: str) -> List[Tuple[int,int,int,int]]:
    m = metodo.lower()
    if m == "components":
        boxes = segmentar_por_componentes(bw)
    elif m == "proj":
        boxes = segmentar_por_proyeccion(bw)
    elif m == "contours":
        boxes = segmentar_por_contornos(bw)
    else:
        b1 = segmentar_por_componentes(bw)
        b2 = segmentar_por_proyeccion(bw)
        b3 = segmentar_por_contornos(bw)
        b4 = _force7_by_projection(bw)   # << NUEVA candidata “estricta”
        # elige por nº de cajas como quick-win
        boxes = max([b1, b2, b3, b4], key=lambda L: len(L))

    # merge/split + unión cercana (si ya las tienes, mantenlas)
    boxes = _merge_split(boxes, bw)
    H, W = bw.shape
    boxes = _join_near_boxes(boxes, W)

    # si aún no hay 6–8, intenta forzar 7
    if len(boxes) < 6 or len(boxes) > 8:
        fb = _force7_by_projection(bw)
        if len(fb) >= 6:
            boxes = fb

    # recorta/valida a imagen
    boxes = _clip_and_filter_boxes(boxes, H, W)
    boxes = top7_por_altura(boxes)
    return boxes



def _predict_char(img44x64: np.ndarray, clases: str) -> Tuple[str,float]:
    f = hog_feat(img44x64)
    if clases==DIGITOS:
        s = _cosine_scores(f, CENT_DIG)
        idx = int(np.argmax(s)); return clases[idx], float(s[idx])
    else:
        s = _cosine_scores(f, CENT_LET)
        idx = int(np.argmax(s)); return clases[idx], float(s[idx])


def reconocer_desde_recorte(placa_bgr: np.ndarray,
                            seg_method: str = "auto",
                            devolver_debug: bool = False
                           ) -> Tuple[str, List[Tuple[int,int,int,int]], Optional[Dict[str,np.ndarray]]]:
    _, bw = preprocesar_placa(placa_bgr)
    bw, placa_bgr = _deskew_best(bw, placa_bgr)

    # Candidata 1: segmentación normal
    boxes1 = _segmentar(bw, seg_method)
    score1 = _score_boxes(placa_bgr, boxes1)

    # Candidata 2: forzar 7 por proyección (por si todo está pegado)
    boxes2 = _force7_by_projection(bw)
    score2 = _score_boxes(placa_bgr, boxes2)
    # tras: boxes1 = _segmentar(bw, seg_method)
    H, W = placa_bgr.shape[:2]
    boxes1 = _clip_and_filter_boxes(boxes1, H, W)

    # si usas fallback por proyección:
    boxes2 = _force7_by_projection(bw)
    boxes2 = _clip_and_filter_boxes(boxes2, H, W)

    boxes = boxes1 if score1 >= score2 else boxes2

    # Partición 4 | 3
    boxes = sorted(boxes, key=lambda b:b[0])
    if len(boxes) >= 6:
        split = mayor_gap(boxes)
        seq = boxes[:split] + boxes[split:]
        izq = seq[:4]; der = seq[4:7]
    else:
        scores=[]
        for i,b in enumerate(boxes):
            im = normalizar_char(placa_bgr, b)
            _,sd = _predict_char(im, DIGITOS)
            _,sl = _predict_char(im, LETRAS)
            scores.append((i, sd-sl))  # >0 → parece dígito
        scores.sort(key=lambda t:t[1], reverse=True)
        cut = min(4, len(boxes))
        izq = [boxes[i] for i,_ in scores[:cut]]; izq.sort(key=lambda b:b[0])
        der = [b for i,b in enumerate(boxes) if i not in [k for k,_ in scores[:cut]]]
        der.sort(key=lambda b:b[0])

    txt_izq = ""; txt_der = ""
    for b in izq:
        im = normalizar_char(placa_bgr, b)
        ch,_ = _predict_char(im, DIGITOS)
        ch = ch.translate(MAPA_DIG)
        if not ch.isdigit(): ch='?'
        txt_izq += ch
    for b in der:
        im = normalizar_char(placa_bgr, b)
        ch,_ = _predict_char(im, LETRAS)
        ch = ch.translate(MAPA_LET)
        if ch not in LETRAS: ch='?'
        txt_der += ch

    txt_izq = (re.sub(r"[^0-9]","",txt_izq)[:4]).ljust(4,'?')
    txt_der = (''.join([c for c in txt_der if c in LETRAS])[:3]).ljust(3,'?')
    texto = f"{txt_izq} {txt_der}"

    if not devolver_debug:
        return texto, boxes, None
    H, W = placa_bgr.shape[:2]
    boxes = _clip_and_filter_boxes(boxes, H, W)

    overlay = placa_bgr.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(overlay, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(overlay, texto, (10, max(30, overlay.shape[0]-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    return texto, boxes, {'bw': bw, 'overlay': overlay}


def _clip_and_filter_boxes(boxes, H, W, min_wh=3):
    """Recorta cajas a los límites de la imagen y descarta las inválidas."""
    out = []
    for (x,y,w,h) in boxes:
        x = max(0, int(round(x))); y = max(0, int(round(y)))
        w = int(round(w)); h = int(round(h))
        if x >= W or y >= H:  # totalmente fuera
            continue
        w = min(w, W - x); h = min(h, H - y)
        if w >= min_wh and h >= min_wh:
            out.append((x,y,w,h))
    return out

# =========================
# 7) Guardado y limpieza
# =========================

def guardar_caracteres_segmentados(out_sub: str, base: str, placa_bgr: np.ndarray, boxes):
    H, W = placa_bgr.shape[:2]
    boxes = _clip_and_filter_boxes(boxes, H, W)
    if not boxes:
        return
    ch_dir = os.path.join(out_sub, base + "_chars")
    os.makedirs(ch_dir, exist_ok=True)

    for i, (x,y,w,h) in enumerate(boxes):
        if w <= 0 or h <= 0:
            continue
        crop = placa_bgr[y:y+h, x:x+w]
        # seguridad extra
        if crop is None or crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            # log opcional:
            # print(f"[SKIP] Caja vacía en {base} -> ({x},{y},{w},{h})")
            continue

        ok1 = cv2.imwrite(os.path.join(ch_dir, f"{base}_c{i:02d}.png"), crop)

        norm = normalizar_char(placa_bgr, (x,y,w,h))
        if norm is not None and norm.size > 0 and norm.shape[0] > 0 and norm.shape[1] > 0:
            ok2 = cv2.imwrite(os.path.join(ch_dir, f"{base}_c{i:02d}_norm.png"), norm)
        # si quieres, puedes comprobar ok1/ok2 y hacer print si fallan


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
            if os.path.exists(fp):
                try: os.remove(fp)
                except Exception: pass
        return True
    return False


# =========================
# 8) MAIN
# =========================

def main():
    global PROT_DIG_X, PROT_DIG_y, PROT_LET_X, PROT_LET_y, CENT_DIG, CENT_LET

    ap = argparse.ArgumentParser(description="OCR matrículas (OpenCV puro): prepro+deskew+segmentación+HOG centroides")
    ap.add_argument("--in_dir",  default=IN_DIR,  help="Carpeta de entrada (recortes de matrículas)")
    ap.add_argument("--out_dir", default=OUT_DIR, help="Carpeta de salida (txt/overlays/binarios)")
    ap.add_argument("--models_dir", default=MODELS_DIR, help="Carpeta de cache (.npz)")
    ap.add_argument("--seg", choices=["auto","components","proj","contours"], default="auto", help="Método de segmentación")
    ap.add_argument("--show", action="store_true", help="Mostrar overlay por pantalla")
    ap.add_argument("--show_ms", type=int, default=0, help="Mostrar overlays sin bloquear (ms por imagen). Requiere --show.")
    ap.add_argument("--keep_failed", action="store_true", help="No borrar outputs aunque no haya texto")
    args = ap.parse_args()

    # Cache/cálculo de prototipos HOG
    os.makedirs(args.models_dir, exist_ok=True)
    npz_d = os.path.join(args.models_dir, "hog_centroids_digits.npz")
    npz_l = os.path.join(args.models_dir, "hog_centroids_letters.npz")

    def load_or_build(npz_path, chars):
        try:
            if os.path.exists(npz_path):
                d = np.load(npz_path)
                X = d['X'].astype(np.float32); y = d['y'].astype(np.int32); C = d['C'].astype(np.float32)
                return X,y,C
        except Exception:
            pass
        print(f"[INFO] Construyendo prototipos HOG para '{chars}'…")
        X,y,C = _build_prototypes(chars, n_per_class=260)
        try:
            np.savez_compressed(npz_path, X=X, y=y, C=C)
        except Exception:
            pass
        return X,y,C

    PROT_DIG_X, PROT_DIG_y, CENT_DIG = load_or_build(npz_d, DIGITOS)
    PROT_LET_X, PROT_LET_y, CENT_LET = load_or_build(npz_l, LETRAS)

    # salida limpia
    reset_output_dir(args.out_dir)

    # encuentra imágenes
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

        # espejo de carpetas
        rel  = os.path.relpath(os.path.dirname(p), args.in_dir)
        o_sub = os.path.join(args.out_dir, rel)
        os.makedirs(o_sub, exist_ok=True)
        base = os.path.splitext(os.path.basename(p))[0]

        # guarda txt
        with open(os.path.join(o_sub, base + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)

        # guardar binaria y overlay
        if dbg is not None:
            cv2.imwrite(os.path.join(o_sub, base + "_bin.png"), dbg['bw'])
            cv2.imwrite(os.path.join(o_sub, base + "_ocr.jpg"), dbg['overlay'])

        # guardar chars
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
            delay = args.show_ms if args.show_ms and args.show_ms > 0 else 1
            key = cv2.waitKey(delay) & 0xFF
            if key == 27:  # ESC
                break

    if args.show:
        cv2.destroyAllWindows()

    # === Resúmenes ===
    csv_path = os.path.join(args.out_dir, "resultats_ocr.csv")
    txt_all  = os.path.join(args.out_dir, "placas_validas.txt")
    txt_uni  = os.path.join(args.out_dir, "placas_unicas.txt")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(csv_path, index=False, encoding="utf-8")
        with open(txt_all, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['plate']}	{r['path']}")
        cnt = Counter([r["plate"] for r in rows])
        with open(txt_uni, "w", encoding="utf-8") as f:
            for plate, n in cnt.most_common():
                f.write(f"{plate}	{n}")
        print("Guardado:")
        print(" -", csv_path)
        print(" -", txt_all)
        print(" -", txt_uni)

    print(f"Resumen: OK={ok}  FAIL={fail}  Total={ok+fail}")


if __name__ == "__main__":
    main()
