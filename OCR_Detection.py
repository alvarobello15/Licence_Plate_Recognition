# Instal·la una vegada a Colab
# %pip -q install easyocr opencv-contrib-python pandas

import cv2, os, glob, re, pandas as pd
import numpy as np
import easyocr

# --- 1) preprocess + segmentació robusta ---
def segment_plate_chars(plate_bgr):
    # 1) A gris i normalitza l'escala (fixem alçada i mantenim proporció)
    g = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    MIN_H, MAX_H = 120, 200
    h, w = g.shape
    new_h = min(MAX_H, max(MIN_H, h))
    new_w = int(w * (new_h / float(h)))

    interp = cv2.INTER_CUBIC if new_h > h else cv2.INTER_AREA
    g = cv2.resize(g, (new_w, new_h), interpolation=interp)
    plate_resized = cv2.resize(plate_bgr, (new_w, new_h), interpolation=interp)

    # 2) CLAHE + BlackHat
    #    - CLAHE: millora el contrast **local** (ombres/reflexos) sense cremar el fons.
    #    - BlackHat: ressalta **detalls foscos petits** (caràcters) sobre **fons clar** (placa).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    # kernel morfològic proporcional a la mida del text (no fix) i imparell (|1)
    H, W = g.shape
    kW = max(3, (W // 12) | 1)
    kH = max(3, (H // 6)  | 1)
    se  = cv2.getStructuringElement(cv2.MORPH_RECT, (kW, kH))
    bh  = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)

    # Normalitzem a [0,255] perquè la binarització sigui estable
    enh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)

    # 3) Binarització
    _, bw = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4) Morfologia lleu per tancar talls i llevar soroll fi
    """
    Podem passar d'això crec
    """

    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    # bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k, iterations=1)

    # 5) Components connexes + filtres geomètrics bàsics (altura, amplada, AR, àrea)
    H, W = bw.shape
    num, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        ar = ww / float(hh + 1e-6)
        if hh < 0.50*H or hh > 0.95*H:  continue
        if ww < 0.02*W or ww > 0.35*W:  continue
        if ar < 0.18 or ar > 0.95:     continue
        if area < 0.012*H*W:           continue
        boxes.append((x, y, ww, hh))

    boxes = sorted(boxes, key=lambda b: b[0])

    # 6) Retalls BGR a la mateixa escala que 'bw'
    char_crops = [plate_resized[y:y+hh, x:x+ww] for (x, y, ww, hh) in boxes]
    return bw, boxes, char_crops



def cleanup_if_empty(text, out_sub, base):
    # vacío si no hay nada, o solo ?/espacios
    clean = re.sub(r'[^A-Z0-9?]', '', (text or '').upper()).replace('?', '')
    if len(clean) == 0:
        # borra txt + imágenes asociadas
        targets = [
            os.path.join(out_sub, base + ".txt"),
            os.path.join(out_sub, base + "_ocr.jpg"),
            os.path.join(out_sub, base + "_bin.png"),
        ] + glob.glob(os.path.join(out_sub, base + "_ch*.png"))
        for fp in targets:
            if os.path.exists(fp):
                os.remove(fp)
        return True
    return False

# --- 2) OCR amb regles: 4 dígits + 3 lletres ---
reader = easyocr.Reader(['en'])  # prou per digits+MAJ
DIGITS = "0123456789"
LETTERS = "BCDFGHJKLMNPQRSTVWXYZ"

# correccions context-dependents
to_digit = str.maketrans({'D':'0','Q':'0','L':'1','Z':'2','S':'5','B':'8','G':'6'})
to_letter = str.maketrans({'2':'Z','5':'S','6':'G','8':'B'})

def ocr_char(img, allow):
    # prepara imatge: blanc de fons, negre de text, mida fixa
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (44,64), interpolation=cv2.INTER_CUBIC) #Fem una mida fixa per totes les imatges
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Convertim la imatge a blanc i negre amb el threshold, el otsu ens troba el threshold perfecte el binary ens passsa valorsa  0  i 255
    b = 255 - b
    txt = reader.readtext(b, detail=0, allowlist=allow)
    return (txt[0] if txt else "")

def recognize_plate(plate_bgr):
    _, boxes, chars = segment_plate_chars(plate_bgr)

    # si hi ha soroll, queda't amb com a màxim 7 (els més alts)
    if len(chars) > 7:
        idx = np.argsort([-c.shape[0] for c in chars])[:7]
        chars = [chars[i] for i in sorted(idx)]
        boxes = [boxes[i] for i in sorted(idx)]

    # si n'hi ha massa pocs, prova OCR global molt simple
    if len(chars) < 6:
        t = reader.readtext(plate_bgr, detail=0, allowlist=DIGITS+LETTERS)
        if t:
            s = re.sub(r'[^A-Z0-9]', '', ''.join(t).upper())
            m = re.search(r'(\d{4})([A-Z]{3})', s)
            if m: return f"{m.group(1)} {m.group(2)}", []
        return "", []

    # separa dígits | lletres pel gap més gran; força 4+3
    gaps = []
    for i in range(1, len(boxes)):
        xL = boxes[i-1][0] + boxes[i-1][2]
        xR = boxes[i][0]
        gaps.append((xR - xL, i))
        
    split = max(gaps)[1] if gaps else 4
    if abs(split - 4) > 2:  # estabilitza
        split = 4 if len(boxes) >= 7 else min(4, len(boxes))

    left  = (chars[:split] + chars[split:])[:4]
    right = (chars[:split] + chars[split:])[4:7]

    left_txt  = ''.join((ocr_char(c, DIGITS)   or '?') for c in left).translate(to_digit)
    right_txt = ''.join((ocr_char(c, LETTERS) or '?') for c in right).upper().translate(to_letter)

    left_txt  = re.sub(r'[^0-9]', '', left_txt)[:4].ljust(4, '?')
    right_txt = re.sub(r'[^A-Z]', '', right_txt)[:3].ljust(3, '?')
    return f"{left_txt} {right_txt}", boxes


# --- 3) procés lot: llegeix de plates/**, guarda TXT, CSV i imatges anotades ---
IN_DIR  = "/content/gdrive/MyDrive/yolo_outputs/predict_cv2/fotos_mobil/plates"
OUT_DIR = "/content/gdrive/MyDrive/yolo_outputs/ocr_chars"
os.makedirs(OUT_DIR, exist_ok=True)

ex = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff")
paths=[]
for e in ex: paths += glob.glob(os.path.join(IN_DIR,"**",e), recursive=True)

rows = []
for p in paths:
    img = cv2.imread(p)
    if img is None:
        continue

    text, boxes = recognize_plate(img)

    # carpeta de salida
    rel = os.path.relpath(os.path.dirname(p), IN_DIR)
    out_sub = os.path.join(OUT_DIR, rel)
    os.makedirs(out_sub, exist_ok=True)
    base = os.path.splitext(os.path.basename(p))[0]

    # guarda el .txt primero
    txt_path = os.path.join(out_sub, base + ".txt")
    with open(txt_path, "w") as f:
        f.write(text)

    # si el texto está vacío (o solo ?), limpia y NO lo añadimos a "rows"
    if cleanup_if_empty(text, out_sub, base):
        continue

    # aquí ya es válido -> añádelo a la colección para el CSV final
    rows.append({"path": p, "plate": text})

    # (opcional) debug: comentar si no lo quieres
    bw, seg_boxes, _ = segment_plate_chars(img)
    ann = img.copy()
    for (x,y,w,h) in seg_boxes:
        cv2.rectangle(ann,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(ann, text, (10, max(30, ann.shape[0]-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.imwrite(os.path.join(out_sub, base + "_ocr.jpg"), ann)
    cv2.imwrite(os.path.join(out_sub, base + "_bin.png"), bw)

# ===== al final del script: escribe los resúmenes =====
import pandas as pd
csv_path = os.path.join(OUT_DIR, "resultats_ocr.csv")
txt_all  = os.path.join(OUT_DIR, "placas_validas.txt")
txt_uni  = os.path.join(OUT_DIR, "placas_unicas.txt")

# CSV con path y placa (solo válidas)
df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)

# TXT con todas las placas (una por línea)
with open(txt_all, "w") as f:
    for r in rows:
        f.write(f"{r['plate']}\t{r['path']}\n")

# TXT con placas únicas y conteo
from collections import Counter
cnt = Counter([r["plate"] for r in rows])
with open(txt_uni, "w") as f:
    for plate, n in cnt.most_common():
        f.write(f"{plate}\t{n}\n")

print("Guardado:")
print(" -", csv_path)
print(" -", txt_all)
print(" -", txt_uni)



#MORE COMPLEX VERSION

# Instal·la una vegada a Colab
# %pip -q install easyocr opencv-contrib-python pandas

# import cv2, os, glob, re, pandas as pd
# import numpy as np
# import easyocr

# # --- 1) preprocess + segmentació robusta ---
# def segment_plate_chars(plate_bgr):
#     g = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) #Passem la imatge de BGR a escala de grisos (0,255)
#     # mida mínima útil
#     target_h = 160
#     scale = max(1.0, target_h / max(1, g.shape[0]))
#     g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

#     # millora local (CLAHE) + BlackHat per ressaltar text fosc
#     """
#     Clahe i Blackhat van bé per si hi ha sombres ja que ressalten els caràcters
#     """
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     g = clahe.apply(g)
#     H0, W0 = g.shape
#     pad = max(1, int(0.03 * min(H0, W0)))
#     g = g[pad:H0-pad, pad:W0-pad]
#     H, W = g.shape

#     kW = max(3, ((W // 12) | 1)); kH = max(3, ((H // 6) | 1))
#     bh  = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
#                            cv2.getStructuringElement(cv2.MORPH_RECT,(kW,kH)))
#     enh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)

#     # Sauvola/Niblack si hi ha contrib; si no, Otsu
#     try:
#         import cv2.ximgproc as xip
#         bw = xip.niBlackThreshold(enh, 255, cv2.THRESH_BINARY,
#                                   blockSize=31, k=-0.2)
#         bw = 255 - bw
#     except Exception:
#         _, bw = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#     # morfologia
#     bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
#     bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), 1)

#     # limita a la línia del text (projecció horitzontal)
#     row = (bw>0).sum(axis=1)
#     y1 = np.argmax(row>0.05*bw.shape[1])
#     y2 = bw.shape[0]-np.argmax(row[::-1]>0.05*bw.shape[1])
#     bw  = bw[y1:y2,:]; H, W = bw.shape

#     # components connexes
#     num, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
#     boxes=[]
#     for i in range(1,num):
#         x,y,w,h,area = stats[i]
#         ar = w/(h+1e-6)
#         if h < 0.50*H or h > 0.95*H:  continue
#         if w < 0.02*W or w > 0.35*W:  continue
#         if ar < 0.18 or ar > 0.95:    continue
#         if area < 0.012*H*W:          continue
#         boxes.append((x,y,w,h))

#     # fallback: talls per projecció vertical si n’hi ha massa/pocs
#     if len(boxes) < 6 or len(boxes) > 9:
#         col = (bw>0).sum(axis=0).astype(np.float32)
#         col = cv2.GaussianBlur(col,(0,0),sigmaX=3)
#         thr = 0.10*H
#         runs=[]; inside=False; s=0
#         for i in range(W):
#             if col[i]>=thr and not inside: inside=True; s=i
#             if (col[i]<thr or i==W-1) and inside:
#                 inside=False; e=i
#                 if e-s>4: runs.append((s,e))
#         boxes = [(s, 0, e-s, H) for (s,e) in runs]

#     boxes = sorted(boxes, key=lambda b: b[0])
#     plate_pad = plate_bgr[pad:H0-pad, pad:W0-pad]
#     char_crops = [plate_pad[y+y1:y+y1+h, x:x+w] for (x,y,w,h) in boxes]
#     return bw, boxes, char_crops

# def cleanup_if_empty(text, out_sub, base):
#     # vacío si no hay nada, o solo ?/espacios
#     clean = re.sub(r'[^A-Z0-9?]', '', (text or '').upper()).replace('?', '')
#     if len(clean) == 0:
#         # borra txt + imágenes asociadas
#         targets = [
#             os.path.join(out_sub, base + ".txt"),
#             os.path.join(out_sub, base + "_ocr.jpg"),
#             os.path.join(out_sub, base + "_bin.png"),
#         ] + glob.glob(os.path.join(out_sub, base + "_ch*.png"))
#         for fp in targets:
#             if os.path.exists(fp):
#                 os.remove(fp)
#         return True
#     return False

# # --- 2) OCR amb regles: 4 dígits + 3 lletres ---
# reader = easyocr.Reader(['en'])  # prou per digits+MAJ
# DIGITS = "0123456789"
# LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# # correccions context-dependents
# to_digit = str.maketrans({'O':'0','D':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','G':'6'})
# to_letter = str.maketrans({'0':'O','1':'I','2':'Z','5':'S','6':'G','8':'B'})

# def ocr_char(img, allow):
#     # prepara imatge: blanc de fons, negre de text, mida fixa
#     g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     g = cv2.resize(g, (44,64), interpolation=cv2.INTER_CUBIC) #Fem una mida fixa per totes les imatges
#     _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Convertim la imatge a blanc i negre amb el threshold, el otsu ens troba el threshold perfecte el binary ens passsa valorsa  0  i 255
#     b = 255 - b
#     txt = reader.readtext(b, detail=0, allowlist=allow)
#     return (txt[0] if txt else "")

# def recognize_plate(plate_bgr):
#     _, boxes, chars = segment_plate_chars(plate_bgr)

#     # si no tenim al menys 6 crops, prova OCR global
#     if len(chars) < 6:
#         txt = reader.readtext(plate_bgr, detail=0, allowlist=DIGITS+LETTERS)
#         if txt:
#             s = re.sub(r'[^A-Z0-9]','', ''.join(txt).upper())
#             # força patró ####AAA si podem
#             m = re.search(r'(\d{4})([A-Z]{3})', s)
#             if m: return f"{m.group(1)} {m.group(2)}", []
#         return "", []

#     # tria els 7 millors per alçada si hi ha soroll
#     if len(chars) > 7:
#         order = np.argsort([-c.shape[0] for c in chars])[:7]
#         chars = [chars[i] for i in sorted(order)]
#         boxes = [boxes[i] for i in sorted(order)]

#     # troba la gran separació per dividir dígits|lletres
#     gaps = []
#     for i in range(1,len(boxes)):
#         xL = boxes[i-1][0] + boxes[i-1][2]
#         xR = boxes[i][0]
#         gaps.append((xR - xL, i))
#     split = max(gaps)[1] if gaps else 4
#     # acostem-nos a 4 si la màxima no és raonable
#     if abs(split-4) > 2: split = 4 if len(boxes)>=7 else min(4, len(boxes))

#     left, right = chars[:split], chars[split:]
#     # forcem 4 i 3
#     left  = (left + right)[:4]
#     right = (left + right)[4:7]

#     # OCR amb llistes permeses
#     left_txt  = ''.join( (ocr_char(c, DIGITS) or '?') for c in left ).translate(to_digit)
#     right_txt = ''.join( (ocr_char(c, LETTERS) or '?') for c in right ).upper().translate(to_letter)

#     # neteja i força regex
#     left_txt  = re.sub(r'[^0-9]','', left_txt)[:4].ljust(4,'?')
#     right_txt = re.sub(r'[^A-Z]','', right_txt)[:3].ljust(3,'?')
#     return f"{left_txt} {right_txt}", boxes

# # --- 3) procés lot: llegeix de plates/**, guarda TXT, CSV i imatges anotades ---
# IN_DIR  = "/content/drive/MyDrive/yolo_outputs/predict_cv2/fotos_mobil/plates"
# OUT_DIR = "/content/drive/MyDrive/yolo_outputs/ocr_chars"
# os.makedirs(OUT_DIR, exist_ok=True)

# ex = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff")
# paths=[]
# for e in ex: paths += glob.glob(os.path.join(IN_DIR,"**",e), recursive=True)

# rows = []
# for p in paths:
#     img = cv2.imread(p)
#     if img is None:
#         continue

#     text, boxes = recognize_plate(img)

#     # carpeta de salida
#     rel = os.path.relpath(os.path.dirname(p), IN_DIR)
#     out_sub = os.path.join(OUT_DIR, rel)
#     os.makedirs(out_sub, exist_ok=True)
#     base = os.path.splitext(os.path.basename(p))[0]

#     # guarda el .txt primero
#     txt_path = os.path.join(out_sub, base + ".txt")
#     with open(txt_path, "w") as f:
#         f.write(text)

#     # si el texto está vacío (o solo ?), limpia y NO lo añadimos a "rows"
#     if cleanup_if_empty(text, out_sub, base):
#         continue

#     # aquí ya es válido -> añádelo a la colección para el CSV final
#     rows.append({"path": p, "plate": text})

#     # (opcional) debug: comentar si no lo quieres
#     # bw, seg_boxes, _ = segment_plate_chars(img)
#     # ann = img.copy()
#     # for (x,y,w,h) in seg_boxes:
#     #     cv2.rectangle(ann,(x,y),(x+w,y+h),(0,255,0),2)
#     # cv2.putText(ann, text, (10, max(30, ann.shape[0]-10)),
#     #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
#     # cv2.imwrite(os.path.join(out_sub, base + "_ocr.jpg"), ann)
#     # cv2.imwrite(os.path.join(out_sub, base + "_bin.png"), bw)

# # ===== al final del script: escribe los resúmenes =====
# import pandas as pd
# csv_path = os.path.join(OUT_DIR, "resultats_ocr.csv")
# txt_all  = os.path.join(OUT_DIR, "placas_validas.txt")
# txt_uni  = os.path.join(OUT_DIR, "placas_unicas.txt")

# # CSV con path y placa (solo válidas)
# df = pd.DataFrame(rows)
# df.to_csv(csv_path, index=False)

# # TXT con todas las placas (una por línea)
# with open(txt_all, "w") as f:
#     for r in rows:
#         f.write(f"{r['plate']}\t{r['path']}\n")

# # TXT con placas únicas y conteo
# from collections import Counter
# cnt = Counter([r["plate"] for r in rows])
# with open(txt_uni, "w") as f:
#     for plate, n in cnt.most_common():
#         f.write(f"{plate}\t{n}\n")

# print("Guardado:")
# print(" -", csv_path)
# print(" -", txt_all)
# print(" -", txt_uni)
