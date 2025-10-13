# predict_cv2_drive.py
import os, glob, argparse, shutil, time
import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def in_colab():
    try:
        import google.colab  
        return True
    except Exception:
        return False

def mount_drive_if_needed():
    if not in_colab():
        return False
    if os.path.isdir("/content/drive/MyDrive"):
        return True
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return os.path.isdir("/content/drive/MyDrive")
    except Exception as e:
        print(f"[WARN] No pude montar Drive automáticamente: {e}")
        return False

def is_in_drive(path):
    return os.path.abspath(path).startswith("/content/drive/")





def iter_paths(src):
    if os.path.isdir(src):
        for ext in IMG_EXTS:
            yield from glob.glob(os.path.join(src, f"*{ext}"))
    else:
        yield src

def get_target_class_ids(model_names, aliases=("license_plate","licence_plate","plate","matricula","placa","licence plate","license plate")):
    name2id = {str(v).strip().lower(): k for k, v in model_names.items()}
    target_ids = set()
    for a in aliases:
        a = a.strip().lower()
        if a in name2id:
            target_ids.add(name2id[a])
    return target_ids

def extract_licence_plates(img, result, save_dir, model_names):
    os.makedirs(save_dir, exist_ok=True)
    target_ids = get_target_class_ids(model_names)
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return 0

    h, w = img.shape[:2]
    count = 0
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
        if target_ids and cls_id not in target_ids:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:  continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:       continue
        out_name = f"plate_{count:02d}.jpg"
        out_path = os.path.join(save_dir, out_name)
        cv2.imwrite(out_path, crop)
        count += 1
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/content/drive/MyDrive/yolo_runs/plates_v8n/weights/best.pt",
                    help="Ruta al modelo (best.pt)")
    ap.add_argument("--source", default="/content/drive/MyDrive/fotos_matricules_mobil",
                    help="Imagen o carpeta (en Drive)")
    ap.add_argument("--outdir", default="/content/drive/MyDrive/yolo_outputs/predict_cv2/fotos_mobil",
                    help="Carpeta de salida (en Drive)")
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None, help="cpu, cuda o índice (0)")
    ap.add_argument("--show", action="store_true", help="Mostrar ventana CV2")
    args, _ = ap.parse_known_args()


    if (args.outdir.startswith("/content/drive") or args.backup_drive_dir.startswith("/content/drive")):
        mounted = mount_drive_if_needed()
        if not mounted:
            print("[WARN] No se montó Drive. Intentaré continuar, pero no se guardará en Drive.")

    os.makedirs(args.outdir, exist_ok=True)
    plates_root = os.path.join(args.outdir, "plates")
    os.makedirs(plates_root, exist_ok=True)

    print(f"[INFO] Guardando salidas en: {args.outdir}")
    model = YOLO(args.model)

    processed = 0
    for path in iter_paths(args.source):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] No pude leer: {path}")
            continue

        results = model.predict(
            source=img,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False
        )
        if not results or results[0] is None:
            print(f"[WARN] Sin resultados en: {path}")
            continue

        res0 = results[0]
        annotated = res0.plot()
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.outdir, base + "_pred.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"✅ Guardado: {out_path}")

        per_image_dir = os.path.join(plates_root, base)
        saved = extract_licence_plates(img, res0, per_image_dir, model.names)
        if saved > 0:
            print(f"📦 Matrículas guardadas: {saved} → {per_image_dir}")
        else:
            print("ℹ No se detectaron matrículas para recortar.")
        processed += 1


        os.sync() if hasattr(os, "sync") else None

        if args.show:
            cv2.imshow("Prediction", annotated)
            if cv2.waitKey(0) & 0xFF == 27:
                break

    if args.show:
        cv2.destroyAllWindows()

    print(f"Process acabat: {processed}")

if __name__ == "__main__":
    main()
