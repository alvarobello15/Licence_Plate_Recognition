from os import PathLike

model = YOLO('yolov8n.pt')

model.train(
    data=DATA_YAML,
    epochs=15,
    patience=15,
    imgsz=640,
    device=0,        # fuerza GPU
    workers=2,       # 2 suele ir bien en Colab
    batch=16, # ajusta según VRAM (8–32)
    project=SAVE_ROOT,
    name="plates_v8n",
    exist_ok=True
)
