from os import PathLike

model = YOLO('yolov8n.pt')

model.train(
    data=DATA_YAML,
    epochs=15,
    patience=15,
    imgsz=640,
    device=0,        
    workers=2,       
    batch=16, 
    project=SAVE_ROOT,
    name="plates_v8n",
    exist_ok=True
)
