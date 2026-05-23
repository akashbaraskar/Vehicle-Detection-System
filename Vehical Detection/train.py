from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="car damage",
    epochs=20,
    imgsz=224,
    batch=16,
    name="damage_model"
)

# Optional validation
model.val()