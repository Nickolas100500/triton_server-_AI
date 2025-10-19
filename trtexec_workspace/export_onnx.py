
from ultralytics import YOLO

print('=== Завантаження YOLOv8n ===')
model = YOLO('yolov8n.pt')

print('=== Початок експорту в ONNX (статичний розмір 640x640, opset 17) ===')
model.export(
    format='onnx',
    imgsz=640,  # статичний розмір
    opset=17,
    dynamic=False,
    simplify=False,
    verbose=True
)

print('✅ Експорт в ONNX завершено!')