#вхід в контейнер 
docker exec -it trtexec bash
----------------------------------------------------------------------------------
#конвертація .onnx - .plan
trtexec --onnx=model.onnx --saveEngine=model.plan --workspace=4096 --fp16
-------------------------------------------------------------------------------------
# Після компіляції перевірити
trtexec --loadEngine=model.plan --verbose --noDataTransfers
-----------------------------------------------------------------------------------
#перевірка .onnx  модель 
trtexec --onnx=model.onnx --verbose
-----------------------------------------------------------------------------------
#спрощеня моделі .onnx - кращої конверації  в .plan
python3 -c "
import onnx
import onnxsim

print('Завантаження моделі...')
model = onnx.load('model.onnx')
print('✅ ONNX модель завантажена')

print('Спрощення моделі...')
model_simp, check = onnxsim.simplify(model)
onnx.save(model_simp, 'model_simplified.onnx')
print('✅ Модель успішно спрощена')

print(f'Input: {model_simp.graph.input[0].name} - {[d.dim_value for d in model_simp.graph.input[0].type.tensor_type.shape.dim]}')
print(f'Output: {model_simp.graph.output[0].name} - {[d.dim_value for d in model_simp.graph.output[0].type.tensor_type.shape.dim]}')
print('Готово до компіляції в TensorRT!')
"
--------------------------------------------------------------------------------------------------------------------------------------------------------
# експорт yolov8n.onnx
python3 -c "
from ultralytics import YOLO

print('=== Завантаження YOLOv8n ===')
model = YOLO('yolov8n.pt')

print('=== Початок експорту в TensorRT ===')
model.export(
    format='engine',
    device=0,
    half=True,
    workspace=4,
    verbose=True
)

print('✅ Експорт завершено!')
"

