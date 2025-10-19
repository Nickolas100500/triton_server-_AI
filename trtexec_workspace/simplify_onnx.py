import onnx
import onnxsim

print('Завантаження моделі...')
model = onnx.load('yolov8n.onnx')
print('✅ ONNX модель завантажена')

print('Спрощення моделі...')
model_simp, check = onnxsim.simplify(model)
onnx.save(model_simp, 'model_simplified.onnx')
print('✅ Модель успішно спрощена')

print(f'Input: {model_simp.graph.input[0].name} - {[d.dim_value for d in model_simp.graph.input[0].type.tensor_type.shape.dim]}')
print(f'Output: {model_simp.graph.output[0].name} - {[d.dim_value for d in model_simp.graph.output[0].type.tensor_type.shape.dim]}')
print('Готово до компіляції в TensorRT!')