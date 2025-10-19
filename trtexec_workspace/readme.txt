#вхід в контейнер 
docker exec -it trtexec bash
----------------------------------------------------------------------------------
#конвертація .onnx - .plan
trtexec --onnx=yolov8n.onnx --saveEngine=model.plan --workspace=4096
-------------------------------------------------------------------------------------
# перевірка .plan
trtexec --loadEngine=model.plan --verbose --noDataTransfers
-----------------------------------------------------------------------------------
#перевірка .onnx   
trtexec --onnx=yolov8n.onnx --verbose
-----------------------------------------------------------------------------------
#спрощеня моделі .onnx - кращої конверації  в .plan
python3 -c "import onnx; import onnxsim; print('Завантаження моделі...'); model = onnx.load('model.onnx'); print('✅ ONNX модель завантажена'); print('Спрощення моделі...'); model_simp, check = onnxsim.simplify(model); onnx.save(model_simp, 'model_simplified.onnx'); print('✅ Модель успішно спрощена'); print(f'Input: {model_simp.graph.input[0].name} - {[d.dim_value for d in model_simp.graph.input[0].type.tensor_type.shape.dim]}'); print(f'Output: {model_simp.graph.output[0].name} - {[d.dim_value for d in model_simp.graph.output[0].type.tensor_type.shape.dim]}'); print('Готово до компіляції в TensorRT!')"
--------------------------------------------------------------------------------------------------------------------------------------------------------
# експорт yolov8n.onnx
python3 simplify_onnx.py

# Експорт в onnx
python3 export_onnx.py

# In bash contenter trtexec
експортуєм yolov8n.onnx ->  перевіряєм -> спрощуєм  -> конвертуєм в model.plan  -> загружаєм на тритон 