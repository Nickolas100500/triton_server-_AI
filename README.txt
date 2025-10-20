========================== ЗАПУСК TRITON SERVER ==========================

# Потрібен Docker для роботи сервера
# Я працюю у Windows 10, тому команди нижче для PowerShell

docker-compose up -d


# Запуск API сервера
python -m uvicorn api:app --reload --port 5000 --host localhost
# URL: http://localhost:5000/chat


# Запуск сервера DialoGPT
python -m uvicorn chat_api:app --reload --port 4000 --host localhost
# URL: http://localhost:4000/chat


# Запуск сервера YOLO API
python -m uvicorn api_yolo:app --reload --port 2500 --host localhost


# Запуск Triton Server контейнера
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 `
  -v "D:\Coding Program\steam server  html\triton\models:/models" nvcr.io/nvidia/tritonserver:22.12-py3 `
  tritonserver --model-repository=/models


# Grafana Monitor Server
http://localhost:3000


======================== DialoGPT-medium запуск ==========================

1️⃣ Перевірка тензорів у моделі:
   - Відкрити note.ipynb
   - Перевірити вхідні та вихідні тензори model.onnx

2️⃣ Для створення `config.pbtxt`:
   - Запусти Triton з логами:
     docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 `
       -v "d:/Coding Program/Triton invidia/triton/models:/models" `
       nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver `
       "--model-repository=/models" "--log-verbose=1"

3️⃣ ВАЖЛИВО:
   - Модель повинна бути зкомпільована правильно.
   - Всі шляхи в `config.pbtxt` мають бути вірні.

4️⃣ Експорт DialoGPT у ONNX:
   - Використовуй фіксований batching → [1,7]
   - Робити padding на клієнті під стартові значення.
   - Використовуються спеціальні токени DialoGPT.
   - Часто виникає дублювання токенів → через обрізку до 7.
   - Краще зробити динамічну довжину токенів (оновити `torch.onnx.export()` і `config.pbtxt`).

5️⃣ Рефакторинг:
   - Код занадто довгий, бажано структурувати під окремі модулі.


=========================== RESNET (RizNet) ===============================

1️⃣ Експорт ResNet у формат .onnx:
   - Використати PyTorch.
   - Задати розмір зображення 256 (resize) і CenterCrop 224.

2️⃣ Preprocess:
   - Використовувати torchvision (стабільна робота).

3️⃣ Softmax:
   - Додати `nn.Softmax(dim=1)` в архітектуру перед експортом.

4️⃣ API сервер:
   - Обгорнути модель у `api.py`
   - Обов’язково використовувати асинхронні функції.
   - API використовує gRPC (порт 8001) для швидшої роботи.
   - REST (порт 8000) – повільніший, але простіший.


=========================== YOLO (ultralytics_yolo) ======================

1️⃣ Експорт YOLO в TensorRT (.plan):
   - Компілювати у правильній версії TensorRT (через офіційний NVIDIA контейнер).
   - План залежить від архітектури GPU.

2️⃣ ONNX оптимізація:
   - Спрощення ONNX перед конвертацією в TensorRT.
   - Сумісність TritonServer, TensorRT і trtexec повинна співпадати.

3️⃣ Проблеми:
   - Якщо модель не бачить `.plan`, перевірити сумісність версій.
   - Низький FPS → спробувати збільшити batch до 4.
   - Якщо GPU не завантажується повністю → перенести preprocess/postprocess на GPU.


======================== СИЛКИ НА МОЇ ПРОЕКТИ ============================

🎥 YOLOv8.engine на Jetson Nano:
https://youtu.be/uc21cH0cDNo

🎯 Міні-турель (демо):
https://youtu.be/6AvGxFFgK4Y

🧠 Triton Server у роботі:
https://youtu.be/7a1Jx6ZR8Qw


============================= СИСТЕМА ===================================

Windows 10  
NVIDIA CUDA 11.8  
GeForce GTX 1060 (6GB)  
Intel i7-7700HQ
