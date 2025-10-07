  # запуск сервака api.py для візуальної демонстрації роботи 
   python -m uvicorn api:app --reload --port 5000 --host localhost
    #http://localhost:5000/chat

#  запуск сервера dialoGPT
python -m uvicorn chat_api:app --reload --port 4000 --host localhost
 # http://localhost:4000/chat


 # запускає teriton server conteiner 
   docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 `
   -v "D:\Coding Program\steam server  html\triton\models:/models" nvcr.io/nvidia/tritonserver:23.08-py3 `
   tritonserver --model-repository=/models

 
# щоб відкрити термінал в контейнері  docker девись в докері
  docker exec -it trtexec_container bash
    

------------------------------нейронка DialoGPT-medium запуск------------------------

перевірка які тензори виводить model.onnx на  пітон коді note.ipynb/ Провірка  вхідних вихідних тензорів 

 щоб написати під модель config.pbtxt потрібно  запустити  в логах найти інфу про модель 

  docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 `
>> -v "d:/Coding Program/Triton invidia/triton/models:/models" `
>> nvcr.io/nvidia/tritonserver:23.08-py3 tritonserver `
>> "--model-repository=/models" "--log-verbose=1" 

ВАЖЛИВО ЩОБ МОДЕЛЬ БУЛА ПРАВЕЛЬНО ЗКОМПІЛЬОВАНА І ПРОПИСАНА В  config.pbtxt

dialoGPT  експортувати .onnx потрібно встановити фіксований  batching   я встановив [1,7]
клієнті робити pading під стартові значення (не забути обновити config.pbtxt) 
використвував спеціальні токени DialoGPT і потрібно побавитись з фільтрацією кривих відповідей .
і часту генерувала повторювані токени іза того що обрізаєтся до 7 токенів , краще 
зробити це денамічную довженною токненів но це для ugrade на майбутнє.

потрібно переписати config.pbtxt  i torch.onnx.export() під денамічную довженною токненів

потрібно зробити refactoring коду так як він дуже довгий 


-----------------------------------нейронка RizNet -----------------------------------------------







 ------------------------------ Силки на мої проекти YouTube----------------------------------





 ютуб силка роботи yolov8.engine  на jetson nano
 https://youtu.be/uc21cH0cDNo

 ютуб  силка на роботу міні турелі 
 https://youtu.be/6AvGxFFgK4Y





ідея дял насткпного проеукту :
нейронки
1. yolo бачить обєкти  на зображенні 
2. описує  що вона бачтить на зображені 
3. приймає рішен з багатьма змінними 

дати дупля як працює DialoGPT 
розібратися з pytorch B RizNet

попробувати запустити pfgecnbnb YOLO.plan
(перевстановити tensorRT 8.5)

