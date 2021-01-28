# DL deployment instruments contest by Huawei

# Task
- Take any modern object/keypoint detection model
- Deploy to the target device by using any IE discussed in the lecture
- Document and describe all the steps and challenges (if you faced any)
- Use all the possible selected IE knobs to get the best performance (for latency or throughput scenarios, or both if applicable)
- Document your performance findings 
- Share your notes and code over github repo
- Small app demonstrating some use case is a plus

# Зависимости
- PyTorch
- OpenCV
- TensorRT
- torch2trt

# Структура проекта
- `data` - директория, которая содержит вспомогательные файлы для
работы проекта
- `src` - исходный код проекта
    - `models` - директория, которая содержит реализацию архитектур
    моделей машинного обучения
    - `utils` - набор скриптов, в которых реализован некоторый 
    вспомогательный функционал проекта
    - `benchmarking.py` - скрипт для проведения сравнительного 
    анализа точности предсказаний модели
    - `conversion_tensorrt.py` - скрипт для автоматической
    конвертации модели из PyTorch в TensorRT
    - `inference.py` - скрипт, который запускает демонстрационное
    приложение
- `config.yaml` - конфигурационный файл с параметрами запуска

# Запуск
## Демо
Для запуска демонстрационного приложения необходимо в файле 
`config.yaml` изменить следующие параметры:
- `weights_path` - путь до весов обученной модели
- `video_path` - путь до видео, на котором будет запускаться приложение
- `use_fp16_mode` - (ОПЦИОНАЛЬНО) установить значение `false`, если
используются `fp32` веса
- `use_tensorrt` - (ОПЦИОНАЛЬНО) установить значение `true`, если
используются веса, сконвертированные для `TensorRT`

После чего ввести в терминале следующую команду:
```shell script
python src/inference.py
```

## Бенчмаркинг
Для запуска бенчмаркинга необходимо в файле `config.yaml` изменить
следующие параметры:
- `weights_path` - путь до весов обученной модели
- `coco_data_path` - путь до директории, в которой находится датасет
`COCO 2017 Dataset`
- `use_fp16_mode` - (ОПЦИОНАЛЬНО) установить значение `false`, если
используются `fp32` веса
- `use_tensorrt` - (ОПЦИОНАЛЬНО) установить значение `true`, если
используются веса, сконвертированные для `TensorRT`
- `eval_batch_size` - (ОПЦИОНАЛЬНО) размер батча

После чего ввести в терминале следующую команду:
```shell script
python src/benchmarking.py
```
