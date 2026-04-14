# Background Removal in Real Time

Реализация удаления/замены фона в реальном времени на CPU для видеопотока с веб-камеры или из файла.

## Выбранное решение
**MediaPipe Selfie Segmentation + OpenCV**

Почему выбран именно этот вариант:
- решение не требует обучения модели;
- работает локально на CPU;
- ориентировано на сегментацию человека в кадре;
- легко интегрируется с OpenCV для захвата видео, композитинга и отображения результата в реальном времени.

### Источники
- MediaPipe Selfie Segmentation: https://chuoling.github.io/mediapipe/solutions/selfie_segmentation.html
- MediaPipe package (PyPI): https://pypi.org/project/mediapipe/
- OpenCV VideoCapture basics: https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html

## Описание архитектуры
В основе решения используется **CNN-модель семейства MobileNetV3**, адаптированная для selfie segmentation. В официальном описании MediaPipe указано, что доступны две модели:
- `general` — входной размер 256×256;
- `landscape` — входной размер 144×256, она легче и работает быстрее.

В данной работе по умолчанию используется режим **`landscape`**, так как он лучше подходит под требование real-time на CPU.

### Пайплайн обработки
1. Захват кадра с камеры или из видео.
2. Запуск сегментации человека.
3. Постобработка маски:
   - пороговая бинаризация,
   - временное сглаживание (EMA),
   - Gaussian blur и median blur.
4. Композитинг кадра с одним из фонов:
   - однотонный цвет,
   - изображение,
   - размытие исходного фона.
5. Отображение результата в окне в реальном времени.
6. Подсчёт **FPS по обработанным кадрам**, а не по частоте камеры.

## Структура проекта
```text
.
├── app.py
├── benchmark.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Установка
Python: **3.10–3.12**.

```bash
python -m venv .venv
```

### Windows
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```


## Запуск
### 1. Запуск с веб-камеры
```bash
python app.py --source 0 --mode blur --width 640 --height 480 --model landscape
```

### 2. Замена фона на цвет
```bash
python app.py --source 0 --mode color --bg-color 0,255,0 --width 640 --height 480
```

### 3. Замена фона на изображение
```bash
python app.py --source 0 --mode image --bg-image assets/background.jpg --width 640 --height 480
```

### 4. Запуск на видеофайле
```bash
python app.py --source demo.mp4 --mode blur --width 640 --height 480
```

## Параметры запуска
- `--source` — индекс камеры или путь к видео.
- `--mode` — `color`, `image`, `blur`.
- `--bg-color` — цвет фона в формате `B,G,R`.
- `--bg-image` — путь к изображению для замены фона.
- `--width`, `--height` — разрешение обработки.
- `--model` — `general` или `landscape`.
- `--threshold` — порог маски.
- `--smooth-alpha` — сглаживание маски во времени.
- `--mask-blur` — размер blur для маски.
- `--blur-kernel` — сила размытия фона.
- `--infer-every` — выполнять инференс 1 раз в N кадров.

## Управление во время работы
- `q` или `Esc` — выход.
- `m` — переключение между режимами фона.
- `s` — сохранить текущий кадр.

## Замер производительности
Для честного измерения FPS по обработанным кадрам используется отдельный скрипт `benchmark.py`.

### Пример запуска benchmark
```bash
python benchmark.py --video demo.mp4 --mode blur --width 640 --height 480 --model landscape
```

Скрипт выводит:
- модель CPU,
- разрешение,
- число измеренных кадров,
- среднюю latency на кадр,
- median latency,
- средний FPS по обработанным кадрам.

## Результаты

- **Разрешение:** 640×480
- **FPS (по обработанным кадрам):** 18.59
- **Качество:** границы силуэта аккуратные; возможны артефакты на волосах и при быстром движении, но временное сглаживание уменьшает мерцание.


## Обоснование выбора решения
MediaPipe Selfie Segmentation подходит для задачи лучше тяжёлых matting-моделей, потому что:
1. запускается на CPU без отдельной подготовки модели;
2. ориентирован именно на человека в кадре;
3. обеспечивает хороший баланс качества и скорости для real-time сценария;
4. легко воспроизводится на другой машине через `pip install -r requirements.txt`.

## Демо
```text
https://drive.google.com/file/d/1Osv-QoN2c7zKkC-wGs2K22oy6AUdwq6C/view?usp=sharing
```
