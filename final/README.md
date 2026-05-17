# Video Highlight Detection on MR.HiSum

## Описание проекта

Проект посвящён задаче автоматического поиска хайлайтов в видео (`video highlight detection`) на датасете MR.HiSum.

Цель модели — предсказывать важность временных сегментов видео (`gtscore`) и выделять наиболее значимые моменты ролика.

Проект реализует полный воспроизводимый ML-пайплайн:
- загрузка данных;
- preprocessing;
- train / validation / test split;
- обучение temporal-модели;
- evaluation;
- ranking metrics;
- early stopping;
- сохранение результатов и графиков.

---

# Датасет

Используется датасет **MR.HiSum**.

Особенности:
- 31 892 видео;
- weak supervision через YouTube Most Replayed;
- временные importance scores;
- benchmark для задач video highlight detection и summarization.

Используемые поля:
- `gtscore`
- `gt_summary`
- `change_points`

---

# Ограничения данных

Доступная версия `mr_hisum.h5` содержала:
- `gtscore`
- `gt_summary`
- `change_points`

но не содержала готовых visual embeddings (`features`).

Поэтому был реализован fallback-подход с temporal positional features:
- нормализованная временная позиция;
- sinusoidal temporal encoding;
- информация о длине видео.

Это позволило протестировать полный ML pipeline на реальных видео MR.HiSum.

---

# Архитектура пайплайна

```text
MR.HiSum h5
        ↓
loading gtscore + temporal features
        ↓
train / validation / test split
        ↓
Temporal BiLSTM model
        ↓
importance score prediction
        ↓
ranking metrics + top-k highlight selection
```

---

# Структура проекта

```text
final_project_highlights/
│
├── checkpoints/        # сохранённые модели
├── configs/            # config.yaml
├── data/               # MR.HiSum h5 (не загружается в GitHub)
├── notebooks/          # Colab notebooks
├── outputs/            # графики и результаты
├── reports/            # отчёт и материалы защиты
├── src/                # исходный код
│
├── README.md
└── requirements.txt
```

---

# Используемые библиотеки

- Python 3.11
- PyTorch
- NumPy
- pandas
- matplotlib
- h5py
- scikit-learn

---

# Обучение модели

Используемая архитектура:
- Temporal BiLSTM.

Реализованы:
- train / validation / test split;
- early stopping;
- ranking evaluation;
- reproducible config.

---

# Метрики качества

Используемые метрики:
- MSE
- MAE
- Spearman correlation
- Kendall correlation
- MAP@15
- MAP@50
- Precision@K
- Recall@K
- F1@K

---

# Финальные результаты

## Test metrics

| Metric | Value |
|---|---|
| MSE | 0.0686 |
| MAE | 0.2162 |
| Spearman | 0.1655 |
| Kendall | 0.1178 |
| MAP@15 | 0.2455 |
| F1@15 | 0.1631 |
| MAP@50 | 0.5954 |
| F1@50 | 0.5425 |

---

# Интерпретация результатов

Несмотря на отсутствие visual embeddings, модель смогла:
- извлекать temporal structure видео;
- стабильно обучаться;
- демонстрировать ненулевые ranking metrics;
- корректно выделять top-k сегменты.

Основным ограничением качества является отсутствие visual semantic features.

---

# Графики

В папке `outputs/` находятся:
- `train_loss.png`
- `val_mse.png`
- `val_spearman.png`
- `val_map50.png`

---

# Установка зависимостей

```bash
pip install -r requirements.txt
```

---

# Запуск обучения

```bash
python src/train.py --config configs/config.yaml
```

---

# Воспроизводимость

Для воспроизводимости:
- фиксируются random seeds;
- используется `config.yaml`;
- реализован единый train pipeline;
- структура проекта разделена на train / outputs / reports.

---

# Возможные улучшения

Дальнейшее улучшение проекта возможно за счёт:
- использования реальных YouTube-8M visual embeddings;
- attention-архитектур;
- multimodal fusion;
- transformer-based temporal models;
- использования raw video frames.

---