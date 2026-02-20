Pet‑проект контентной рекомендательной системы для новостной ленты на основе логов действий пользователей.
Финальный проект курса "ML engineer" от karpov.courses

Проект включает:
- подготовку фич на основе событий (просмотры, лайки);
- обучение модели CatBoost;
- генерацию оффлайн‑рекомендаций и сохранение их в PostgreSQL;
- веб‑сервис на FastAPI, который по `user_id` возвращает рекомендованные посты.

---

## Стек технологий

- Python 3.10
- CatBoost
- PostgreSQL
- SQLAlchemy + pandas
- FastAPI
- Uvicorn
- Docker (для контейнеризации сервиса)

---

## Структура проекта

```text
myenv311/
├─ app/
│  ├─ feature_upload.py   # работа с БД, генерация и загрузка фич
│  ├─ model_cv.py         # подбор гиперпараметров CatBoost (cv)
│  ├─ model_usage.py      # обучение/загрузка модели, запись рекомендаций в БД
│  ├─ schema.py           # Pydantic-схема (PostGet)
│  ├─ webserver.py        # FastAPI веб‑сервис /post/recommendations/
│  └─ __init__.py
├─ models/
│  └─ catboost_model.cbm  # обученная модель CatBoost (локально, в .gitignore)
├─ requirements.txt       # зависимости проекта
├─ Dockerfile             # сборка Docker-образа с веб‑сервисом
├─ .gitignore
└─ README.md
```

---

## Данные и признаки

Исходные таблицы в PostgreSQL:

- `feed_data` — логи действий пользователей (`timestamp`, `user_id`, `post_id`, `action`, `target`, ...);
- `user_data` — данные о пользователях (пол, возраст, город, страна, платформа, экспериментальная группа и т.п.);
- `post_text_df` — тексты постов и их тематика (`post_id`, `text`, `topic`).

### Сгенерированные признаки

В `app/feature_upload.py` формируются и сохраняются признаки в таблицу
`pavel_golovin_data_for_training`. Примеры фич:

- `hour` — час взаимодействия;
- `is_weekend` — выходной или нет;
- `user_views` — количество просмотренных постов пользователем;
- `user_ctr` — отношение лайков к просмотрам пользователя;
- `post_ctr` — CTR поста по всем пользователям;
- `topic_popularity` — популярность темы (кол-во действий по topic);
- `user_topic_ctr` — CTR пользователя по конкретной теме (`user_id`, `topic`).

После генерации фич данные сохраняются в таблицу `pavel_golovin_data_for_training`.

---

## Обучение модели

В `app/model_cv.py` подбираются гиперпараметры CatBoost (через `cv`) и выбирается лучшая конфигурация.

В `app/model_usage.py`:

- загружаются подготовленные фичи (`load_features()` из `feature_upload.py`);
- обучается/загружается модель `CatBoostClassifier`;
- модель сохраняется в `models/catboost_model.cbm`;
- затем она используется для оффлайн‑предсказаний.


---

## Генерация оффлайн‑рекомендаций

Сценарий в `app/model_usage.py`:

1. Загружается обученная модель `catboost_model.cbm`.
2. Из таблицы `pavel_golovin_data_for_training` выбираются строки, где `action != 'like'`
   (убираем уже лайкнутые посты).
3. Для них считаются предсказания модели (и/или вероятности).
4. Формируется таблица с рекомендациями:
   - `user_id`
   - `post_id`
   - `user_topic_ctr`
   - `prob` (вероятность лайка по модели)
5. Результат записывается в отдельную таблицу `pavel_golovin_data_for_recommendations`
   в PostgreSQL.

Эта таблица уже содержит «готовые» рекомендации, которые веб‑сервис только читает.

---

## Как запустить локально

### 1. Клонирование и окружение

```bash
git clone https://github.com/pavmil/recsys.git
cd recsys/myenv311

python -m venv .venv
\\Scripts\\activate  # для Windows
# source /bin/activate  # для Linux/Mac

pip install -r requirements.txt
```

### 2. Настройка доступа к PostgreSQL

В коде подключения (`app/feature_upload.py`) сейчас строка подключения зашита явно:

```python
from sqlalchemy import create_engine


ENGINE = create_engine(
	"postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
	"postgres.lab.karpov.courses:6432/startml"
)
```

Если используете свою БД — замените строку подключения на свою.

### 3. Генерация фич и обучение (опционально)

Если нужно полностью воспроизвести пайплайн:

- запустить скрипт генерации фич (`feature_upload.py`),
- обучить модель и сохранить её (`model_usage.py` / `model_cv.py`),
- сгенерировать таблицу `pavel_golovin_data_for_recommendations`.

Для демонстрации сервиса достаточно, чтобы в БД уже была таблица
`pavel_golovin_data_for_recommendations` и `post_text_df`.

### 4. Запуск веб‑сервера

```bash
uvicorn app.webserver:app --reload
```

После запуска:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Пример запроса:

```bash
curl "http://127.0.0.1:8000/post/recommendations/?id=3225&time=2021-12-20T12:00:00&limit=5"
```

---

### Сборка и запуск

В корне `myenv311` есть `Dockerfile`

```bash
cd myenv311
docker build -t recsys-api .
docker run -p 8000:8000 recsys-api
```

После этого сервис будет доступен по адресу:

- <http://localhost:8000/docs>

Если захочешь вынести строку подключения к БД в переменную окружения, можно будет добавить в Dockerfile:

```dockerfile
ENV DATABASE_URL=postgresql://user:pass@host:port/dbname
```

и читать её в `feature_upload.py` через `os.getenv("DATABASE_URL", <default>)`.

---

## Результаты и метрики

- Обучение проводилось на логах за последние 2 недели (87% активных пользователей), т.к. нехватило мощностей на большее.
- Целевая метрика: AUC для предсказания лайка (`target`).
- Лучшая модель: CatBoostClassifier с гиперпараметрами (пример):
  - `iterations=1500`
  - `learning_rate=0.4`
  - `depth=4`
  - `loss_function=CrossEntropy`
  - `eval_metric=AUC`
- Модель сохраняется в `models/catboost_model.cbm` и используется для оффлайн‑предсказаний.

До валидации параметров и подбора признаков AUC был 0.55
После - AUC = 0.89


---

## Идеи для доработки / использования в проде

- Онлайн‑обновление рекомендаций и дообучение модели.
- ML‑pipeline с использованием Airflowt.
- Фича‑стор (например, с использованием Feast).
- A/B‑тестирование разных моделей или наборов фич.

