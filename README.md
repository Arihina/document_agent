# Document Agent (OCR + LLM)

Агент платформы: пользователь прикрепляет документ (PDF/DOCX/изображение) и
задаёт вопрос, MinerU разбирает документ в markdown, Ollama отвечает по его
содержимому. Полноценный диалог — сессии, история, продолжение разговора.

Реализует канонический контракт `master_node`: `transport="contract"`,
`capabilities={"chat", "documents", "ocr"}`, `routable=True`.

## Подготовка перед запуском

MinerU и Ollama — отдельные процессы, агент к ним только стучится по HTTP,
сам их не поднимает и не устанавливает.

```bash
# MinerU (опционально)
python3 -m venv venv-mineru
source venv-mineru/bin/activate
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"
mineru-api --host 127.0.0.1 --port 8010

# Ollama
curl -fsSl https://ollama.com/install.sh | sh
ollama pull qwen3.6:35b
```

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
sudo docker compose up -d --build
sudo docker compose exec mineru mineru-models-download -s huggingface -m pipeline
sudo docker compose restart mineru

alembic upgrade head
```

```bash
python3 main.py
```

> Схему ведёт только Alembic — приложение не создаёт таблицы при старте.
> `alembic upgrade head` обязателен перед первым запуском.

## Аутентификация

Сервис не управляет пользователями — это задача платформы (мастер-агент +
Keycloak). Агент получает UUID пользователя в заголовке `X-User-Id` и
использует его как скоуп для своих данных. Заголовок обязателен **во всех**
запросах:

```
X-User-Id: 11111111-1111-1111-1111-111111111111
```

JWT валидирует мастер-агент; агент доверяет внутреннему трафику (закрыт
снаружи в обход мастера). При переходе на валидацию JWT по JWKS Keycloak
меняется только `get_user_id`, эндпоинты не затрагиваются.

| Ситуация | Код |
|----------|-----|
| Заголовок `X-User-Id` отсутствует | `401` |
| `X-User-Id` не является валидным UUID | `401` |
| Обращение к чужому чату/сообщению | `404` |
| Отсутствует или не строка `message` | `422` |
| Файл больше 25 МБ | `413` |
| Неподдерживаемый `Content-Type` тела | `415` |

Возврат `404` (а не `403`) для чужих объектов сознателен: сервис не
подтверждает их существование.

## База данных

PostgreSQL 16, в Docker Compose (`docker-compose.yaml`). Подключение — через
`.env`:

```env
DB_HOST=localhost
DB_PORT=5436
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=ocr_llm_db
```

Миграции — Alembic:
```bash
alembic upgrade head
```

> Все идентификаторы — **UUID**, генерируются приложением (`default=uuid4`),
> кроме `message_feedback.id` (`SERIAL`, идентификатор самой записи фидбэка,
> наружу не используется). Порядок сообщений — по `created_at`, не по `id`.

```
chat_sessions
├── id          UUID          PK, генерируется приложением (uuid4)
├── user_id     UUID          NOT NULL, индексирован — владелец чата
├── title       VARCHAR(255)  nullable (подставляется из первого вопроса)
├── created_at  TIMESTAMPTZ
└── updated_at  TIMESTAMPTZ

chat_messages
├── id          UUID          PK, генерируется приложением (uuid4)
├── session_id  UUID          FK → chat_sessions.id, ON DELETE CASCADE
├── role        VARCHAR(16)   "user" | "assistant"
├── content     TEXT
├── sources     JSONB         [filename], если ответ использовал документ
└── created_at  TIMESTAMPTZ

message_feedback
├── id          SERIAL        PK (запись фидбэка, не используется в API)
├── message_id  UUID          FK → chat_messages.id (уникальный)
├── vote        INTEGER       1 = лайк / -1 = дизлайк / NULL = без оценки
├── comment     TEXT          nullable
├── created_at  TIMESTAMPTZ
└── updated_at  TIMESTAMPTZ

documents
├── id                 UUID    PK, генерируется приложением (uuid4)
├── session_id         UUID    FK → chat_sessions.id, ON DELETE CASCADE
├── message_id         UUID    FK → chat_messages.id, ON DELETE CASCADE, UNIQUE
├── original_filename  VARCHAR(255)
├── mime_type          VARCHAR(127)  nullable
├── size_bytes         BIGINT
├── content            BYTEA         сам файл, хранится в БД (не на диске)
├── status             VARCHAR(16)   pending|processing|done|failed
├── markdown_content   TEXT          nullable, результат MinerU
├── ocr_backend        VARCHAR(64)   nullable, напр. "mineru:pipeline:cyrillic"
├── error_message      TEXT          nullable
├── created_at         TIMESTAMPTZ
└── updated_at         TIMESTAMPTZ
```

Один документ на сообщение (`UNIQUE` на `message_id`). Удаление сессии или
сообщения удаляет и документ, и его содержимое — двумя независимыми путями
(ORM `cascade="all, delete-orphan"` + `ON DELETE CASCADE` в БД). Блоб файла
(`content`) не грузится при обычных запросах — колонка `deferred`, тянется
явно только там, где реально нужны байты.

## Внешние сервисы

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.6:35b

MINERU_API_URL=http://127.0.0.1:8010
MINERU_BACKEND=pipeline
MINERU_LANG=cyrillic
MINERU_TIMEOUT_SECONDS=600
```

## API

Все эндпоинты требуют заголовок `X-User-Id: <uuid>`. `{session_id}` и
`{message_id}` в путях — UUID, некорректный формат → `422`.

### Чаты

#### `POST /sessions`
Тело (опционально): `{ "title": "Название чата" }`
Ответ: `{ "id", "title", "created_at", "updated_at" }`

#### `GET /sessions`
Список чатов пользователя, новые первые (по `updated_at`).

#### `GET /sessions/{session_id}/messages`
История сообщений с фидбэком и метаданными вложенного документа:
```json
[
  {
    "id": "9c858901-...", "role": "user",
    "content": "Какая сумма в накладной?",
    "sources": [], "created_at": "...", "feedback": null,
    "document": {
      "id": "85b365de-...", "filename": "накладная.pdf",
      "status": "done", "error": null
    }
  },
  {
    "id": "1e6b7ee7-...", "role": "assistant",
    "content": "В документе сумма 1000 руб.",
    "sources": ["накладная.pdf"], "created_at": "...",
    "feedback": { "vote": 1, "comment": "точно" },
    "document": null
  }
]
```
`document` — `null`, если к сообщению файл не прикреплялся.

#### `PATCH /sessions/{session_id}`
Тело: `{ "title": "Новое название" }`

#### `DELETE /sessions/{session_id}`
Каскадно удаляет сообщения, фидбэк, документы (с содержимым). `204`.

---

### Сообщения

#### `POST /sessions/{session_id}/chat`
Принимает **два** варианта тела:

- `application/json` — вопрос без вложения:
  ```json
  { "message": "Что такое MinerU?" }
  ```
- `multipart/form-data` — вопрос с документом:
  - `message` — текст вопроса
  - `file` — файл (опционально)

Документ не обязателен на каждом ходу: если файл не прислан, но в сессии
уже есть успешно обработанный документ — он остаётся действующим контекстом
(берётся последний по времени `status=done`). Новый файл заменяет его как
активный документ сессии.

Ответ — SSE:
```
data: {"chunks": [{"text": "...", "source": "накладная.pdf", "score": 1.0}]}
data: {"token": "В"}
data: {"token": " документе"}
...
data: {"message_id": "e4d1b6a7-..."}
data: [DONE]
```
- `chunks` — первым, только если для этого ответа есть активный документ
  (аналог RAG-цитат у epoz, но источник один — сам документ, не выборка
  фрагментов; `score` всегда `1.0`, ранжирования нет)
- `token` — токены ответа
- `message_id` — UUID сохранённого ответа ассистента, для фидбэка
- `[DONE]` — конец потока

Если MinerU не смог разобрать документ — `502` **до** начала стрима;
вопрос пользователя при этом уже сохранён, а `document.status = "failed"`
с `error_message` виден в истории сообщений.

---

### Фидбэк

Идентично epoz — `POST`/`GET`/`DELETE /messages/{message_id}/feedback`,
те же поля (`vote`: `1`/`-1`/`null`, `comment`).

---

## Примеры curl

```bash
U=11111111-1111-1111-1111-111111111111

# Создать чат
curl -X POST http://127.0.0.1:8002/sessions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"title": "Накладная"}'
# -> {"id": "3fa85f64-...", ...}

SID=3fa85f64-5717-4562-b3fc-2c963f66afa6

# Вопрос с документом
curl -N -X POST http://127.0.0.1:8002/sessions/$SID/chat \
  -H "X-User-Id: $U" \
  -F "message=Какая сумма в накладной?" \
  -F "file=@накладная.pdf;type=application/pdf"

# Следующий вопрос без файла — документ выше остаётся контекстом
curl -N -X POST http://127.0.0.1:8002/sessions/$SID/chat \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"message": "А дата какая?"}'

MID=e4d1b6a7-78b5-405a-a314-06635f2ef7b9

# История
curl http://127.0.0.1:8002/sessions/$SID/messages -H "X-User-Id: $U"

# Фидбэк
curl -X POST http://127.0.0.1:8002/messages/$MID/feedback \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"vote": 1, "comment": "точно"}'

# Переименовать / удалить
curl -X PATCH http://127.0.0.1:8002/sessions/$SID \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"title": "Накладная №1"}'

curl -X DELETE http://127.0.0.1:8002/sessions/$SID -H "X-User-Id: $U"
```