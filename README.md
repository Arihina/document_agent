# Document Agent (OCR + LLM)

Агент платформы: пользователь загружает документ (PDF/DOCX/изображение)
через отдельную ручку, MinerU разбирает его в markdown, дальше документ
можно подключать к любому количеству вопросов явной ссылкой на его `id` —
без диалоговой памяти о «последнем загруженном файле». Плюс полноценный
диалог: чаты, история, продолжение разговора.

API совместим с OpenAI Chat Completions (`/v1/chat/completions`) и OpenAI
Files API (`/v1/files`).

Реализует канонический контракт `master_node`: `transport="contract"`,
`capabilities={"chat", "documents"}`, `routable=True`.

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
| Обращение к чужому completion'у, чату (`conversation`) или файлу | `404` |
| Файл больше 25 МБ | `413` |
| Файл ещё не обработан (`status != "done"`), а на него сослались в чате | `422` |

Возврат `404` (а не `403`) для чужих объектов сознателен: сервис не
подтверждает их существование.

## API — общая идея

Три независимые части:

- **`/v1/chat/completions`** — OpenAI-совместимый эндпоинт генерации.
  **Полностью stateless**: клиент присылает всю историю в `messages[]` при
  каждом запросе, сервер её не хранит и не переиспользует между вызовами.
- **`/v1/files`** — OpenAI Files API. Загрузка документа отдельным вызовом
  (MinerU-разбор происходит здесь, синхронно), дальше файл — независимый,
  переиспользуемый ресурс: подключается к любому вопросу явной ссылкой на
  `id`, а не автоматически «текущим документом чата».
- **`/v1/platform/conversations`** — платформенное (не входящее в
  OpenAI-стандарт) расширение для UI: список чатов, история, переименование,
  удаление. Не участвует в генерации и не хранит контекст для модели — это
  только группировка сообщений для отображения.

**Документ не «помнится» между ходами диалога сам по себе.** Раньше файл,
приложенный к одному сообщению, автоматически оставался «активным
документом» для всех следующих вопросов в той же сессии, пока не пришёл
новый файл. Сейчас такой магии нет: документ входит в контекст конкретного
вызова `/v1/chat/completions` **только** если клиент явно сослался на его
`file_id` в `content` текущего сообщения — на следующем ходу, если документ
всё ещё нужен, ссылку нужно передать снова. Это не ограничение реализации,
а то же самое поведение, что и у мультимодального контента в настоящем
OpenAI API: файл/изображение из истории не «помнится» моделью между
вызовами, его нужно включать в `messages[]` каждый раз, когда он должен
быть в контексте — сам LLM-вызов ничего не хранит между запросами. Именно
поэтому `conversation_id` — просто ярлык для группировки чата в UI и не
участвует в сборке контекста ни при каких условиях.

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
conversations                    платформенная сущность, только для UI/истории
├── id          UUID          PK, генерируется приложением (uuid4)
├── user_id     UUID          NOT NULL, индексирован — владелец чата
├── title       VARCHAR(255)  nullable (подставляется из первого вопроса)
├── created_at  TIMESTAMPTZ
└── updated_at  TIMESTAMPTZ

chat_messages
├── id                 UUID          PK, генерируется приложением (uuid4);
│                                    у ассистентских сообщений совпадает с
│                                    частью после "chatcmpl-" в id completion'а
├── user_id            UUID          NOT NULL, индексирован — владелец записи
├── conversation_id    UUID          nullable, FK → conversations.id (ON DELETE CASCADE) —
│                                    НЕ участвует в сборке контекста, только
│                                    привязка к чату в UI-списке
├── role                VARCHAR(16)   "user" | "assistant"
├── content             TEXT
├── sources             JSONB         [filename], если ответ использовал файл (для отображения)
├── model               VARCHAR(64)   значение "model" из запроса
├── prompt_tokens       INTEGER       nullable
├── completion_tokens   INTEGER       nullable
└── created_at          TIMESTAMPTZ

message_feedback
├── id          SERIAL        PK (запись фидбэка, не используется в API)
├── message_id  UUID          FK → chat_messages.id (уникальный)
├── vote        INTEGER       1 = лайк / -1 = дизлайк / NULL = без оценки
├── comment     TEXT          nullable
├── created_at  TIMESTAMPTZ
└── updated_at  TIMESTAMPTZ

files                             НЕЗАВИСИМЫЙ ресурс — без FK на conversations/chat_messages
├── id                 UUID          PK, генерируется приложением (uuid4)
├── user_id            UUID          NOT NULL, индексирован — владелец файла
├── filename           VARCHAR(255)
├── mime_type          VARCHAR(127)  nullable
├── size_bytes         BIGINT
├── content             BYTEA         сам файл, хранится в БД (не на диске)
├── status              VARCHAR(16)   pending|processing|done|failed
├── markdown_content    TEXT          nullable, результат MinerU
├── ocr_backend         VARCHAR(64)   nullable, напр. "mineru:pipeline:cyrillic"
├── error_message       TEXT          nullable
├── created_at          TIMESTAMPTZ
└── updated_at          TIMESTAMPTZ
```

Блоб файла (`content`) не грузится при обычных запросах — колонка
`deferred`, тянется явно только там, где реально нужны байты (передача в
MinerU). Удаление чата каскадно удаляет его сообщения и их фидбэк; удаление
файла — только сам файл (он ни от чего не зависит и ни на что не ссылается).

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

Все эндпоинты требуют заголовок `X-User-Id: <uuid>`. Ресурсы скоупятся по
этому идентификатору; обращение к чужому ресурсу возвращает `404`.

### `POST /v1/files`

Загрузить документ. `multipart/form-data`, поле `file`. MinerU разбирает
документ **синхронно** внутри этого вызова; таймаут — `MINERU_TIMEOUT_SECONDS`
(до 600 секунд по умолчанию — клиент ждёт весь разбор в одном HTTP-вызове).

Ответ `201` — объект в формате OpenAI Files API + нестандартные поля
статуса разбора:
```json
{
  "id": "file-85b365de-1234-4c7d-8e9f-0a1b2c3d4e5f",
  "object": "file",
  "bytes": 245678,
  "created_at": 1735900000,
  "filename": "накладная.pdf",
  "purpose": "assistants",
  "status": "done",
  "status_details": null
}
```
| Поле | Описание |
|---|---|
| `id` | `file-<uuid>` — используется дальше в `/v1/chat/completions` и в путях `/v1/files/{id}` |
| `bytes` | размер файла в байтах |
| `purpose` | всегда `"assistants"` — единственное используемое значение сейчас |
| `status` | `done` \| `failed` (платформенное расширение, не из спеки OpenAI) |
| `status_details` | текст ошибки MinerU, если `status: "failed"`; иначе `null` |

Если MinerU не смог разобрать документ — ответ `502`:
```json
{"error": {"message": "Не удалось обработать документ: ...", "type": "server_error", "param": null, "code": null}}
```
Файл в БД при этом остаётся со `status: "failed"` и `error_message` (для
отладки), но клиенту доступен только текст ошибки — файл с таким `id`
дальше использовать нельзя (в контекст чата попадёт `404`/`422`, см. ниже).

---

#### `GET /v1/files`
Список загруженных файлов пользователя, новые первые.
```json
{
  "object": "list",
  "data": [
    {"id": "file-85b365de-...", "object": "file", "bytes": 245678, "created_at": 1735900000,
     "filename": "накладная.pdf", "purpose": "assistants", "status": "done", "status_details": null}
  ]
}
```

---

#### `GET /v1/files/{file_id}`
Метаданные и статус одного файла — тот же формат объекта, что у `POST /v1/files`.

---

#### `DELETE /v1/files/{file_id}`
```json
{"id": "file-85b365de-1234-4c7d-8e9f-0a1b2c3d4e5f", "object": "file", "deleted": true}
```

---

### `POST /v1/chat/completions`

Генерация ответа. Клиент присылает **всю** историю диалога в `messages[]` —
сервис её не хранит и не переиспользует между запросами.

Тело запроса:
```json
{
  "model": "document_chat",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Какая сумма в накладной?"},
        {"type": "file", "file": {"file_id": "file-85b365de-1234-4c7d-8e9f-0a1b2c3d4e5f"}}
      ]
    }
  ],
  "stream": true,
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```
| Поле | Тип | Обязательно | Описание |
|---|---|---|---|
| `model` | string | нет (по умолчанию `"document_chat"`) | не влияет на поведение, только эхом в ответе |
| `messages` | array | да | последнее сообщение — `role: "user"`, это и есть текущий вопрос |
| `stream` | bool | нет (по умолчанию `false`) | стримить ответ через SSE |
| `conversation_id` | UUID string | нет | платформенное расширение — привязать сообщение к чату из `/v1/platform/conversations`. Чужой/несуществующий `conversation_id` → `404` |

`content` у сообщения с `role: "user"` может быть:
- обычной строкой — вопрос без документа;
- массивом content-частей — `{"type": "text", "text": "..."}` (текст вопроса)
  и не более одной `{"type": "file", "file": {"file_id": "file-..."}}`
  (ссылка на ранее загруженный документ). Файл должен принадлежать
  вызывающему и иметь `status: "done"` — иначе `404` (файл не найден/чужой)
  или `422` (`status` не `"done"`).

Ссылка на файл действует **только для этого сообщения** — см. «API — общая
идея» выше про отсутствие памяти о документе между ходами.

**Нестрим-ответ** (`chat.completion`):
```json
{
  "id": "chatcmpl-1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
  "object": "chat.completion",
  "created": 1735900000,
  "model": "document_chat",
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "В накладной сумма 1000 руб."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 1204, "completion_tokens": 18, "total_tokens": 1222}
}
```

**Стрим-ответ** (`stream: true`) — SSE, `chat.completion.chunk`, один и тот
же `id` во всех чанках:
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1735900000,"model":"document_chat","conversation_id":"3fa85f64-...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1735900000,"model":"document_chat","conversation_id":"3fa85f64-...","choices":[{"index":0,"delta":{"content":"В"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1735900000,"model":"document_chat","conversation_id":"3fa85f64-...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```
`conversation_id` в ответе — `null`, если не был передан в запросе. `id`
(`chatcmpl-<uuid>`) — ключ для повторного чтения и фидбэка ниже. Если в
сообщении был `file_id`, в сохранённом ответе поле `sources` = `[filename]`
(видно в истории чата, см. `/v1/platform/conversations` ниже).

---

### `GET /v1/chat/completions/{completion_id}`

Получить ранее сгенерированный ответ повторно — по `id`, который пришёл в
`POST`-ответе (`chatcmpl-<uuid>` целиком или голый UUID). Ответ — тот же
`chat.completion`-объект, что и у `POST` в нестрим-режиме, восстановленный
из БД (включая `usage`).

`404`, если `id` не найден, принадлежит не вам, либо указывает на сообщение
с `role: "user"` (в норме такой `id` клиенту никогда не отдаётся).

---

### `POST/GET/DELETE /v1/chat/completions/{completion_id}/feedback`

Оценка ответа ассистента. `{completion_id}` — значение `id` из ответа
(`chatcmpl-<uuid>` целиком или голый UUID — оба варианта принимаются).

Тело `POST`-запроса (все поля опциональны, повторный вызов обновляет
существующую оценку):
```json
{ "vote": 1, "comment": "Очень подробно" }
```
| Поле | Тип | Значения |
|---|---|---|
| `vote` | integer | `1` — лайк, `-1` — дизлайк, `null`/отсутствует — без оценки |
| `comment` | string | любой текст, опционально |

Ответ (`POST`/`GET`):
```json
{
  "message_id": "1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
  "vote": 1,
  "comment": "Очень подробно",
  "created_at": "2026-08-03T12:01:00",
  "updated_at": "2026-08-03T12:01:00"
}
```
`DELETE` сбрасывает оценку (`vote=null`, `comment=null`) и возвращает `204`.
Оценивать можно только сообщения ассистента — `400` при попытке оставить
фидбэк на сообщение пользователя.

---

### Чаты — `/v1/platform/conversations`

Платформенное расширение для UI (список чатов, история, переименование,
удаление). **Не входит в OpenAI-стандарт** и не участвует в генерации — см.
«API — общая идея» выше.

#### `POST /v1/platform/conversations`
Создать новый чат.

Тело запроса (опционально): `{ "title": "Название чата" }`

Ответ `201`:
```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "title": "Название чата",
  "created_at": "2026-08-03T12:00:00",
  "updated_at": "2026-08-03T12:00:00"
}
```

#### `GET /v1/platform/conversations`
Список чатов текущего пользователя, отсортированных по дате последнего
сообщения (новые первые). Формат элемента — как у `POST`.

#### `GET /v1/platform/conversations/{id}/messages`
История сообщений чата вместе с фидбэком — для восстановления `messages[]`
на фронте при открытии чата.
```json
[
  {
    "id": "9c858901-8a57-4791-81fe-4c455b099bc9",
    "role": "user",
    "content": "Какая сумма в накладной?",
    "sources": [],
    "created_at": "2026-08-03T12:00:00",
    "feedback": null
  },
  {
    "id": "1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
    "role": "assistant",
    "content": "В накладной сумма 1000 руб.",
    "sources": ["накладная.pdf"],
    "created_at": "2026-08-03T12:00:05",
    "feedback": {"vote": 1, "comment": "точно"}
  }
]
```
`id` ассистентского сообщения — тот же UUID, что использовать для
`/v1/chat/completions/{id}/feedback` и повторного чтения. Обратите внимание:
исходный `content` пользовательского сообщения здесь — только текстовая
часть вопроса; сама ссылка на файл (`file_id`) в истории не хранится как
структурная часть сообщения — она видна косвенно, через `sources` в ответе
ассистента.

#### `PATCH /v1/platform/conversations/{id}`
Переименовать чат. Тело: `{ "title": "Новое название" }`.

#### `DELETE /v1/platform/conversations/{id}`
Удалить чат со всеми сообщениями и их фидбэком (каскадно). Ответ: `204 No Content`.

## Ошибки

Единый формат вместо FastAPI-дефолта `{"detail": ...}`:
```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}
```
`type` — грубая классификация по HTTP-статусу: `400/413/415/422` →
`invalid_request_error`, `401` → `authentication_error`, `404` →
`not_found_error`, остальное → `server_error`.

## Примеры curl

```bash
U=11111111-1111-1111-1111-111111111111
BASE=http://127.0.0.1:8006

# --- загрузить документ ---
curl -X POST $BASE/v1/files \
  -H "X-User-Id: $U" -F "file=@накладная.pdf;type=application/pdf"
# -> {"id": "file-85b365de-...", "status": "done", ...}

FID=file-85b365de-1234-4c7d-8e9f-0a1b2c3d4e5f

# --- вопрос по документу, без привязки к чату ---
curl -N -X POST $BASE/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"document_chat\", \"stream\": true, \"messages\": [
        {\"role\": \"user\", \"content\": [
          {\"type\": \"text\", \"text\": \"Какая сумма в накладной?\"},
          {\"type\": \"file\", \"file\": {\"file_id\": \"$FID\"}}
        ]}
      ]}"

# --- следующий вопрос по тому же документу — file_id нужно передать снова ---
curl -N -X POST $BASE/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"document_chat\", \"messages\": [
        {\"role\": \"user\", \"content\": [
          {\"type\": \"text\", \"text\": \"Какая сумма в накладной?\"},
          {\"type\": \"file\", \"file\": {\"file_id\": \"$FID\"}}
        ]},
        {\"role\": \"assistant\", \"content\": \"В накладной сумма 1000 руб.\"},
        {\"role\": \"user\", \"content\": [
          {\"type\": \"text\", \"text\": \"А дата какая?\"},
          {\"type\": \"file\", \"file\": {\"file_id\": \"$FID\"}}
        ]}
      ]}"

# --- вопрос без документа ---
curl -X POST $BASE/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"model": "document_chat", "messages": [{"role": "user", "content": "Что такое MinerU?"}]}'

ID=chatcmpl-1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2

# --- получить ответ повторно ---
curl $BASE/v1/chat/completions/$ID -H "X-User-Id: $U"

# --- фидбэк ---
curl -X POST $BASE/v1/chat/completions/$ID/feedback \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"vote": 1, "comment": "точно"}'
curl $BASE/v1/chat/completions/$ID/feedback -H "X-User-Id: $U"
curl -X DELETE $BASE/v1/chat/completions/$ID/feedback -H "X-User-Id: $U"

# --- чаты (платформенный CRUD) ---
curl -X POST $BASE/v1/platform/conversations \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"title": "Накладная"}'
# -> {"id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", ...}

CID=3fa85f64-5717-4562-b3fc-2c963f66afa6

# сообщение внутри чата — conversation_id привязывает запись к нему
curl -X POST $BASE/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"document_chat\", \"conversation_id\": \"$CID\", \"messages\": [
        {\"role\": \"user\", \"content\": [
          {\"type\": \"text\", \"text\": \"Какая сумма в накладной?\"},
          {\"type\": \"file\", \"file\": {\"file_id\": \"$FID\"}}
        ]}
      ]}"

curl $BASE/v1/platform/conversations -H "X-User-Id: $U"
curl $BASE/v1/platform/conversations/$CID/messages -H "X-User-Id: $U"

curl -X PATCH $BASE/v1/platform/conversations/$CID \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"title": "Накладная №1"}'
curl -X DELETE $BASE/v1/platform/conversations/$CID -H "X-User-Id: $U"

# --- удалить файл, когда он больше не нужен ---
curl -X DELETE $BASE/v1/files/$FID -H "X-User-Id: $U"
```