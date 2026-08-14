# Document Agent (OCR + LLM)

Агент платформы: пользователь загружает документ (PDF/DOCX/изображение)
через отдельную ручку, MinerU разбирает его в markdown, дальше документ
можно подключать к вопросам либо явной ссылкой на его `id`, либо (в форме
Responses, если файл был загружен в чат) автоматически. Плюс полноценный
диалог: чаты, история, продолжение разговора.

Отдаёт генерацию в **двух** формах OpenAI API одновременно — Chat
Completions (`/v1/chat/completions`) и Responses (`/v1/responses`) — плюс
OpenAI Files API (`/v1/files`).

Реализует канонический контракт `master_node`: `transport="contract"`,
`capabilities={"chat", "attachments"}`, `routable=True`.

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

### MinerU на GPU

По умолчанию контейнер MinerU считает на CPU. Бэкенд `pipeline` использует
torch (не paddle), поэтому перевод на GPU — это доступ к карте из контейнера
плюс явное указание устройства.

На хосте нужен `nvidia-container-toolkit`:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```


```bash
sudo docker compose build --no-cache mineru
sudo docker compose up -d mineru
```

Проверка, что карта реально видна процессу:

```bash
sudo docker compose exec mineru python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Если `torch.__version__` заканчивается на `+cpu` — установлена CPU-сборка,
нужен пересбор с `TORCH_INDEX_URL` под свою версию CUDA (см. `mineru/Dockerfile`).

`MINERU_VIRTUAL_VRAM_SIZE` ограничивает, сколько видеопамяти MinerU считает
доступной, и через это — размер батча.

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
| Обращение к чужому completion'у/response'у, чату (`conversation`) или файлу | `404` |
| Файл больше 25 МБ | `413` |
| Файл ещё не обработан (`processing_status != "done"`), а на него сослались в чате | `400` |
| Приложено больше `MAX_ATTACHED_FILES` файлов | `400` |
| Промпт не помещается в контекст модели (см. «Контекстное окно») | `413` |

Возврат `404` (а не `403`) для чужих объектов сознателен: сервис не
подтверждает их существование.

## API — общая идея

Четыре независимые части:

- **`/v1/chat/completions`** — форма Chat Completions. **Полностью
  stateless**: клиент присылает всю историю в `messages[]` при каждом
  запросе. Путь для быстрой интеграции сторонних клиентов/вендоров и для
  отладки.
- **`/v1/responses`** — форма Responses API. Тоже полностью рабочая сама
  по себе (весь `input` от клиента) — но при переданном `conversation_id`
  **сервис сам собирает текстовую историю из БД**, клиенту нужно прислать
  только новый ход. Это путь, которым пользуется собственный фронт
  платформы. Подробности — в разделе `POST /v1/responses` ниже.
- **`/v1/files`** — OpenAI Files API. Загрузка документа отдельным вызовом
  (MinerU-разбор происходит здесь, синхронно). Файл — независимый,
  переиспользуемый ресурс, подключается к вопросу явной ссылкой на `id` —
  это работает всегда, в обеих формах. Дополнительно, если при загрузке
  указан `conversation_id`, файл привязывается к чату и в форме Responses
  может подключаться автоматически (см. ниже). **Общий ресурс для обеих
  форм генерации.**
- **`/v1/platform/conversations`** — платформенное (не входящее ни в одну
  спеку OpenAI) расширение для UI: список чатов, история, переименование,
  удаление. Обе формы генерации читают/пишут в одни и те же
  `conversations`/`chat_messages`.

Фидбэк (`/v1/chat/completions/{id}/feedback`) — **общий для обеих форм
генерации**, не дублируется под `/v1/responses/...`.

**Подключение документа зависит от формы:**

Правило одно и то же в обеих формах — **липкий файл**:

1. Агент идёт по присланным сообщениям **с конца** и берёт первое, в котором
   есть ссылки на файлы. Все файлы этого сообщения становятся рабочим
   набором. Накопления по всей истории нет: приложили новый документ —
   набор сменился на него.
2. Если ссылок нет нигде, но передан `conversation_id`, подставляется
   последний обработанный (`status: "done"`) файл, загруженный в этот чат.
3. Если нет ни того, ни другого — вопрос уходит без документа.

Отсюда важное следствие: последовательность «приложил файл → спросил →
спросил ещё раз» работает, документ не теряется на втором вопросе. Явная
ссылка в текущем ходу всегда важнее автоматики по чату.

Файлов в одном сообщении может быть несколько — до `MAX_ATTACHED_FILES`.
Превышение лимита — `400`, а не молчаливый выбор одного из них.

Текстовая история (через `conversation_id`) и подключение файла остаются
двумя независимыми механизмами: `conversation_id` в форме Responses
подтягивает текст прошлых сообщений всегда, а автоподстановка файла
работает только для файлов, загруженных именно с этим `conversation_id`.

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
│                                    частью после "chatcmpl-"/"resp_" в id ответа
├── user_id            UUID          NOT NULL, индексирован — владелец записи
├── conversation_id    UUID          nullable, FK → conversations.id (ON DELETE CASCADE) —
│                                    в форме Chat Completions только привязка к чату
│                                    в UI-списке; в форме Responses ЕЩЁ И источник
│                                    ТЕКСТОВОЙ истории для генерации (файлы — отдельно,
│                                    см. «API — общая идея»)
├── role                VARCHAR(16)   "user" | "assistant"
├── content             TEXT
├── sources             JSONB         [filename], если ответ использовал файл (для отображения)
├── model               VARCHAR(64)   значение "model" из запроса
├── prompt_tokens       INTEGER       nullable
├── completion_tokens   INTEGER       nullable
├── tokens              INTEGER       nullable, стоимость самой реплики — кэш для
│                                     сборки окна контекста (NULL = не посчитано)
└── created_at          TIMESTAMPTZ

message_feedback
├── id          SERIAL        PK (запись фидбэка, не используется в API)
├── message_id  UUID          FK → chat_messages.id (уникальный)
├── vote        INTEGER       1 = лайк / -1 = дизлайк / NULL = без оценки
├── comment     TEXT          nullable
├── created_at  TIMESTAMPTZ
└── updated_at  TIMESTAMPTZ

files                             только user_id обязателен; conversation_id — опционален
├── id                 UUID          PK, генерируется приложением (uuid4)
├── user_id            UUID          NOT NULL, индексирован — владелец файла
├── conversation_id    UUID          nullable, FK → conversations.id (ON DELETE SET NULL),
│                                    индексирован — если задан при загрузке, файл
│                                    автоматически подключается в форме Responses
│                                    (см. «API — общая идея»); в форме Chat Completions
│                                    не используется вообще, только явная ссылка
├── filename           VARCHAR(255)
├── mime_type          VARCHAR(127)  nullable
├── size_bytes         BIGINT
├── content             BYTEA         сам файл, хранится в БД (не на диске)
├── status              VARCHAR(16)   pending|processing|done|failed
├── markdown_content    TEXT          nullable, результат MinerU
├── markdown_tokens     INTEGER       nullable, стоимость markdown в токенах —
│                                     считается один раз при разборе
├── ocr_backend         VARCHAR(64)   nullable, напр. "mineru:pipeline:cyrillic"
├── error_message       TEXT          nullable
├── created_at          TIMESTAMPTZ
└── updated_at          TIMESTAMPTZ
```

Блоб файла (`content`) не грузится при обычных запросах — колонка
`deferred`, тянется явно только там, где реально нужны байты (передача в
MinerU). Удаление чата каскадно удаляет его сообщения и их фидбэк; файлы
этого чата не удаляются — только теряют привязку (`conversation_id` → `NULL`),
сам файл и его содержимое остаются доступны по `id`.

## Внешние сервисы

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.6:35b

MINERU_API_URL=http://127.0.0.1:8010
MINERU_BACKEND=pipeline
MINERU_LANG=cyrillic
MINERU_TIMEOUT_SECONDS=600
```

## Контекстное окно

Промпт собирается под фактический размер контекста модели, а не «последние
N сообщений». `num_ctx` передаётся в Ollama **явно**: без этого сервер молча
срезает начало промпта (context shift) вместе с правилами «отвечай только по
документу, не выдумывай» — агент теряет ровно те инструкции, ради которых
существует, и начинает свободно фантазировать.

```env
CONTEXT_WINDOW=4096
RESERVE_OUTPUT_TOKENS=512
CONTEXT_SAFETY_TOKENS=96
HISTORY_MIN_TOKENS=384
HISTORY_MAX_MESSAGES=200
TOKENIZER_REPO=unsloth/gemma-2-2b-it

MAX_ATTACHED_FILES=1
DOCUMENT_OVERFLOW=truncate
```

или 

```env
CONTEXT_WINDOW=32768
RESERVE_OUTPUT_TOKENS=8192
CONTEXT_SAFETY_TOKENS=128
HISTORY_MIN_TOKENS=4096
HISTORY_MAX_MESSAGES=200
TOKENIZER_REPO=Qwen/Qwen3.6-35B-A3B
MAX_ATTACHED_FILES=5
DOCUMENT_OVERFLOW=strict
```

| Переменная | Смысл |
|---|---|
| `CONTEXT_WINDOW` | `num_ctx` модели. Не «сколько влезет», а сколько выделяет Ollama под KV-кэш |
| `RESERVE_OUTPUT_TOKENS` | запас под генерацию, если клиент не прислал `max_tokens` |
| `CONTEXT_SAFETY_TOKENS` | подушка на служебные токены chat-шаблона |
| `HISTORY_MIN_TOKENS` | гарантированный минимум истории, чтобы уточняющие вопросы не теряли связь с прошлым ответом |
| `HISTORY_MAX_MESSAGES` | потолок выборки из БД (форма Responses), не логика контекста |
| `TOKENIZER_REPO` | HF-репозиторий токенайзера **той же** модели, что в Ollama |
| `MAX_ATTACHED_FILES` | сколько документов можно приложить к одному вопросу |
| `DOCUMENT_OVERFLOW` | `truncate` — обрезать документ с явным маркером; `strict` — `413` |

**Приоритет вытеснения.** Инструкции и текущий вопрос не выбрасываются
никогда. Документ получает основную долю бюджета — он и есть предмет
разговора. История вытесняется первой, скользящим окном с конца, непрерывным
куском; если окно начинается с реплики ассистента без вопроса, она
отбрасывается. При нескольких документах бюджет делится равными долями с
перераспределением излишков: документ, которому нужно меньше своей доли,
возвращает остаток остальным — иначе один большой вытеснил бы второй
маленький, хотя запрос был «сравни эти два».

**Переполнение.** Если документ не помещается:
- `DOCUMENT_OVERFLOW=truncate` — документ режется с конца по границе
  markdown-блока, в промпт вставляется маркер с долей помещённого текста и
  указанием прямо сказать, что документ показан не полностью. Без такого
  маркера модель уверенно отвечает «в документе этого нет» про то, что было
  в отрезанной части, — неотличимо от честного ответа.
- `DOCUMENT_OVERFLOW=strict` — `413`.

Обрезка видна снаружи: в `sources` имя файла получает суффикс
`"договор.pdf (частично)"`.

Если не помещается даже вопрос с инструкциями — `413` в любом режиме.

**`TOKENIZER_REPO` практически обязателен.** Без него используется
эвристический счётчик, который на словарях вроде gemma2 (256k,
SentencePiece) занижает оценку на UUID и URL более чем вдвое — а
переполнение происходит именно на них. Стоимость документа считается один
раз при загрузке и кэшируется в `files.markdown_tokens`; при смене
токенайзера кэш инвалидируется вручную:

```sql
UPDATE files SET markdown_tokens = NULL;
```

## API

Все эндпоинты требуют заголовок `X-User-Id: <uuid>`. Ресурсы скоупятся по
этому идентификатору; обращение к чужому ресурсу возвращает `404`.

### `POST /v1/files`

Загрузить документ. `multipart/form-data`, поле `file`, опционально ещё
одно текстовое поле формы — `conversation_id`. MinerU разбирает документ
**синхронно** внутри этого вызова; таймаут — `MINERU_TIMEOUT_SECONDS`
(до 600 секунд по умолчанию — клиент ждёт весь разбор в одном HTTP-вызове).
Общий ресурс для обеих форм генерации.

Если передан `conversation_id` — файл привязывается к этому чату (чат
должен принадлежать вызывающему, иначе `404`). Привязка используется
**только** формой Responses для автоматического подключения файла к ответам
этого чата — см. `POST /v1/responses` ниже. Без `conversation_id` файл
остаётся независимым, подключается только явной ссылкой в любой из форм —
как и раньше.

Ответ `201` — объект в формате OpenAI Files API + нестандартные поля:
```json
{
  "id": "file-85b365de-1234-4c7d-8e9f-0a1b2c3d4e5f",
  "object": "file",
  "bytes": 245678,
  "created_at": 1735900000,
  "filename": "накладная.pdf",
  "purpose": "assistants",
  "status": "processed",
  "status_details": null,
  "processing_status": "done",
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```
| Поле | Описание |
|---|---|
| `id` | `file-<uuid>` — используется в `/v1/chat/completions`, `/v1/responses` и путях `/v1/files/{id}` |
| `bytes` | размер файла в байтах |
| `purpose` | всегда `"assistants"` — единственное используемое значение сейчас |
| `status` | `uploaded` \| `processed` \| `error` — значения из спецификации OpenAI |
| `processing_status` | платформенное расширение: внутренний статус конвейера — `pending` \| `processing` \| `done` \| `failed` |
| `status_details` | текст ошибки MinerU, если обработка упала; иначе `null` |
| `conversation_id` | платформенное расширение; `null`, если файл не привязан ни к какому чату |

Внутренний конвейер обработки богаче, чем набор статусов OpenAI (там допустимы только `uploaded`/`processed`/`error`), поэтому в `status` уходит сведённое значение, а подробное — в `processing_status`. Схему БД это не затрагивает: сведение происходит на границе API.

| `processing_status` | `status` |
|---|---|
| `pending`, `processing` | `uploaded` |
| `done` | `processed` |
| `failed` | `error` |

Если MinerU не смог разобрать документ — ответ `502`:
```json
{"error": {"message": "Не удалось обработать документ: ...", "type": "server_error", "param": null, "code": null}}
```
Файл в БД при этом остаётся со `status: "failed"` и `error_message` (для
отладки), но клиенту доступен только текст ошибки — файл с таким `id`
дальше использовать нельзя (в контекст чата попадёт `404`/`422` при явной
ссылке, либо просто не будет подхвачен автоматикой — см. ниже).

---

#### `GET /v1/files`
Список загруженных файлов пользователя, новые первые.
```json
{
  "object": "list",
  "data": [
    {"id": "file-85b365de-...", "object": "file", "bytes": 245678, "created_at": 1735900000,
     "filename": "накладная.pdf", "purpose": "assistants", "status": "done", "status_details": null,
     "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
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

**Параметры OpenAI, которые сервис учитывает** (обе формы): `model`, `stream`, `stream_options.include_usage`, `temperature`, `top_p`, `max_tokens` / `max_completion_tokens` (в Responses — `max_output_tokens`), `store`, `n` (только `1` — другое значение отклоняется с `400`, а не игнорируется молча). Сообщения с `role: "system"` / `"developer"` (и поле `instructions` в Responses) добавляются к промпту сервиса как дополнительные инструкции: они уточняют поведение, но не отменяют язык ответа и запрет выдумывать содержимое документа.

**Не поддерживаются** (молча игнорируются): `tools`, `tool_choice`, `response_format`, `seed`, `logprobs`, `logit_bias`, `presence_penalty`, `frequency_penalty`, `stop`, `user`, `service_tier`.

**Вложения принимаются только по `file_id`.** Документ сначала загружается через `POST /v1/files` (там его разбирает MinerU), и уже полученный `file-<uuid>` передаётся в `content`. Части с инлайновым вложением — `image_url` и `input_audio` в Chat Completions, `input_image` и `input_audio` в Responses, а также `file` без `file_id` — отклоняются с `400` и подсказкой загрузить файл. Молча отвечать «документ не приложен» на такой запрос нельзя: мастер маршрутизирует его сюда именно из-за вложения, и пользователь ждёт ответа по нему.

**`DELETE /v1/chat/completions/{id}` и `DELETE /v1/responses/{id}`** удаляют сообщение ассистента вместе с его фидбэком (каскад по FK) и отдают `{"id": ..., "object": "chat.completion.deleted" | "response.deleted", "deleted": true}`. Реплика пользователя остаётся в чате, загруженный документ — тоже: он живёт своей жизнью в `/v1/files`.

### `POST /v1/chat/completions`

Генерация ответа, форма Chat Completions. Клиент присылает **всю** историю
диалога в `messages[]` — сервис её не хранит и не переиспользует между
запросами, независимо от `conversation_id`.

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
| `conversation_id` | UUID string | нет | платформенное расширение — привязать сообщение к чату из `/v1/platform/conversations`. **Только ярлык для UI** в этой форме — контекст всегда из `messages[]`. Чужой/несуществующий `conversation_id` → `404` |

`content` у сообщения с `role: "user"` может быть:
- обычной строкой — вопрос без документа;
- массивом content-частей — `{"type": "text", "text": "..."}` (текст вопроса)
  и до `MAX_ATTACHED_FILES` частей
  `{"type": "file", "file": {"file_id": "file-..."}}`
  (ссылки на ранее загруженные документы, `file_id` **вложен** под ключ
  `"file"`). Каждый файл должен принадлежать вызывающему и иметь
  `status: "done"` — иначе `404` (файл не найден/чужой) или `400`
  (`status` не `"done"`). Дубликаты `file_id` схлопываются.

Ссылка на файл **липкая**: она действует и на последующие вопросы, пока
клиент не приложит другой документ (см. «API — общая идея»). Набор файлов
задаёт последнее сообщение, в котором вложения были.

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
сообщении был `file_id`, в сохранённом ответе поле `sources` содержит имена
использованных файлов; обрезанный по контексту документ помечается
суффиксом — `["договор.pdf (частично)"]`.

---

### `GET /v1/chat/completions/{completion_id}`

Получить ранее сгенерированный ответ повторно (в форме `chat.completion`) —
по `id`, который пришёл в ответе. Работает для сообщений, сгенерированных
**любой** из двух форм: `chatcmpl-<uuid>`, `resp_<uuid>` или голый UUID.

`404`, если `id` не найден, принадлежит не вам, либо указывает на сообщение
с `role: "user"`.

---

### `POST /v1/responses`

> Поле в спецификации называется `conversation` (строка или `{"id": ...}`) — сервис принимает его, а `conversation_id` оставлен алиасом для фронта платформы. Поддерживается и стандартный `previous_response_id`: сервис находит предыдущий ответ и берёт чат, которому тот принадлежал; неизвестный или чужой id → `404`. В объекте ответа возвращается расширение `file_id` — документ, по которому агент фактически отвечал (явный из `input` или подхваченный из чата).

Генерация ответа, форма Responses API. Два режима в зависимости от того,
передан ли `conversation_id` — про текстовую историю:

- **Без `conversation_id`** — stateless: `input` должен содержать всё, что
  нужно модели (аналог `messages[]`).
- **С `conversation_id`** — `input` должен содержать **только новый ход**
  (текст + опционально ссылку на файл). Сервис сам читает последние
  **текстовые** сообщения этого чата из БД (не более
  `HISTORY_MAX_MESSAGES`) и подставляет их как историю; сколько из них
  реально войдёт в промпт, решает бюджет контекста. Если клиент всё равно
  пришлёт историю в `input` вместе с `conversation_id` — `422`:
  ```json
  {"error": {"message": "При переданном conversation_id input должен содержать только новый ход (без истории) — история собирается агентом из БД по conversation_id", ...}}
  ```

И отдельно, независимо от текстовой истории — режим подключения файла:

- **Есть `input_file` в `input`** — используется он, всегда, вне
  зависимости от того, привязан ли этот (или любой другой) файл к чату.
  Явная ссылка — высший приоритет. Если ссылки нет в текущем ходу, но есть
  в более раннем сообщении присланного `input`, берётся оно (липкий файл).
- **Нет `input_file` нигде, но есть `conversation_id`** — агент сам ищет
  последний файл со `status: "done"`, загруженный в этот чат через
  `POST /v1/files` с тем же `conversation_id`, и подставляет его без
  участия клиента. Если такого файла нет (ничего не загружали в этот чат,
  или загруженный файл ещё не разобрался/упал) — вопрос уходит без
  документа, как обычный текстовый.
- **Нет ни `input_file`, ни `conversation_id`** — как и раньше, документа в
  контексте нет вообще.

Тело запроса — файл подключён явной ссылкой:
```json
{
  "model": "document_chat",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Какая сумма в накладной?"},
        {"type": "input_file", "file_id": "file-85b365de-1234-4c7d-8e9f-0a1b2c3d4e5f"}
      ]
    }
  ],
  "stream": true,
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```
Тело запроса — файл подключён автоматически (был загружен с этим же
`conversation_id`, `input_file` в текущем ходу не указан):
```json
{
  "model": "document_chat",
  "input": "Какая сумма в накладной?",
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```
**Важно:** формат явной ссылки на файл здесь **отличается** от формы Chat
Completions — `file_id` лежит **плоским полем прямо в части**
(`{"type": "input_file", "file_id": "..."}`), а не вложен под ключ `"file"`,
как в Chat Completions (`{"type": "file", "file": {"file_id": "..."}}`).
Это настоящий формат спеки Responses API, не опечатка и не расхождение
между формами этого агента. `"output_text"` в частях — эхо прошлого ответа
модели при ручном ведении истории без `conversation_id` (наравне с
`"input_text"`).

**Нестрим-ответ**:
```json
{
  "id": "resp_1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
  "object": "response",
  "created_at": 1735900000,
  "status": "completed",
  "model": "document_chat",
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "output": [{
    "id": "msg_1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2",
    "type": "message",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "В накладной сумма 1000 руб.", "annotations": []}]
  }],
  "usage": {"input_tokens": 1204, "output_tokens": 18, "total_tokens": 1222}
}
```
`usage` здесь — `input_tokens`/`output_tokens` (терминология Responses
API), а не `prompt_tokens`/`completion_tokens`.

**Стрим-ответ** — гранулярные типизированные SSE-события:
```
event: response.created
data: {"type":"response.created","sequence_number":1,"response":{"id":"resp_...","status":"in_progress",...}}

event: response.output_item.added
data: {...}

event: response.content_part.added
data: {...}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_...","delta":"В"}

... (ещё response.output_text.delta на каждый токен) ...

event: response.output_text.done
data: {...}

event: response.content_part.done
data: {...}

event: response.output_item.done
data: {...}

event: response.completed
data: {"type":"response.completed","sequence_number":N,"response":{"id":"resp_...","status":"completed","output":[...],"usage":{...}}}
```
`sequence_number` — сквозной монотонный счётчик на весь поток. Ошибка в
процессе генерации — `event: error` с `sequence_number: 9999`.

---

### `GET /v1/responses/{completion_id}`

То же самое, что `GET /v1/chat/completions/{completion_id}`, но возвращает
`response`-объект. Принимает id в любом виде.

---

### `POST/GET/DELETE /v1/chat/completions/{completion_id}/feedback`

Оценка ответа ассистента — **общий путь для обеих форм генерации**,
`{completion_id}` принимает `chatcmpl-<uuid>`, `resp_<uuid>` или голый UUID
одинаково.

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
удаление). **Не входит ни в одну спеку OpenAI** и не участвует в генерации
напрямую — см. «API — общая идея» выше.

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
История сообщений чата вместе с фидбэком.
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
`/v1/chat/completions/{id}/feedback` и повторного чтения в любой из форм.
Обратите внимание: исходный `content` пользовательского сообщения здесь —
только текстовая часть вопроса; сама ссылка на файл (`file_id`) в истории
не хранится как структурная часть сообщения — она видна косвенно, через
`sources` в ответе ассистента.

#### `PATCH /v1/platform/conversations/{id}`
Переименовать чат. Тело: `{ "title": "Новое название" }`.

#### `DELETE /v1/platform/conversations/{id}`
Удалить чат со всеми сообщениями и их фидбэком (каскадно). Ответ: `204 No Content`.

## Ошибки

Единый формат вместо FastAPI-дефолта `{"detail": ...}`:
```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}
```
Невалидное тело запроса — **`400`**, а не `422`: OpenAI отвечает на такие запросы именно `400`, а SDK мапит `422` в `UnprocessableEntityError`, мимо клиентского `except BadRequestError`. Поле `param` заполняется путём до проблемного поля (`messages.0.role`), а не остаётся `null`.

`type` — грубая классификация по HTTP-статусу: `400/413/415/422` →
`invalid_request_error`, `401` → `authentication_error`, `404` →
`not_found_error`, остальное → `server_error`.

## Примеры curl

```bash
U=11111111-1111-1111-1111-111111111111
BASE=http://127.0.0.1:8006

# --- загрузить документ (общий шаг для обеих форм) ---
curl -X POST $BASE/v1/files \
  -H "X-User-Id: $U" -F "file=@накладная.pdf;type=application/pdf"
# -> {"id": "file-85b365de-...", "status": "done", "conversation_id": null, ...}

FID=file-85b365de-1234-4c7d-8e9f-0a1b2c3d4e5f

# ============ форма Chat Completions ============

curl -N -X POST $BASE/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"document_chat\", \"stream\": true, \"messages\": [
        {\"role\": \"user\", \"content\": [
          {\"type\": \"text\", \"text\": \"Какая сумма в накладной?\"},
          {\"type\": \"file\", \"file\": {\"file_id\": \"$FID\"}}
        ]}
      ]}"

curl -X POST $BASE/v1/chat/completions \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d '{"model": "document_chat", "messages": [{"role": "user", "content": "Что такое MinerU?"}]}'

# ============ форма Responses — файл явной ссылкой ============

# без conversation_id — вопрос по документу, история целиком от клиента
curl -N -X POST $BASE/v1/responses \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"document_chat\", \"stream\": true, \"input\": [
        {\"role\": \"user\", \"content\": [
          {\"type\": \"input_text\", \"text\": \"Какая сумма в накладной?\"},
          {\"type\": \"input_file\", \"file_id\": \"$FID\"}
        ]}
      ]}"

# ============ форма Responses — файл автоматически (загружен в чат) ============

# создать чат
curl -X POST $BASE/v1/platform/conversations \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"title": "Накладная"}'

CID=3fa85f64-5717-4562-b3fc-2c963f66afa6

# загрузить файл СРАЗУ с привязкой к чату
curl -X POST $BASE/v1/files \
  -H "X-User-Id: $U" -F "file=@накладная.pdf;type=application/pdf" -F "conversation_id=$CID"
# -> {"id": "file-...", "conversation_id": "3fa85f64-...", "status": "done", ...}

# первый вопрос — file_id указывать не нужно, подхватится сам
curl -X POST $BASE/v1/responses \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"document_chat\", \"conversation_id\": \"$CID\", \"input\": \"Какая сумма в накладной?\"}"

# следующий вопрос — снова только новый ход, документ по-прежнему подхватится сам
curl -X POST $BASE/v1/responses \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"document_chat\", \"conversation_id\": \"$CID\", \"input\": \"А дата какая?\"}"

# если в этом же чате нужно спросить про ДРУГОЙ файл — просто указать file_id явно,
# явная ссылка всегда переопределяет автоматику
FID2=file-99999999-1234-4c7d-8e9f-0a1b2c3d4e5f
curl -X POST $BASE/v1/responses \
  -H "X-User-Id: $U" -H "Content-Type: application/json" \
  -d "{\"model\": \"document_chat\", \"conversation_id\": \"$CID\", \"input\": [
        {\"role\": \"user\", \"content\": [
          {\"type\": \"input_text\", \"text\": \"А в этом документе что?\"},
          {\"type\": \"input_file\", \"file_id\": \"$FID2\"}
        ]}
      ]}"

ID=resp_1e6b7ee7-d5bb-4f0a-8f9e-a06f19a8f3c2

curl $BASE/v1/responses/$ID -H "X-User-Id: $U"

# ============ общее для обеих форм ============

curl $BASE/v1/chat/completions/$ID -H "X-User-Id: $U"

curl -X POST $BASE/v1/chat/completions/$ID/feedback \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"vote": 1, "comment": "точно"}'
curl $BASE/v1/chat/completions/$ID/feedback -H "X-User-Id: $U"
curl -X DELETE $BASE/v1/chat/completions/$ID/feedback -H "X-User-Id: $U"

curl $BASE/v1/platform/conversations -H "X-User-Id: $U"
curl $BASE/v1/platform/conversations/$CID/messages -H "X-User-Id: $U"

curl -X PATCH $BASE/v1/platform/conversations/$CID \
  -H "X-User-Id: $U" -H "Content-Type: application/json" -d '{"title": "Накладная №1"}'
curl -X DELETE $BASE/v1/platform/conversations/$CID -H "X-User-Id: $U"

# --- удалить файл, когда он больше не нужен ---
curl -X DELETE $BASE/v1/files/$FID -H "X-User-Id: $U"
```