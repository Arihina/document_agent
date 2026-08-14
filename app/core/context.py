from __future__ import annotations

from dataclasses import dataclass, field

from app.core.tokens import TokenCounter


JOIN_SLACK = 16

HEAD_GENERAL = """Ты русскоязычный AI ассистент, который помогает разбираться с документами.
Отвечай ТОЛЬКО на русском языке.
Сейчас документ не прикреплён к этому вопросу — это либо общий вопрос,
либо вопрос до загрузки файла. Если по смыслу вопроса нужен документ,
которого нет, — вежливо попроси его прикрепить, не выдумывай ответ."""

HEAD_DOCUMENT = """Ты русскоязычный AI ассистент, который отвечает на вопросы по содержимому документа.
Отвечай ТОЛЬКО на русском языке.
Использу ТОЛЬКО информацию из документа ниже. Если ответа в документе нет —
так и скажи, не выдумывай."""

TRUNCATION_NOTE = (
    "[Документ приведён частично: помещено примерно {percent}% содержимого, "
    "остальное не влезло в контекст модели. Если ответа в приведённой части "
    "нет — прямо скажи, что документ показан не полностью, и не делай выводов "
    "о том, чего в нём нет.]"
)

PARTIAL_SUFFIX = " (частично)"


class ContextOverflow(Exception):
    """Промпт не помещается в контекстное окно модели.

    Транспортный слой переводит это в 413. Модуль намеренно не знает про
    FastAPI: у RAG-агентов переполнение обрабатывается иначе — выбрасыванием
    чанков, а не ошибкой клиенту.
    """

    def __init__(self, message: str, needed: int, available: int) -> None:
        super().__init__(message)
        self.needed = needed
        self.available = available


@dataclass
class Document:
    filename: str
    markdown: str
    tokens: int | None = None


@dataclass
class PromptStats:
    num_ctx: int
    prompt_tokens: int
    reserved_output: int
    documents: int
    truncated: int
    document_tokens: int
    history_kept: int
    history_total: int
    history_budget: int
    counter: str

    def as_log(self) -> str:
        trunc = f" ({self.truncated} обрезан)" if self.truncated else ""
        return (
            f"ctx={self.num_ctx} prompt≈{self.prompt_tokens} "
            f"reserve={self.reserved_output} docs={self.documents}{trunc} "
            f"doc_tokens={self.document_tokens} "
            f"history={self.history_kept}/{self.history_total} "
            f"history_budget={self.history_budget} counter={self.counter}"
        )


@dataclass
class BuiltPrompt:
    prompt: str
    sources: list[str] | None
    stats: PromptStats
    truncated_files: list[str] = field(default_factory=list)


def _fit_shares(costs: list[int], budget: int) -> list[int]:
    """Равные доли с перераспределением излишков.

    Документ, которому нужно меньше своей доли, возвращает остаток в общий
    котёл — иначе один большой документ вытеснит второй маленький, хотя
    запрос был именно «сравни эти два».
    """
    alloc = [0] * len(costs)
    hungry = list(range(len(costs)))
    left = budget

    while hungry and left > 0:
        share = left // len(hungry)
        if share == 0:
            break
        still: list[int] = []
        for i in hungry:
            take = min(share, costs[i] - alloc[i])
            alloc[i] += take
            left -= take
            if alloc[i] < costs[i]:
                still.append(i)
        hungry = still

    return alloc


def _truncate(text: str, limit: int, counter: TokenCounter) -> str:
    """Обрезать текст до limit токенов по границе markdown-блока."""
    if limit <= 0:
        return ""

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1

    cut = lo
    boundary = text.rfind("\n\n", 0, cut)
    if boundary > cut * 0.6:
        cut = boundary

    return text[:cut].rstrip()


def _document_block(docs: list[Document], notes: dict[int, str]) -> str:
    if not docs:
        return ""

    if len(docs) == 1:
        body = docs[0].markdown
        if 0 in notes:
            body = f"{notes[0]}\n\n{body}"
        return f"Документ: {docs[0].filename}\n{body}"

    parts = []
    for i, d in enumerate(docs):
        body = d.markdown
        if i in notes:
            body = f"{notes[i]}\n\n{body}"
        parts.append(
            f"Документ {i + 1} из {len(docs)}: {d.filename}\n{body}")

    return "\n\n".join(parts)


def _history_block(history: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"{'Пользователь' if role == 'user' else 'Ассистент'}: {text}"
        for role, text in history
    )


def _skeleton(head: str, instructions: str | None, documents: str,
              history: str, question: str) -> str:
    instr = ""
    if instructions and instructions.strip():
        instr = (f"\nДополнительные инструкции пользователя:\n"
                 f"{instructions.strip()}\n")

    doc_section = f"\n{documents}\n" if documents else ""

    return f"""{head}
{instr}{doc_section}
Предыдущий диалог:
{history}

Вопрос пользователя:
{question}

Ответ:""".strip()


def build_prompt(
    *,
    question: str,
    documents: list[Document],
    history: list[tuple[str, str]],
    instructions: str | None,
    counter: TokenCounter,
    num_ctx: int,
    reserve_output: int,
    safety: int = 96,
    history_min_tokens: int = 384,
    overflow: str = "truncate",
) -> BuiltPrompt:
    """Собирает промпт, укладываясь в num_ctx с запасом на генерацию.

    Приоритет вытеснения: инструкции и вопрос неприкосновенны, документы
    получают основную долю бюджета, история жертвуется первой.
    """
    head = HEAD_DOCUMENT if documents else HEAD_GENERAL

    fixed = counter.count(
        _skeleton(head, instructions, "", "", question)) + JOIN_SLACK
    free = num_ctx - reserve_output - safety - fixed

    if free < 0:
        needed = fixed + reserve_output + safety
        raise ContextOverflow(
            "Вопрос вместе с инструкциями не помещается в контекст модели: "
            f"нужно {needed} токенов при лимите {num_ctx}. Сократите вопрос "
            "или увеличьте CONTEXT_WINDOW.",
            needed=needed, available=num_ctx,
        )

    doc_costs = [
        d.tokens if d.tokens is not None else counter.count(d.markdown)
        for d in documents
    ]

    doc_costs = [
        c + counter.count(d.filename) + 12
        for c, d in zip(doc_costs, documents)
    ]

    history_floor = min(history_min_tokens, free) if history else 0
    doc_budget = max(0, free - history_floor)
    shares = _fit_shares(doc_costs, doc_budget)

    note_cost = counter.count(TRUNCATION_NOTE.format(percent=100)) + 4
    fitted: list[Document] = []
    notes: dict[int, str] = {}
    truncated_files: list[str] = []

    for i, (d, cost, share) in enumerate(zip(documents, doc_costs, shares)):
        if share >= cost:
            fitted.append(d)
            continue

        if overflow == "strict":
            raise ContextOverflow(
                f"Документ «{d.filename}» не помещается в контекст модели: "
                f"нужно {cost} токенов, доступно {share}. Увеличьте "
                "CONTEXT_WINDOW или используйте документ меньшего объёма.",
                needed=cost, available=share,
            )

        body = _truncate(d.markdown, max(0, share - note_cost), counter)
        percent = max(1, round(100 * len(body) / max(1, len(d.markdown))))
        notes[i] = TRUNCATION_NOTE.format(percent=percent)
        fitted.append(Document(d.filename, body))
        truncated_files.append(d.filename)

    doc_block = _document_block(fitted, notes)
    doc_used = counter.count(doc_block) if doc_block else 0

    history_budget = max(0, free - doc_used)
    kept: list[tuple[str, str]] = []
    used = 0

    for role, text in reversed(history):
        line = f"{'Пользователь' if role == 'user' else 'Ассистент'}: {text}"
        cost = counter.count(line) + 1
        if used + cost > history_budget:
            break
        used += cost
        kept.append((role, text))

    kept.reverse()
    while kept and kept[0][0] == "assistant":
        kept.pop(0)

    prompt = _skeleton(head, instructions, doc_block,
                       _history_block(kept), question)

    sources = None
    if documents:
        sources = [
            d.filename +
            (PARTIAL_SUFFIX if d.filename in truncated_files else "")
            for d in documents
        ]

    stats = PromptStats(
        num_ctx=num_ctx,
        prompt_tokens=counter.count(prompt),
        reserved_output=reserve_output,
        documents=len(documents),
        truncated=len(truncated_files),
        document_tokens=doc_used,
        history_kept=len(kept),
        history_total=len(history),
        history_budget=history_budget,
        counter=getattr(counter, "name", "unknown"),
    )

    return BuiltPrompt(prompt, sources, stats, truncated_files)
