from __future__ import annotations

import logging
import re
from math import ceil
from typing import Protocol

log = logging.getLogger(__name__)

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


class TokenCounter(Protocol):
    """Любой счётчик токенов: точный или эвристический."""

    name: str

    def count(self, text: str) -> int: ...


class HeuristicCounter:
    """Оценка без загрузки токенайзера.

    Кириллица и латиница считаются отдельно: у BPE-словарей русский текст
    даёт заметно больше токенов на символ. Коэффициенты подобраны так, чтобы
    ОЦЕНКА БЫЛА СВЕРХУ — лучше отрезать лишнее сообщение, чем упереться в ctx.
    """

    name = "heuristic"

    def __init__(
        self,
        latin_chars_per_token: float = 3.5,
        cyrillic_chars_per_token: float = 2.0,
    ) -> None:
        self._latin = latin_chars_per_token
        self._cyrillic = cyrillic_chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        cyrillic = len(_CYRILLIC.findall(text))
        other = len(text) - cyrillic
        return max(1, ceil(cyrillic / self._cyrillic + other / self._latin))


class HFCounter:
    """Точный подсчёт через токенайзер модели (пакет `tokenizers`).

    Имя берётся из HF-репозитория той же модели, что крутится в ollama:
    llama3.1 -> "meta-llama/Llama-3.1-8B-Instruct", qwen3 -> "Qwen/Qwen3-8B" и т.д.
    """

    name = "hf"

    def __init__(self, repo_id: str) -> None:
        from tokenizers import Tokenizer  # локальный импорт: зависимость опциональна

        self._tokenizer = Tokenizer.from_pretrained(repo_id)
        self.name = f"hf:{repo_id}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class CachedCounter:
    """Мемоизация по тексту: системный промпт и повторяющиеся реплики
    считаются один раз."""

    def __init__(self, inner: TokenCounter, maxsize: int = 4096) -> None:
        self._inner = inner
        self._maxsize = maxsize
        self._cache: dict[str, int] = {}
        self.name = inner.name

    def count(self, text: str) -> int:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        value = self._inner.count(text)
        if len(self._cache) >= self._maxsize:
            self._cache.clear()
        self._cache[text] = value
        return value


def make_counter(repo_id: str | None) -> TokenCounter:
    """Пробуем точный токенайзер, при неудаче — эвристика."""
    if repo_id:
        try:
            return CachedCounter(HFCounter(repo_id))
        except Exception as e:
            log.warning(
                "не удалось загрузить токенайзер %r (%s), "
                "переключаюсь на эвристический подсчёт",
                repo_id, e,
            )
    return CachedCounter(HeuristicCounter())
