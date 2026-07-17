import httpx
from pathlib import Path

MINERU_API_URL = "http://127.0.0.1:8010"


def parse_document(
    file_path: str,
    lang: str = "cyrillic",
    backend: str = "pipeline",
) -> str:
    path = Path(file_path)
    with open(path, "rb") as f:
        files = {"files": (path.name, f, "application/octet-stream")}
        data = {
            "backend": backend,
            "lang_list": lang,
            "parse_method": "auto",
            "formula_enable": "true",
            "table_enable": "true",
            "return_md": "true",
        }
        resp = httpx.post(
            f"{MINERU_API_URL}/file_parse",
            files=files,
            data=data,
            timeout=600,
        )
    resp.raise_for_status()
    payload = resp.json()
    # ключ результата — имя файла без расширения
    return payload["results"][path.stem]["md_content"]


if __name__ == "__main__":
    md = parse_document("utochnenie_parametrov_modeli_razrusheniya_splava_amg6_pri_vysokoskorostnom.pdf")
    print(md[:3000])
