#!/usr/bin/env python3
"""
run_eval.py — прогон фото блюд через DeepSeek V4.1 Flash Free.

Что делает скрипт
-----------------
1. Читает food_catalog.json из папки nut-01.
2. Для каждого блюда открывает фото из data/raw/dishes/.
3. Отправляет фото и вес порции в модель DeepSeek V4.1 Flash Free.
4. Сохраняет ответ модели в out/runs/DeepSeek_v1/dish_XX.json.

Почему используется локальный API OpenCode
------------------------------------------
Бесплатная модель DeepSeek V4.1 Flash Free доступна только внутри OpenCode
(таково ограничение провайдера). Поэтому скрипт обращается не напрямую к
платному OpenAI-совместимому шлюзу, а к локальному HTTP API OpenCode.
OpenCode уже хранит авторизацию и сам проксирует запрос в модель.

Что должно быть запущено
------------------------
- Приложение OpenCode Desktop (или фоновая служба OpenCode).
- Python 3.9+ и пакет requests:
      python3 -m pip install requests

Как запускать
-------------
Сначала полезно прогнать self-test только на одном блюде:

    python3 workspace/works/nut-01/tools/run_eval.py --self-test

Один конкретный диск:

    python3 workspace/works/nut-01/tools/run_eval.py --dish dish_01

Все блюда из каталога:

    python3 workspace/works/nut-01/tools/run_eval.py

Посмотреть, что будет отправлено, без вызова модели:

    python3 workspace/works/nut-01/tools/run_eval.py --dish dish_01 --dry-run

Необязательные переменные окружения
-----------------------------------
    OPENCODE_SERVER_URL       адрес API, например http://127.0.0.1:49374
    OPENCODE_SERVER_PASSWORD  пароль службы OpenCode
    OPENCODE_MODEL            ID модели (по умолчанию deepseek-v4.1-flash-free)

Если адрес не задан, скрипт возьмёт его из `opencode service status`,
а пароль — из ~/.config/opencode/service.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import requests


# ---------------------------------------------------------------------------
# Пути и настройки
# ---------------------------------------------------------------------------

# tools/run_eval.py -> tools -> nut-01
BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE_DIR / "food_catalog.json"
DEFAULT_OUT_DIR = BASE_DIR / "out" / "runs" / "BigPickle_v1"

# Модель и провайдер в терминах OpenCode.
PROVIDER_ID = "opencode"
MODEL_ID = os.environ.get("OPENCODE_MODEL", "big-pickle")

# Схема S-01 из skill/SKILL.md: ровно эти поля должны быть в ответе.
TOP_LEVEL_FIELDS = {"components", "kcal", "protein", "fat", "carbs", "confidence", "notes"}
COMPONENT_FIELDS = {"name", "weight_g", "kcal", "protein", "fat", "carbs"}

# Регулярка для поиска URL в выводе `opencode service status`.
URL_RE = re.compile(r"https?://[^\s]+")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Промпт для модели
# ---------------------------------------------------------------------------

def build_prompt(portion_g: float) -> str:
    """
    Собирает промпт P-01 в компактном виде.

    Важно: название блюда из каталога НЕ передаём — модель должна распознать
    блюдо по фото. Передаём только вес порции, как требует контракт скилла.
    """
    portion = f"{portion_g:g}"
    return f"""Ты — модуль распознавания блюд и оценки пищевой ценности.
На фото одно готовое блюдо. Масса всей порции: {portion} г.
Разбери блюдо на компоненты, оцени вес каждого компонента и рассчитай калорийность, белки, жиры и углеводы НА ВСЮ ПОРЦИЮ (не на 100 г).

Верни строго один JSON-объект без markdown, комментариев и текста вокруг.
Схема:
{{
  "components": [
    {{"name": "название компонента", "weight_g": 0, "kcal": 0, "protein": 0, "fat": 0, "carbs": 0}}
  ],
  "kcal": 0, "protein": 0, "fat": 0, "carbs": 0,
  "confidence": "low|medium|high",
  "notes": ["допущение"]
}}
Правила:
- 1–15 компонентов; все числа — числа, не строки.
- Сумма weight_g должна быть в пределах ±10% от {portion} г.
- Верхнеуровневые kcal/protein/fat/carbs равны сумме по компонентам (±2%).
- notes: 0–5 коротких строк только про допущения и ограничения распознавания.
- Не давай медицинских рекомендаций, диагнозов и гарантий по аллергенам.
- Если блюдо распознано неуверенно, поставь confidence "low", но всё равно верни JSON.
- Не используй инструменты. Ответь сразу JSON."""


# ---------------------------------------------------------------------------
# Поиск локального сервера OpenCode
# ---------------------------------------------------------------------------

def service_config_path() -> Path:
    """Путь к service.json (там лежит пароль локального API)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "opencode" / "service.json"
    return Path.home() / ".config" / "opencode" / "service.json"


def find_opencode_cli() -> Optional[str]:
    """Ищет исполняемый файл opencode в PATH и типовых местах установки."""
    candidates = []
    which = shutil.which("opencode")
    if which:
        candidates.append(Path(which))
    candidates.append(Path.home() / ".opencode" / "bin" / "opencode")

    # macOS-приложение OpenCode Desktop кладёт CLI сюда.
    desktop_cli_dir = Path.home() / "Library" / "Application Support" / "ai.opencode.desktop" / "cli"
    if desktop_cli_dir.is_dir():
        candidates.extend(sorted(desktop_cli_dir.glob("*/opencode-cli"), reverse=True))

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def discover_server_url() -> str:
    """
    Возвращает адрес локального API OpenCode.

    Сначала смотрим переменную окружения, затем спрашиваем `opencode service status`.
    """
    env_url = os.environ.get("OPENCODE_SERVER_URL")
    if env_url:
        return env_url.rstrip("/")

    cli = find_opencode_cli()
    if not cli:
        raise RuntimeError(
            "Не найден исполняемый файл opencode. Запустите OpenCode Desktop "
            "или укажите адрес вручную: OPENCODE_SERVER_URL=http://127.0.0.1:PORT"
        )

    try:
        proc = subprocess.run(
            [cli, "service", "status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Не удалось выполнить `opencode service status`: {exc}") from exc

    output = ANSI_RE.sub("", (proc.stdout or "") + "\n" + (proc.stderr or ""))
    match = URL_RE.search(output)
    if not match:
        raise RuntimeError(
            "OpenCode не сообщил адрес API. Убедитесь, что приложение запущено, "
            "или задайте OPENCODE_SERVER_URL вручную.\n"
            f"Вывод `opencode service status`:\n{output.strip()}"
        )
    return match.group(0).rstrip("/")


def read_password() -> str:
    """Берёт пароль API из окружения или из service.json."""
    env_password = os.environ.get("OPENCODE_SERVER_PASSWORD")
    if env_password:
        return env_password

    path = service_config_path()
    if not path.is_file():
        raise RuntimeError(
            f"Не найден файл с паролем OpenCode: {path}. "
            "Запустите OpenCode Desktop или задайте OPENCODE_SERVER_PASSWORD."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Не удалось прочитать {path}: {exc}") from exc

    password = data.get("password")
    if not password:
        raise RuntimeError(f"В {path} нет поля password")
    return password


# ---------------------------------------------------------------------------
# Клиент локального API OpenCode
# ---------------------------------------------------------------------------

class OpenCodeClient:
    """Минимальный клиент HTTP API OpenCode: сессия -> промпт -> ожидание -> ответ."""

    def __init__(self, base_url: str, password: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.http = requests.Session()
        # Локальный API использует Basic Auth: логин "opencode", пароль из service.json.
        self.http.auth = ("opencode", password)
        self.http.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "nut-01-run-eval/1.0",
            }
        )

    def request(self, method: str, path: str, *, timeout: Optional[int] = None, **kwargs: Any) -> requests.Response:
        """Выполняет запрос и превращает HTTP-ошибки в понятные исключения."""
        try:
            response = self.http.request(
                method,
                self.base_url + path,
                timeout=timeout if timeout is not None else self.timeout,
                **kwargs,
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Таймаут запроса {method} {path}. "
                "Модель могла не ответить за отведённое время; увеличьте --timeout."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Ошибка сети при запросе {method} {path}: {exc}") from exc

        if response.status_code >= 400:
            body = response.text.strip()
            if len(body) > 1000:
                body = body[:1000] + "..."
            raise RuntimeError(f"{method} {path} -> HTTP {response.status_code}: {body}")
        return response

    def create_session(self, title: str) -> str:
        """Создаёт отдельную сессию, чтобы у каждого блюда не было истории."""
        response = self.request(
            "POST",
            "/api/session",
            json={
                "title": title,
                "model": {"id": MODEL_ID, "providerID": PROVIDER_ID},
            },
        )
        try:
            return response.json()["data"]["id"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Неожиданный ответ при создании сессии: {response.text[:500]}") from exc

    def ask(self, session_id: str, image_path: Path, prompt: str, timeout: int) -> str:
        """
        Отправляет промпт с фото, ждёт завершения хода и возвращает текст ответа.

        Файл передаётся как file:// URI; OpenCode сам читает его и прикладывает
        к сообщению как изображение.
        """
        uri = image_path.resolve().as_uri()
        self.request(
            "POST",
            f"/api/session/{session_id}/prompt",
            json={
                "text": prompt,
                "files": [
                    {
                        "uri": uri,
                        "name": image_path.name,
                        "source": {"type": "uri", "uri": uri},
                    }
                ],
            },
        )

        # Ждём, пока агентский цикл сессии станет idle.
        self.request(
            "POST",
            f"/api/experimental/session/{session_id}/wait",
            timeout=timeout,
        )

        # Забираем последние сообщения ассистента. Нам нужен только текст,
        # reasoning-блоки пропускаем.
        response = self.request(
            "GET",
            f"/api/session/{session_id}/message",
            params={"type": "assistant", "order": "desc", "limit": "5"},
        )
        try:
            messages = response.json().get("data", [])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Неожиданный ответ при чтении сообщений: {response.text[:500]}") from exc

        for message in messages:
            text = extract_text(message)
            if text:
                return text

        raise RuntimeError("Модель не вернула текст (ответ пустой или содержит только reasoning)")


def extract_text(message: dict) -> str:
    """Склеивает все текстовые части сообщения ассистента."""
    parts = message.get("content")
    if isinstance(parts, str):
        return parts.strip()
    if not isinstance(parts, list):
        return ""

    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            value = part.get("text")
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
    return "\n".join(texts).strip()


# ---------------------------------------------------------------------------
# Разбор и проверка ответа
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> dict:
    """
    Достаёт JSON-объект из ответа модели.

    Модель иногда оборачивает JSON в ```json ... ``` или добавляет пояснение.
    Сначала пробуем разобрать весь текст, затем ищем первый сбалансированный
    объект { ... } с учётом строк и экранирования.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("В ответе модели нет JSON-объекта")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : index + 1]
                value = json.loads(candidate)
                if not isinstance(value, dict):
                    raise ValueError("JSON в ответе модели — не объект")
                return value

    raise ValueError("В ответе модели нет закрытого JSON-объекта")


def is_number(value: Any) -> bool:
    """bool в Python — тоже int, поэтому проверяем его отдельно."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_answer(answer: dict, portion_g: float) -> list[str]:
    """
    Проверяет ответ по схеме S-01 и возвращает список предупреждений.

    Скрипт не исправляет ответ и не выбрасывает его: он сохраняет то, что
    вернула модель, а человек видит предупреждения в консоли.
    """
    warnings: list[str] = []

    missing = TOP_LEVEL_FIELDS - set(answer)
    extra = set(answer) - TOP_LEVEL_FIELDS
    if missing:
        warnings.append(f"нет обязательных полей: {', '.join(sorted(missing))}")
    if extra:
        warnings.append(f"лишние поля: {', '.join(sorted(extra))}")

    for field in ("kcal", "protein", "fat", "carbs"):
        value = answer.get(field)
        if not is_number(value):
            warnings.append(f"{field}: ожидалось число, получено {type(value).__name__}")
        elif value < 0:
            warnings.append(f"{field}: отрицательное значение {value}")

    confidence = answer.get("confidence")
    if confidence not in ("low", "medium", "high"):
        warnings.append(f"confidence: ожидалось low/medium/high, получено {confidence!r}")

    notes = answer.get("notes")
    if not isinstance(notes, list):
        warnings.append("notes: ожидался массив строк")
    elif len(notes) > 5:
        warnings.append(f"notes: {len(notes)} строк, максимум 5")

    components = answer.get("components")
    if not isinstance(components, list):
        warnings.append("components: ожидался массив")
        return warnings
    if not 1 <= len(components) <= 15:
        warnings.append(f"components: {len(components)} элементов, допустимо 1–15")

    weights: list[float] = []
    component_values: dict[str, list[float]] = {field: [] for field in ("kcal", "protein", "fat", "carbs")}

    for index, component in enumerate(components):
        if not isinstance(component, dict):
            warnings.append(f"components[{index}]: ожидался объект")
            continue

        missing_component = COMPONENT_FIELDS - set(component)
        extra_component = set(component) - COMPONENT_FIELDS
        if missing_component:
            warnings.append(f"components[{index}]: нет полей {', '.join(sorted(missing_component))}")
        if extra_component:
            warnings.append(f"components[{index}]: лишние поля {', '.join(sorted(extra_component))}")

        name = component.get("name")
        if not isinstance(name, str) or not name.strip():
            warnings.append(f"components[{index}].name: ожидалась непустая строка")

        weight = component.get("weight_g")
        if is_number(weight) and weight > 0:
            weights.append(float(weight))
        else:
            warnings.append(f"components[{index}].weight_g: ожидалось число > 0")

        for field in ("kcal", "protein", "fat", "carbs"):
            value = component.get(field)
            if is_number(value) and value >= 0:
                component_values[field].append(float(value))
            else:
                warnings.append(f"components[{index}].{field}: ожидалось число ≥ 0")

    if len(weights) == len(components):
        total_weight = sum(weights)
        tolerance = portion_g * 0.10
        if abs(total_weight - portion_g) > tolerance:
            warnings.append(
                f"сумма весов {total_weight:g} г отличается от порции "
                f"{portion_g:g} г больше чем на 10%"
            )

    if len(component_values["kcal"]) == len(components):
        for field in ("kcal", "protein", "fat", "carbs"):
            top_value = answer.get(field)
            if not is_number(top_value):
                continue
            component_sum = sum(component_values[field])
            tolerance = max(1.0, abs(top_value) * 0.02)
            if abs(component_sum - top_value) > tolerance:
                warnings.append(
                    f"{field}: верхнее значение {top_value:g} не совпадает "
                    f"с суммой компонентов {component_sum:g}"
                )

    return warnings


# ---------------------------------------------------------------------------
# Каталог и выбор блюд
# ---------------------------------------------------------------------------

def load_catalog() -> list[dict]:
    """Читает food_catalog.json и проверяет, что это непустой массив."""
    if not CATALOG_PATH.is_file():
        raise RuntimeError(f"Не найден каталог: {CATALOG_PATH}")

    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Не удалось прочитать {CATALOG_PATH}: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise RuntimeError(f"{CATALOG_PATH} должен содержать непустой JSON-массив")
    return data


def dish_key(dish: dict) -> str:
    """Имя файла без расширения: data/raw/dishes/dish_01.jpeg -> dish_01."""
    return Path(dish.get("photo", "")).stem


def matches_query(dish: dict, query: str) -> bool:
    """Ищет блюдо по dish_id, имени файла фото или пути к фото."""
    query = query.strip().lower()
    candidates = {
        str(dish.get("dish_id", "")).lower(),
        dish_key(dish).lower(),
        Path(dish.get("photo", "")).name.lower(),
        str(dish.get("photo", "")).lower(),
    }
    return query in candidates


def select_dishes(catalog: list[dict], args: argparse.Namespace) -> list[dict]:
    """Выбирает блюда по --self-test / --dish / --limit."""
    if args.self_test:
        queries: Optional[list[str]] = ["dish_01"]
    elif args.dish:
        queries = list(args.dish)
    else:
        queries = None

    if queries is None:
        selected = list(catalog)
    else:
        selected = []
        for query in queries:
            matches = [dish for dish in catalog if matches_query(dish, query)]
            if not matches:
                raise RuntimeError(f"Блюдо не найдено в каталоге: {query}")
            selected.extend(matches)

    # Убираем дубликаты, сохраняя порядок каталога.
    unique = []
    seen = set()
    for dish in selected:
        key = dish_key(dish)
        if key not in seen:
            seen.add(key)
            unique.append(dish)

    if args.limit is not None:
        unique = unique[: args.limit]
    return unique


# ---------------------------------------------------------------------------
# Основной прогон
# ---------------------------------------------------------------------------

def process_dish(
    client: Optional[OpenCodeClient],
    dish: dict,
    args: argparse.Namespace,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Обрабатывает одно блюдо.

    Возвращает (ответ, id сессии). В dry-run ответ и сессия — None.
    """
    key = dish_key(dish)
    image_path = BASE_DIR / dish["photo"]
    if not image_path.is_file():
        raise RuntimeError(f"Фото не найдено: {image_path}")

    portion_g = float(dish["portion_g"])
    prompt = build_prompt(portion_g)

    if args.dry_run:
        print("  [dry-run] промпт, который был бы отправлен:")
        for line in prompt.splitlines():
            print(f"    {line}")
        return None, None

    assert client is not None  # в dry-run сюда не попадаем

    session_id = client.create_session(
        title=f"nut-01 {key} {dish.get('dish_id', '')}".strip()
    )
    print(f"  сессия: {session_id}")

    raw_text = client.ask(session_id, image_path, prompt, timeout=args.timeout)

    try:
        answer = extract_json_object(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        # Если JSON не разобрался, сохраняем сырой текст — его можно изучить вручную.
        raw_path = args.out / f"{key}.raw.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_text, encoding="utf-8")
        raise RuntimeError(f"Не удалось разобрать JSON: {exc}. Сырой ответ: {raw_path}") from exc

    for warning in validate_answer(answer, portion_g):
        print(f"  ВНИМАНИЕ: {warning}")

    return answer, session_id


def save_answer(path: Path, answer: dict) -> None:
    """Сохраняет ответ как обычный JSON с отступами и переводом строки в конце."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(answer, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    """Выполняет прогон и возвращает код выхода (0 — успех)."""
    catalog = load_catalog()
    dishes = select_dishes(catalog, args)
    if not dishes:
        print("Нет блюд для обработки.")
        return 1

    print(f"Каталог: {CATALOG_PATH}")
    print(f"Модель:  {PROVIDER_ID}/{MODEL_ID}")
    print(f"Блюд:    {len(dishes)}")
    print(f"Выход:   {args.out}")
    print()

    client: Optional[OpenCodeClient] = None
    if not args.dry_run:
        base_url = discover_server_url()
        password = read_password()
        print(f"OpenCode API: {base_url}")
        print()
        client = OpenCodeClient(base_url, password)

    results: list[tuple[dict, bool, Optional[str]]] = []

    for index, dish in enumerate(dishes, 1):
        key = dish_key(dish)
        print(
            f"[{index}/{len(dishes)}] {key} "
            f"({dish.get('dish_id', '?')}), порция {dish['portion_g']} г"
        )
        print(f"  фото: {BASE_DIR / dish['photo']}")

        try:
            answer, _session_id = process_dish(client, dish, args)
            if args.dry_run:
                print("  [dry-run] ответ не сохраняется")
            else:
                assert answer is not None
                out_path = args.out / f"{key}.json"
                save_answer(out_path, answer)
                print(f"  сохранено: {out_path}")
            results.append((dish, True, None))
        except Exception as exc:  # noqa: BLE001 — хотим продолжить остальные блюда
            print(f"  ОШИБКА: {exc}")
            results.append((dish, False, str(exc)))
        print()

    succeeded = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - succeeded

    if args.self_test:
        print("=" * 60)
        if args.dry_run:
            print("SELF-TEST (dry-run): API не вызывался, файл не проверялся.")
            return 0

        out_path = args.out / "dish_01.json"
        if not out_path.is_file():
            print(f"SELF-TEST FAILED: файл не создан: {out_path}")
            return 1
        try:
            saved = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"SELF-TEST FAILED: {out_path} не читается как JSON: {exc}")
            return 1
        if not isinstance(saved, dict) or "components" not in saved:
            print(f"SELF-TEST FAILED: в {out_path} нет поля components")
            return 1
        print(
            f"SELF-TEST OK: {out_path} содержит валидный JSON "
            f"({len(saved['components'])} компонентов)."
        )
        return 0

    print("=" * 60)
    print(f"Готово: успешно {succeeded}, с ошибками {failed}.")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогон фото блюд через DeepSeek V4.1 Flash Free (локальный API OpenCode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Примеры:
  # Проверка на одном блюде dish_01
  python3 workspace/works/nut-01/tools/run_eval.py --self-test

  # Одно конкретное блюдо
  python3 workspace/works/nut-01/tools/run_eval.py --dish dish_01

  # Все блюда
  python3 workspace/works/nut-01/tools/run_eval.py

  # Только посмотреть промпт, без запроса к модели
  python3 workspace/works/nut-01/tools/run_eval.py --dish dish_01 --dry-run

Перед запуском должен работать OpenCode Desktop (или фоновая служба OpenCode).
""",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="прогнать только dish_01 и проверить, что JSON сохранился",
    )
    parser.add_argument(
        "--dish",
        action="append",
        metavar="ID",
        help="блюдо: dish_01, NUT-01-001 или имя файла фото; можно указывать несколько раз",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="обработать только первые N блюд",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help=f"папка для результатов (по умолчанию {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        metavar="SEC",
        help="сколько секунд ждать ответ модели на одно блюдо (по умолчанию 300)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="показать промпт и пути, но не вызывать модель",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.limit is not None and args.limit < 1:
        print("--limit должен быть больше нуля", file=sys.stderr)
        return 2

    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
