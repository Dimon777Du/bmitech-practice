#!/usr/bin/env python3
"""
calculate_metrics.py — метрики качества NUT-01 по JSON-данным.

Что читает
----------
1. food_catalog.json — JSON-массив блюд. Эталон берётся из:
       total.kcal, total.protein_g, total.fat_g, total.carbs_g
   Поле ingredients используется только для справочного количества
   эталонных ингредиентов.
2. Все *.json из папки прогона (по умолчанию out/runs/DeepSeek_v1/).
   Предсказание берётся из верхнеуровневых полей:
       kcal, protein, fat, carbs
   Поле components используется только для справочного количества
   компонентов.

Что считает
-----------
MAE % (средняя абсолютная процентная ошибка) для пары «блюдо × нутриент»:

    e(i, j) = |pred(i, j) - true(i, j)| / max(true(i, j), floor(j)) * 100

где floor(kcal) = 10, floor(белки/жиры/углеводы) = 1.
MAE_i — среднее e(i, j) по четырём нутриентам.
Итоговый MAE % — среднее MAE_i по всем блюдам каталога.

Match % — доля пар «блюдо × нутриент», для которых e(i, j) <= 20 %:

    Match % = hits / (n_dishes * 4) * 100

Если для блюда нет JSON-файла или в предсказании отсутствует/некорректно
хотя бы одно из полей kcal/protein/fat/carbs, блюдо считается проваленным:
MAE_i = 100 %, все четыре hit = 0 (правило из spec.md).

Результат сохраняется в out/metrics.json. CSV не используется.

Запуск
------
    python3 tools/calculate_metrics.py

Другие прогоны и пути:

    python3 tools/calculate_metrics.py \
        --runs-dir out/runs/AnotherModel \
        --out out/metrics_another.json

Только стандартная библиотека Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Пути и константы
# ---------------------------------------------------------------------------

# tools/calculate_metrics.py -> tools -> nut-01
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = BASE_DIR / "food_catalog.json"
DEFAULT_RUNS_DIR = BASE_DIR / "out" / "runs" / "DeepSeek_v1"
DEFAULT_OUT_PATH = BASE_DIR / "out" / "metrics.json"

DEFAULT_MATCH_TOLERANCE = 20.0  # %
DEFAULT_MAE_THRESHOLD = 15.0    # %
DEFAULT_MATCH_THRESHOLD = 80.0  # %


@dataclass(frozen=True)
class NutrientSpec:
    """Соответствие нутриента между каталогом и ответом модели."""

    key: str                 # внутренний ключ: kcal / protein / fat / carbs
    catalog_field: str       # поле в total каталога
    prediction_field: str    # поле в JSON предсказания
    label: str               # подпись для отчёта
    unit: str                # единица измерения
    floor: float             # защита от деления на ноль


NUTRIENTS: Tuple[NutrientSpec, ...] = (
    NutrientSpec("kcal", "kcal", "kcal", "Калории", "ккал", 10.0),
    NutrientSpec("protein", "protein_g", "protein", "Белки", "г", 1.0),
    NutrientSpec("fat", "fat_g", "fat", "Жиры", "г", 1.0),
    NutrientSpec("carbs", "carbs_g", "carbs", "Углеводы", "г", 1.0),
)


@dataclass
class CatalogDish:
    """Одно блюдо из food_catalog.json."""

    dish_id: str
    key: str
    name: str
    total: Dict[str, float]
    n_ingredients: int


@dataclass
class Prediction:
    """Один JSON-файл предсказания."""

    path: Path
    data: Optional[dict]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def read_json(path: Path) -> object:
    """Читает JSON-файл и возвращает разобранное значение."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"не удалось прочитать файл: {exc}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"некорректный JSON: {exc}") from exc


def parse_number(value: object) -> Optional[float]:
    """Приводит значение к float.

    Принимает числа и числовые строки (в том числе с запятой как десятичным
    разделителем). Для bool, пустых строк и нечисловых значений — None.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip().replace("\u00a0", "").replace(" ", "")
        if not text:
            return None
        text = text.replace(",", ".")
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def catalog_dish_key(dish: dict, dish_id: str) -> str:
    """Ключ блюда для сопоставления с файлом прогона.

    Основной вариант — имя фото без расширения:
    data/raw/dishes/dish_01.jpeg -> dish_01.
    Если фото нет, используется dish_id.
    """
    photo = dish.get("photo")
    if isinstance(photo, str) and photo.strip():
        normalized = photo.replace("\\", "/").rstrip("/")
        filename = normalized.split("/")[-1]
        stem = Path(filename).stem
        if stem:
            return stem
    return dish_id


def rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Округляет число для JSON-отчёта, сохраняя None."""
    return None if value is None else round(float(value), digits)


def mean(values: Iterable[float]) -> Optional[float]:
    """Среднее значение или None для пустой последовательности."""
    values_list = list(values)
    return statistics.fmean(values_list) if values_list else None


# ---------------------------------------------------------------------------
# Загрузка каталога
# ---------------------------------------------------------------------------

def load_catalog(path: Path) -> List[CatalogDish]:
    """Читает и валидирует food_catalog.json."""
    if not path.is_file():
        raise SystemExit(f"Не найден каталог: {path}")

    try:
        raw = read_json(path)
    except ValueError as exc:
        raise SystemExit(f"Не удалось прочитать каталог {path}: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"{path} должен содержать непустой JSON-массив блюд")

    dishes: List[CatalogDish] = []
    seen_ids: set = set()
    seen_keys: set = set()

    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"{path}: элемент #{index} должен быть объектом")

        dish_id = str(item.get("dish_id") or "").strip()
        if not dish_id:
            raise SystemExit(f"{path}: у элемента #{index} нет dish_id")

        total = item.get("total")
        if not isinstance(total, dict):
            raise SystemExit(f"{dish_id}: поле total должно быть объектом")

        values: Dict[str, float] = {}
        for spec in NUTRIENTS:
            if spec.catalog_field not in total:
                raise SystemExit(
                    f"{dish_id}: в total нет поля {spec.catalog_field}"
                )
            value = parse_number(total[spec.catalog_field])
            if value is None or value < 0:
                raise SystemExit(
                    f"{dish_id}: total.{spec.catalog_field} должно быть "
                    f"числом >= 0, получено {total[spec.catalog_field]!r}"
                )
            values[spec.key] = value

        key = catalog_dish_key(item, dish_id)
        if dish_id in seen_ids:
            raise SystemExit(f"{path}: повторяющийся dish_id {dish_id}")
        if key in seen_keys:
            raise SystemExit(f"{path}: повторяющийся ключ блюда {key}")
        seen_ids.add(dish_id)
        seen_keys.add(key)

        ingredients = item.get("ingredients")
        n_ingredients = len(ingredients) if isinstance(ingredients, list) else 0

        dishes.append(
            CatalogDish(
                dish_id=dish_id,
                key=key,
                name=str(item.get("name") or "").strip(),
                total=values,
                n_ingredients=n_ingredients,
            )
        )

    return dishes


# ---------------------------------------------------------------------------
# Загрузка предсказаний
# ---------------------------------------------------------------------------

def store_prediction(
    predictions: Dict[str, Prediction],
    dish: CatalogDish,
    prediction: Prediction,
    warnings: List[str],
) -> None:
    """Сохраняет предсказание, разрешая дубликаты в пользу валидного файла."""
    existing = predictions.get(dish.key)
    if existing is None:
        predictions[dish.key] = prediction
        return

    if existing.error and not prediction.error:
        warnings.append(
            f"{prediction.path.name}: заменяет невалидный файл "
            f"{existing.path.name} для {dish.dish_id}"
        )
        predictions[dish.key] = prediction
        return

    warnings.append(
        f"{prediction.path.name}: дубликат предсказания для {dish.dish_id}; "
        f"используется {existing.path.name}"
    )


def load_predictions(
    runs_dir: Path,
    dishes: Sequence[CatalogDish],
) -> Tuple[Dict[str, Prediction], List[str], int]:
    """Читает все *.json из папки прогона и сопоставляет их с каталогом.

    Сопоставление:
      1. по полю dish_id внутри JSON, если оно есть;
      2. иначе по имени файла без расширения (dish_01.json -> dish_01).
    """
    if not runs_dir.is_dir():
        raise SystemExit(f"Не найдена папка с прогонами: {runs_dir}")

    files = sorted(runs_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"В папке {runs_dir} нет JSON-файлов")

    by_key = {dish.key: dish for dish in dishes}
    by_id = {dish.dish_id: dish for dish in dishes}

    predictions: Dict[str, Prediction] = {}
    warnings: List[str] = []

    for path in files:
        try:
            data = read_json(path)
        except ValueError as exc:
            dish = by_key.get(path.stem)
            if dish is None:
                warnings.append(
                    f"{path.name}: не удалось прочитать JSON ({exc}); "
                    f"файл пропущен"
                )
                continue
            store_prediction(
                predictions, dish, Prediction(path, None, str(exc)), warnings
            )
            continue

        if not isinstance(data, dict):
            dish = by_key.get(path.stem)
            message = "JSON должен быть объектом"
            if dish is None:
                warnings.append(f"{path.name}: {message}; файл пропущен")
                continue
            store_prediction(
                predictions, dish, Prediction(path, None, message), warnings
            )
            continue

        dish: Optional[CatalogDish] = None
        raw_dish_id = data.get("dish_id")
        if isinstance(raw_dish_id, str) and raw_dish_id.strip() in by_id:
            dish = by_id[raw_dish_id.strip()]
        elif path.stem in by_key:
            dish = by_key[path.stem]
        elif path.stem in by_id:
            dish = by_id[path.stem]

        if dish is None:
            warnings.append(
                f"{path.name}: не найдено соответствующее блюдо в каталоге; "
                f"файл пропущен"
            )
            continue

        store_prediction(predictions, dish, Prediction(path, data), warnings)

    return predictions, warnings, len(files)


# ---------------------------------------------------------------------------
# Расчёт метрик
# ---------------------------------------------------------------------------

def validate_prediction(
    prediction: Prediction,
) -> Tuple[Optional[Dict[str, float]], List[str]]:
    """Проверяет четыре верхнеуровневых поля предсказания.

    Возвращает (значения, предупреждения). Если хотя бы одно поле отсутствует
    или некорректно, блюдо считается проваленным; значения возвращаются
    частично, чтобы их было видно в отчёте.
    """
    if prediction.data is None:
        return None, [prediction.error or "нет данных предсказания"]

    values: Dict[str, float] = {}
    warnings: List[str] = []

    for spec in NUTRIENTS:
        field = spec.prediction_field
        if field not in prediction.data:
            warnings.append(f"нет поля {field}")
            continue

        value = parse_number(prediction.data.get(field))
        if value is None:
            warnings.append(f"{field}: ожидалось число")
        elif value < 0:
            warnings.append(f"{field}: отрицательное значение")
        else:
            values[spec.key] = value

    return (values if values else None), warnings


def make_ok_result(
    dish: CatalogDish,
    predicted: Dict[str, float],
    prediction_file: str,
    tolerance: float,
    n_components: Optional[int],
) -> dict:
    """Результат блюда с валидным предсказанием."""
    errors: Dict[str, float] = {}
    signed_errors: Dict[str, float] = {}
    hits: Dict[str, bool] = {}
    raw_errors: List[float] = []

    for spec in NUTRIENTS:
        reference = dish.total[spec.key]
        value = predicted[spec.key]
        denominator = max(reference, spec.floor)
        error = abs(value - reference) / denominator * 100.0
        raw_errors.append(error)
        errors[spec.key] = rounded(error)
        signed_errors[spec.key] = rounded(
            (value - reference) / denominator * 100.0
        )
        hits[spec.key] = error <= tolerance

    mae = rounded(mean(raw_errors))
    match = rounded(
        sum(1 for hit in hits.values() if hit) / len(NUTRIENTS) * 100.0
    )

    return {
        "dish_id": dish.dish_id,
        "dish_key": dish.key,
        "name": dish.name,
        "status": "ok",
        "prediction_file": prediction_file,
        "reference": dict(dish.total),
        "predicted": dict(predicted),
        "errors_percent": errors,
        "signed_errors_percent": signed_errors,
        "mae_percent": mae,
        "hits": hits,
        "match_percent": match,
        "reference_ingredients": dish.n_ingredients,
        "predicted_components": n_components,
        "warnings": [],
    }


def make_failed_result(
    dish: CatalogDish,
    reason: str,
    prediction_file: Optional[str] = None,
    predicted: Optional[Dict[str, float]] = None,
    n_components: Optional[int] = None,
) -> dict:
    """Результат проваленного блюда: MAE_i = 100 %, все hit = 0."""
    return {
        "dish_id": dish.dish_id,
        "dish_key": dish.key,
        "name": dish.name,
        "status": "failed",
        "prediction_file": prediction_file,
        "reference": dict(dish.total),
        "predicted": predicted,
        "errors_percent": {spec.key: 100.0 for spec in NUTRIENTS},
        "signed_errors_percent": {spec.key: None for spec in NUTRIENTS},
        "mae_percent": 100.0,
        "hits": {spec.key: False for spec in NUTRIENTS},
        "match_percent": 0.0,
        "reference_ingredients": dish.n_ingredients,
        "predicted_components": n_components,
        "warnings": [reason],
    }


def count_components(prediction: Optional[Prediction]) -> Optional[int]:
    """Справочное количество компонентов в ответе модели."""
    if prediction is None or not isinstance(prediction.data, dict):
        return None
    components = prediction.data.get("components")
    return len(components) if isinstance(components, list) else None


def compute_report(
    dishes: Sequence[CatalogDish],
    predictions: Dict[str, Prediction],
    *,
    catalog_path: Path,
    runs_dir: Path,
    out_path: Path,
    tolerance: float,
    mae_threshold: float,
    match_threshold: float,
    n_prediction_files: int,
    load_warnings: Sequence[str],
) -> dict:
    """Собирает полный отчёт по всем блюдам каталога."""
    dish_results: List[dict] = []
    failed_dishes: List[dict] = []

    for dish in dishes:
        prediction = predictions.get(dish.key)
        n_components = count_components(prediction)

        if prediction is None:
            result = make_failed_result(
                dish,
                "нет JSON-файла предсказания",
                n_components=n_components,
            )
        else:
            predicted, warnings = validate_prediction(prediction)
            if predicted is None or len(predicted) != len(NUTRIENTS):
                reason = "; ".join(warnings) if warnings else "невалидное предсказание"
                result = make_failed_result(
                    dish,
                    reason,
                    prediction_file=prediction.path.name,
                    predicted=predicted,
                    n_components=n_components,
                )
            else:
                result = make_ok_result(
                    dish,
                    predicted,
                    prediction.path.name,
                    tolerance,
                    n_components,
                )

        dish_results.append(result)
        if result["status"] == "failed":
            failed_dishes.append(
                {
                    "dish_id": dish.dish_id,
                    "dish_key": dish.key,
                    "name": dish.name,
                    "reason": "; ".join(result["warnings"]),
                }
            )

    n_dishes = len(dish_results)
    n_valid = sum(1 for result in dish_results if result["status"] == "ok")
    n_failed = n_dishes - n_valid
    pairs = n_dishes * len(NUTRIENTS)
    total_hits = sum(
        1
        for result in dish_results
        for spec in NUTRIENTS
        if result["hits"][spec.key]
    )

    overall_mae = mean(result["mae_percent"] for result in dish_results)
    overall_match = (
        total_hits / pairs * 100.0 if pairs else None
    )

    valid_results = [result for result in dish_results if result["status"] == "ok"]
    valid_pairs = len(valid_results) * len(NUTRIENTS)
    valid_hits = sum(
        1
        for result in valid_results
        for spec in NUTRIENTS
        if result["hits"][spec.key]
    )
    overall_mae_valid = mean(result["mae_percent"] for result in valid_results)
    overall_match_valid = (
        valid_hits / valid_pairs * 100.0 if valid_pairs else None
    )

    nutrient_report: Dict[str, dict] = {}
    for spec in NUTRIENTS:
        errors = [result["errors_percent"][spec.key] for result in dish_results]
        hits = [result["hits"][spec.key] for result in dish_results]
        valid_errors = [
            result["errors_percent"][spec.key]
            for result in dish_results
            if result["status"] == "ok"
        ]
        valid_hits_list = [
            result["hits"][spec.key]
            for result in dish_results
            if result["status"] == "ok"
        ]
        signed_errors = [
            result["signed_errors_percent"][spec.key]
            for result in dish_results
            if result["signed_errors_percent"][spec.key] is not None
        ]
        references = [result["reference"][spec.key] for result in dish_results]
        predictions_list = [
            result["predicted"][spec.key]
            for result in dish_results
            if result["status"] == "ok"
            and result["predicted"] is not None
            and spec.key in result["predicted"]
        ]

        nutrient_report[spec.key] = {
            "label": spec.label,
            "unit": spec.unit,
            "floor": spec.floor,
            "n": len(errors),
            "n_valid": len(valid_errors),
            "mae_percent": rounded(mean(errors)),
            "mae_percent_valid": rounded(mean(valid_errors)),
            "match_percent": rounded(
                sum(1 for hit in hits if hit) / len(hits) * 100.0
                if hits else None
            ),
            "match_percent_valid": rounded(
                sum(1 for hit in valid_hits_list if hit) / len(valid_hits_list) * 100.0
                if valid_hits_list else None
            ),
            "hits": sum(1 for hit in hits if hit),
            "bias_percent": rounded(mean(signed_errors)),
            "mean_reference": rounded(mean(references), 2),
            "mean_predicted": rounded(mean(predictions_list), 2),
        }

    report = {
        "meta": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "catalog": str(catalog_path.resolve()),
            "runs_dir": str(runs_dir.resolve()),
            "output": str(out_path.resolve()),
            "model": runs_dir.name,
            "n_catalog_dishes": n_dishes,
            "n_prediction_files": n_prediction_files,
            "n_matched": len(predictions),
            "n_valid": n_valid,
            "n_failed": n_failed,
            "python_version": platform.python_version(),
            "definitions": {
                "mae_percent": (
                    "|pred - true| / max(true, floor) * 100; MAE_i — среднее "
                    "по 4 нутриентам; итог — среднее MAE_i по всем блюдам."
                ),
                "match_percent": (
                    "Доля пар «блюдо × нутриент» с ошибкой <= tolerance; "
                    "итог — hits / (n_dishes * 4) * 100."
                ),
                "match_basis": "nutrient_pair_error <= tolerance",
                "tolerance_percent": tolerance,
                "floors": {spec.key: spec.floor for spec in NUTRIENTS},
                "failed_dish_policy": (
                    "Нет валидного JSON или хотя бы одного из полей "
                    "kcal/protein/fat/carbs -> MAE_i = 100 %, все hit = 0."
                ),
            },
            "warnings": list(load_warnings),
        },
        "summary": {
            "n_dishes": n_dishes,
            "n_valid": n_valid,
            "n_failed": n_failed,
            "pairs": pairs,
            "hits": total_hits,
            "mae_percent": rounded(overall_mae),
            "match_percent": rounded(overall_match),
            "mae_percent_valid": rounded(overall_mae_valid),
            "match_percent_valid": rounded(overall_match_valid),
            "mae_threshold": mae_threshold,
            "match_threshold": match_threshold,
            "mae_threshold_passed": (
                None if overall_mae is None else overall_mae <= mae_threshold
            ),
            "match_threshold_passed": (
                None if overall_match is None else overall_match >= match_threshold
            ),
        },
        "nutrients": nutrient_report,
        "dishes": dish_results,
        "failed_dishes": failed_dishes,
    }

    return report


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

def fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    """Форматирует число для консоли; None -> «—»."""
    if value is None:
        return "—"
    return f"{value:.{digits}f}{suffix}"


def fmt_bool(value: Optional[bool], yes: str = "ПРОЙДЕН", no: str = "НЕ ПРОЙДЕН") -> str:
    if value is None:
        return "—"
    return yes if value else no


def print_report(report: dict, verbose: bool = False) -> None:
    """Печатает отчёт в человекочитаемом виде."""
    meta = report["meta"]
    summary = report["summary"]

    print("=" * 78)
    print("МЕТРИКИ КАЧЕСТВА — NUT-01")
    print("=" * 78)
    print(f"Каталог:       {meta['catalog']}")
    print(f"Прогоны:       {meta['runs_dir']}")
    print(f"Модель:        {meta['model']}")
    print(
        f"Блюд:          {summary['n_dishes']} "
        f"(валидных: {summary['n_valid']}, провалено: {summary['n_failed']})"
    )
    print(f"Допуск Match:  ±{meta['definitions']['tolerance_percent']:g} %")

    print("\n1) MAE % и Match % по нутриентам (Match: ошибка пары <= допуска)")
    print("-" * 78)
    print(
        f"{'Нутриент':<12}{'N':>4}{'MAE %':>10}{'Match %':>10}"
        f"{'Ср. эталон':>14}{'Ср. прогноз':>14}"
    )
    for spec in NUTRIENTS:
        item = report["nutrients"][spec.key]
        print(
            f"{item['label']:<12}{item['n']:>4}"
            f"{fmt(item['mae_percent']):>10}"
            f"{fmt(item['match_percent']):>10}"
            f"{fmt(item['mean_reference']):>14}"
            f"{fmt(item['mean_predicted']):>14}"
        )

    print("-" * 78)
    print(
        f"MAE % (итог):   {fmt(summary['mae_percent'], suffix=' %')}  "
        f"порог ≤ {summary['mae_threshold']:g}: "
        f"{fmt_bool(summary['mae_threshold_passed'])}"
    )
    print(
        f"Match % (итог): {fmt(summary['match_percent'], suffix=' %')}  "
        f"({summary['hits']}/{summary['pairs']} пар)  "
        f"порог ≥ {summary['match_threshold']:g}: "
        f"{fmt_bool(summary['match_threshold_passed'])}"
    )
    if summary["n_failed"]:
        print(
            f"Только по валидным блюдам: "
            f"MAE {fmt(summary['mae_percent_valid'], suffix=' %')}, "
            f"Match {fmt(summary['match_percent_valid'], suffix=' %')}"
        )

    if report["failed_dishes"]:
        print("\n2) Проваленные блюда")
        print("-" * 78)
        for item in report["failed_dishes"]:
            print(f"  - {item['dish_key']} ({item['dish_id']}): {item['reason']}")

    warnings = meta.get("warnings") or []
    if warnings:
        print("\nПредупреждения:")
        for warning in warnings:
            print(f"  - {warning}")

    if verbose:
        print("\nДетали по блюдам")
        print("-" * 78)
        print(
            f"{'Ключ':<10}{'Dish ID':<13}{'MAE %':>9}{'Match %':>9}"
            f"{'Статус':>12}"
        )
        for item in report["dishes"]:
            print(
                f"{item['dish_key']:<10}{item['dish_id']:<13}"
                f"{fmt(item['mae_percent']):>9}"
                f"{fmt(item['match_percent']):>9}"
                f"{item['status']:>12}"
            )

    print("\n" + "=" * 78)


def write_report(report: dict, path: Path) -> None:
    """Атомарно сохраняет отчёт в JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Расчёт MAE % и Match % для NUT-01 по food_catalog.json "
            "и JSON-файлам прогона."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python3 tools/calculate_metrics.py\n"
            "  python3 tools/calculate_metrics.py "
            "--runs-dir out/runs/DeepSeek_v1 --out out/metrics.json\n"
        ),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help=f"путь к food_catalog.json (по умолчанию {DEFAULT_CATALOG_PATH})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"папка с JSON-предсказаниями (по умолчанию {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"путь к итоговому JSON (по умолчанию {DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=DEFAULT_MATCH_TOLERANCE,
        metavar="PCT",
        help=f"допуск ошибки для hit, %% (по умолчанию {DEFAULT_MATCH_TOLERANCE:g})",
    )
    parser.add_argument(
        "--mae-threshold",
        type=float,
        default=DEFAULT_MAE_THRESHOLD,
        metavar="PCT",
        help=f"порог MAE %%, %% (по умолчанию {DEFAULT_MAE_THRESHOLD:g})",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        metavar="PCT",
        help=f"порог Match %%, %% (по умолчанию {DEFAULT_MATCH_THRESHOLD:g})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="показать метрики по каждому блюду",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.match_tolerance <= 0:
        print("--match-tolerance должен быть больше нуля", file=sys.stderr)
        return 2
    if args.mae_threshold < 0 or args.match_threshold < 0:
        print("Пороги не могут быть отрицательными", file=sys.stderr)
        return 2

    dishes = load_catalog(args.catalog)
    predictions, load_warnings, n_prediction_files = load_predictions(
        args.runs_dir, dishes
    )

    report = compute_report(
        dishes,
        predictions,
        catalog_path=args.catalog,
        runs_dir=args.runs_dir,
        out_path=args.out,
        tolerance=args.match_tolerance,
        mae_threshold=args.mae_threshold,
        match_threshold=args.match_threshold,
        n_prediction_files=n_prediction_files,
        load_warnings=load_warnings,
    )

    print_report(report, verbose=args.verbose)
    write_report(report, args.out)
    print(f"Отчёт сохранён: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
