#!/usr/bin/env python3
"""
calculate_metrics.py — метрики качества NUT-01 по JSON-данным.

Что читает
----------
1. food_catalog.json — JSON-массив блюд. Эталон берётся из:
       total.kcal, total.protein_g, total.fat_g, total.carbs_g
   Поле ingredients используется для Match % по ингредиентам:
       ingredients[].name — эталонный состав.
2. Все *.json из папки прогона (по умолчанию out/runs/DeepSeek_v1/).
   Нутриенты берутся из верхнеуровневых полей:
       kcal, protein, fat, carbs
   Поле components используется для Match % по ингредиентам:
       components[].name — распознанный состав.

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

Match % по ингредиентам (поле match_ingredients) считается отдельно:
    эталон — food_catalog.json -> ingredients[].name;
    предсказание — run JSON -> components[].name.
Это метрика распознавания состава, поэтому вес компонента в неё не входит:
    recall_i = matched_reference / reference_total * 100
    match_ingredients.match_percent = recall по всем блюдам (micro)
Дополнительно считаются precision, F1, macro-средние и детали по блюдам.
Названия нормализуются: нижний регистр, ё -> е, пунктуация, стоп-слова,
простой стеммер и словарь синонимов; содержимое скобок используется как
уточнение, но не может дать совпадение само по себе, если головные слова
обеих сторон разные. Матчинг один-к-одному, оптимальный по числу пар.

Дополнительно в match_ingredients.weight_constrained считается
Component Match % из spec.md 5.3 — F1 с проверкой относительной ошибки веса
<= 20 % у совпавших по названию компонентов. Это отдельная метрика, она не
подменяет основной Match % по ингредиентам.

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
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
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
DEFAULT_INGREDIENT_WEIGHT_TOLERANCE = 20.0  # %


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


@dataclass(frozen=True)
class Ingredient:
    """Ингредиент эталона или компонент предсказания."""

    name: str
    weight_g: Optional[float]


@dataclass
class CatalogDish:
    """Одно блюдо из food_catalog.json."""

    dish_id: str
    key: str
    name: str
    total: Dict[str, float]
    ingredients: List[Ingredient]

    @property
    def n_ingredients(self) -> int:
        return len(self.ingredients)


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
# Сопоставление ингредиентов
# ---------------------------------------------------------------------------

# Служебные слова и способы приготовления, которые не помогают отличить
# один ингредиент от другого.
INGREDIENT_NOISE = {
    "свежий", "свежая", "свежее", "свежие", "fresh",
    "вареный", "варёный", "вареная", "варёная", "вареное", "варёное",
    "варенные", "варёные", "boiled",
    "жареный", "жареная", "жареное", "жареные", "жареного", "fried",
    "запеченный", "запечённый", "запеченная", "запечённая",
    "запеченное", "запечённое", "запеченные", "запечённые", "baked",
    "сырой", "сырая", "сырое", "сырые", "raw",
    "молотый", "молотая", "молотое", "молотые", "ground",
    "рубленый", "рубленая", "рубленое", "рубленые", "chopped",
    "резаный", "резаная", "резаное", "резаные", "нарезки", "нарезка",
    "крупный", "крупная", "крупное", "крупные",
    "мелкий", "мелкая", "мелкое", "мелкие",
    "готовый", "готовая", "готовое", "готовые", "готового", "готовой",
    "по", "вкусу", "to", "taste", "в", "и", "с", "на", "из", "для",
    "или", "основе", "шт", "при", "фритюре", "жарке", "обжарке",
    "питьевая", "питьевой", "рафинированное", "рафинированный",
    "дезодорированное", "дезодорированный",
    "впитанное", "впитанный", "впитанная",
    "предположительно", "примерно", "около", "порция", "порции",
    "консервированные", "консервированный", "свежемолотый", "свежемолотая",
    "йодированная", "йодированный", "основа", "основе", "основой",
    "корж", "бортик", "ломтик", "ломтики", "кусочки", "кусочками",
    "кусочков", "листья", "лист", "микс", "тертый", "тёртый",
    "кубиками", "поливка", "заправка", "суши", "гриль", "гриля",
    "добавки", "добавка",
    "2", "3", "9", "30", "см",
}

# Общие слова: сами по себе не доказывают совпадение, если у обеих сторон
# есть более конкретные слова.
INGREDIENT_WEAK = {"мясн", "рыб", "соус", "сыр", "масл", "овощ"}

# Головные слова, которые считаются родовыми. Если головное слово родовое,
# совпадение по уточнению в скобках допустимо.
INGREDIENT_GENERIC_HEADS = INGREDIENT_WEAK | {
    "спец", "приправ", "соль", "зелен", "бифштекс", "котлет", "дёнер",
}

# Каноническое слово -> варианты написания. Один вариант может входить
# в несколько групп (например, «моцарелла» -> и «моцарелл», и «сыр»).
INGREDIENT_SYNONYM_GROUPS = {
    "томат": (
        "помидор", "помидоры", "помидора", "помидоров", "помидорами",
        "томат", "томаты", "томатов", "томатный", "томатная", "томатное",
        "томатные",
    ),
    "куриц": (
        "курица", "курицы", "курицу", "курицей", "куриный", "куриная",
        "куриное", "куриные", "куриного", "куриной", "цыпленок", "цыплёнок",
        "цыпленка", "цыплёнка", "мясн",
    ),
    "хлеб": (
        "хлеб", "хлеба", "хлебный", "хлебная", "хлебное", "хлебные",
        "булочка", "булочки", "булочку", "булочек", "лаваш", "лаваша",
        "лавашом", "лепешка", "лепёшка", "лепешки", "лепёшки", "тортилья",
        "тортильи", "тортилью", "панировка", "панировки", "панировочные",
        "панировочный", "сухари", "сухарики", "сухарь", "сухаря",
    ),
    "сыр": (
        "сыр", "сыра", "сыром", "сыры", "сырный", "сырная", "сырное",
        "сырные", "чеддер", "чеддера", "моцарелла", "моцареллы",
        "моцареллу", "пармезан", "пармезана", "фета", "феты", "фету",
        "сулугуни", "имеретинский", "имеретинского", "эмменталь",
        "эмменталя", "гауда", "гауды",
    ),
    "моцарелл": ("моцарелла", "моцареллы", "моцареллу"),
    "пармезан": ("пармезан", "пармезана"),
    "фет": ("фета", "феты", "фету"),
    "сулугуни": ("сулугуни",),
    "имеретинск": ("имеретинский", "имеретинского"),
    "эмменталь": ("эмменталь", "эмменталя"),
    "гауда": ("гауда", "гауды"),
    "чеддер": ("чеддер", "чеддера"),
    "соус": ("соус", "соуса", "соусом", "соусы", "соусный", "соусная"),
    "кетчуп": ("кетчуп", "кетчупа"),
    "альфредо": ("альфредо",),
    "терияки": ("терияки",),
    "тахини": ("тахини",),
    "горчичн": ("горчичный", "горчичная"),
    "чесночн": ("чесночный", "чесночная"),
    "барбекю": ("барбекю",),
    "цезар": ("цезарь",),
    "салатн": ("салатный", "салатная"),
    "сливочн": ("сливочное", "сливочный", "сливочная", "сливки"),
    "масл": (
        "масло", "масла", "маслом", "масел", "масляный", "масляная",
        "подсолнечное", "подсолнечный", "подсолнечная", "растительное",
        "растительный", "растительная", "оливковое", "оливковый",
        "оливковая", "сливочное", "сливочный", "сливочная",
    ),
    "подсолнечн": ("подсолнечное", "подсолнечный", "подсолнечная"),
    "растительн": ("растительное", "растительный", "растительная"),
    "оливков": ("оливковое", "оливковый", "оливковая"),
    "картофел": (
        "картофель", "картофеля", "картошка", "картошки", "картошку",
        "картофельный", "картофельные",
    ),
    "яйц": (
        "яйцо", "яйца", "яиц", "яйцом", "яичный", "яичная", "яичное",
        "яичные", "желток", "желтки", "желтков", "желтка",
    ),
    "желток": ("желток", "желтки", "желтков", "желтка"),
    "тест": ("тесто", "теста", "тестом", "тесте", "тестовый"),
    "спец": (
        "специи", "специя", "специй", "специями", "приправа", "приправы",
        "приправой", "соль", "соли", "солю",
    ),
    "приправ": ("приправа", "приправы", "приправой"),
    "соль": ("соль", "соли", "солю", "соленый", "соленая"),
    "зелен": (
        "зелень", "зелени", "зеленью", "укроп", "укропа", "кинза",
        "кинзы", "петрушка", "петрушки", "базилик", "базилика",
    ),
    "укроп": ("укроп", "укропа"),
    "кинз": ("кинза", "кинзы"),
    "петрушк": ("петрушка", "петрушки"),
    "базилик": ("базилик", "базилика"),
    "овощ": ("овощи", "овощ", "овощами", "овощей", "овощной", "овощная"),
    "морков": ("морковь", "моркови", "морковью", "морковка", "морковки"),
    "свекл": ("свекла", "свёкла", "свеклы", "свёклы", "свеклу"),
    "капуст": ("капуста", "капусты", "капусту", "капустой"),
    "лук": ("лук", "лука", "луком", "луке", "луковый", "луковая"),
    "чеснок": ("чеснок", "чеснока", "чесноком"),
    "огурец": (
        "огурец", "огурца", "огурцы", "огурцов", "огурчики", "огурчик",
    ),
    "болгарск": ("болгарский", "болгарская", "болгарское", "болгарские"),
    "сладк": ("сладкий", "сладкая", "сладкое", "сладкие"),
    "рис": (
        "рис", "риса", "рисом", "рисовый", "рисовая", "рисовое",
        "рисовые",
    ),
    "говядин": (
        "говядина", "говядины", "говядиной", "говяжий", "говяжья",
        "говяжье", "мясн",
    ),
    "свинин": (
        "свинина", "свинины", "свининой", "свиной", "свиная", "свиное",
        "мясн",
    ),
    "бекон": (
        "бекон", "бекона", "беконом", "грудинка", "грудинки", "гуанчале",
        "мясн",
    ),
    "грудинк": ("грудинка", "грудинки"),
    "гуанчале": ("гуанчале",),
    "ветчин": ("ветчина", "ветчины", "ветчиной", "мясн"),
    "салями": ("салями", "мясн"),
    "пепперони": ("пепперони", "мясн"),
    "фарш": ("фарш", "фарша", "фаршем", "мясн"),
    "котлет": (
        "котлета", "котлеты", "котлету", "бифштекс", "бифштекса", "мясн",
    ),
    "бифштекс": ("бифштекс", "бифштекса", "мясн"),
    "лосос": (
        "лосось", "лосося", "лососем", "семга", "сёмга", "семги", "сёмги",
        "рыб",
    ),
    "семг": ("семга", "сёмга", "семги", "сёмги", "рыб"),
    "кревет": ("креветка", "креветки", "креветок", "рыб"),
    "тунец": ("тунец", "тунца", "рыб"),
    "икр": ("икра", "икры", "тобико", "рыб"),
    "тобико": ("тобико",),
    "нори": ("нори",),
    "васаб": ("васаби",),
    "имбир": ("имбирь", "имбиря"),
    "молок": (
        "молоко", "молока", "молоком", "молочный", "молочная",
        "молочное",
    ),
    "сметан": ("сметана", "сметаны", "сметаной"),
    "кефир": ("кефир", "кефира"),
    "овсян": ("овсяные", "овсяный", "овсяная", "овсяное", "овсянка"),
    "хлоп": ("хлопья", "хлопьев"),
    "киноа": ("киноа",),
    "нут": ("нут", "нута"),
    "авокадо": ("авокадо",),
    "изюм": ("изюм", "изюма"),
    "шампиньон": ("шампиньоны", "шампиньон"),
    "гриб": ("грибы", "гриб", "грибов", "шампиньоны", "шампиньон"),
    "вод": ("вода", "воды", "водой"),
    "бульон": ("бульон", "бульона", "бульоном"),
    "суп": ("суп", "супа"),
    "борщ": ("борщ", "борща"),
    "каш": ("каша", "каши", "кашу"),
    "варень": ("варенье", "варенья", "конфитюр", "джем"),
    "конфитюр": ("конфитюр",),
    "джем": ("джем",),
    "ягод": ("ягоды", "ягодный", "ягодная", "ягодное", "ягодные"),
    "ананас": ("ананас", "ананасы", "ананасов"),
    "оливк": ("оливки", "оливка", "оливок"),
    "паст": ("паста",),
    "спагетт": ("спагетти",),
    "макарон": ("макароны",),
    "лапш": ("лапша", "лапши"),
    "удон": ("удон",),
    "кунжут": ("кунжут", "кунжута"),
    "соев": ("соевый", "соевая", "соевое"),
    "начинк": ("начинка", "начинки"),
}

INGREDIENT_SYNONYMS: Dict[str, Tuple[str, ...]] = {}
for _canonical, _variants in INGREDIENT_SYNONYM_GROUPS.items():
    for _variant in _variants:
        INGREDIENT_SYNONYMS.setdefault(_variant, ())
        INGREDIENT_SYNONYMS[_variant] += (_canonical,)

_RU_SUFFIXES = (
    "иями", "ями", "ами", "ов", "ев", "ей", "ой", "ый", "ий",
    "ая", "яя", "ое", "ее", "ые", "ие", "ах", "ях", "ом", "ем",
    "ам", "ям", "ую", "юю", "у", "ю", "а", "я", "ы", "и", "е", "о",
)


@dataclass(frozen=True)
class IngredientTokens:
    """Нормализованное название ингредиента."""

    all: frozenset
    main: frozenset
    head: frozenset


def _stem_ru(word: str) -> str:
    """Очень простой стеммер для русских существительных и прилагательных."""
    for suffix in _RU_SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _normalize_ingredient_words(text: str) -> Tuple[frozenset, frozenset]:
    """Возвращает (все токены, головные токены) для фрагмента названия."""
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    tokens = set()
    head: frozenset = frozenset()

    for word in text.split():
        if not word or word in INGREDIENT_NOISE or word.isdigit():
            continue
        forms = {word, _stem_ru(word)}
        for form in list(forms):
            for synonym in INGREDIENT_SYNONYMS.get(form, ()):
                forms.add(synonym)
        forms = {
            form for form in forms
            if form and form not in INGREDIENT_NOISE
        }
        if not forms:
            continue
        if not head:
            head = frozenset(forms)
        tokens.update(forms)

    return frozenset(tokens), head


def normalize_ingredient_name(name: str) -> IngredientTokens:
    """Нормализует название, отделяя содержимое скобок от головной части."""
    text = unicodedata.normalize("NFKC", str(name)).strip().lower()
    text = text.replace("ё", "е")
    parenthetical = re.findall(r"\(([^)]*)\)", text)
    main_text = re.sub(r"\([^)]*\)", " ", text)

    main, head = _normalize_ingredient_words(main_text)
    extra = set()
    for part in parenthetical:
        part_tokens, _ = _normalize_ingredient_words(part)
        extra.update(part_tokens)

    return IngredientTokens(
        all=main | frozenset(extra),
        main=main,
        head=head,
    )


def _head_is_generic(head: frozenset) -> bool:
    """True, если все головные токены — родовые слова."""
    return bool(head) and not (head - INGREDIENT_GENERIC_HEADS)


def _ingredient_pair_score(
    reference: IngredientTokens,
    predicted: IngredientTokens,
) -> Optional[int]:
    """Оценка пары «эталонный ингредиент × компонент» или None.

    Совпадение по скобкам допускается только тогда, когда хотя бы одно
    головное слово родовое. Это не даёт «тесту» совпасть с «маслом» только
    потому, что в описании масла упомянуто «в тесте».
    """
    shared_all = reference.all & predicted.all
    if not shared_all:
        return None

    shared_main = reference.main & predicted.main
    strong_main = shared_main - INGREDIENT_WEAK
    strong_all = shared_all - INGREDIENT_WEAK

    if not strong_all:
        # Совпали только родовые слова. Это допустимо, если хотя бы одна
        # сторона состоит только из родовых слов («сыр» vs «сыр моцарелла»).
        if (reference.all - INGREDIENT_WEAK) and (
            predicted.all - INGREDIENT_WEAK
        ):
            return None
        return 1 + len(shared_all)

    if not strong_main and not (
        _head_is_generic(reference.head) or _head_is_generic(predicted.head)
    ):
        return None

    head_shared = reference.head & predicted.head
    return (
        1
        + 4 * len(strong_main)
        + 2 * len(strong_all - strong_main)
        + (2 if head_shared else 0)
        + len(shared_all)
    )


def ingredient_weight_error(
    reference_weight: Optional[float],
    predicted_weight: Optional[float],
) -> Optional[float]:
    """Относительная ошибка веса, %; None, если веса неизвестны."""
    if reference_weight is None or predicted_weight is None:
        return None
    return abs(predicted_weight - reference_weight) / max(
        reference_weight, 1.0
    ) * 100.0


def match_ingredient_lists(
    references: Sequence[Ingredient],
    predictions: Sequence[Ingredient],
    *,
    weight_tolerance: Optional[float] = None,
) -> dict:
    """Оптимальное матчинг один-к-одному эталонных ингредиентов и компонентов.

    Если weight_tolerance задан, пара дополнительно требует известных весов
    и относительной ошибки веса <= weight_tolerance.
    """
    reference_tokens = [
        normalize_ingredient_name(item.name) for item in references
    ]
    prediction_tokens = [
        normalize_ingredient_name(item.name) for item in predictions
    ]

    candidates: Dict[Tuple[int, int], Tuple[int, Optional[float]]] = {}
    for ref_index, (reference, ref_tokens) in enumerate(
        zip(references, reference_tokens)
    ):
        for pred_index, (prediction, pred_tokens) in enumerate(
            zip(predictions, prediction_tokens)
        ):
            score = _ingredient_pair_score(ref_tokens, pred_tokens)
            if score is None:
                continue
            weight_error = ingredient_weight_error(
                reference.weight_g, prediction.weight_g
            )
            if weight_tolerance is not None and (
                weight_error is None or weight_error > weight_tolerance
            ):
                continue
            candidates[(ref_index, pred_index)] = (score, weight_error)

    n_references = len(references)
    n_predictions = len(predictions)
    pairs: Tuple[Tuple[int, int, int, Optional[float]], ...] = ()

    if candidates:
        @lru_cache(maxsize=None)
        def solve(
            ref_index: int,
            used_predictions: int,
        ) -> Tuple[int, int, float, Tuple[Tuple[int, int, int, Optional[float]], ...]]:
            if ref_index == n_references:
                return (0, 0, 0.0, ())

            best = solve(ref_index + 1, used_predictions)

            for pred_index in range(n_predictions):
                bit = 1 << pred_index
                if used_predictions & bit:
                    continue
                candidate = candidates.get((ref_index, pred_index))
                if candidate is None:
                    continue

                score, weight_error = candidate
                rest = solve(ref_index + 1, used_predictions | bit)
                total = (
                    rest[0] + 1,
                    rest[1] + score,
                    rest[2] + (weight_error or 0.0),
                    ((ref_index, pred_index, score, weight_error),) + rest[3],
                )
                if (total[0], total[1], -total[2]) > (
                    best[0], best[1], -best[2]
                ):
                    best = total

            return best

        _, _, _, pairs = solve(0, 0)

    matches = []
    for ref_index, pred_index, score, weight_error in pairs:
        matches.append(
            {
                "reference": references[ref_index].name,
                "predicted": predictions[pred_index].name,
                "reference_weight_g": references[ref_index].weight_g,
                "predicted_weight_g": predictions[pred_index].weight_g,
                "weight_error_percent": rounded(weight_error),
                "score": score,
            }
        )

    matched_reference = {pair[0] for pair in pairs}
    matched_predicted = {pair[1] for pair in pairs}
    weight_errors = [
        weight_error
        for _, _, _, weight_error in pairs
        if weight_error is not None
    ]

    if references:
        recall = len(matches) / len(references) * 100.0
    else:
        recall = 100.0 if not predictions else 0.0

    if predictions:
        precision = len(matches) / len(predictions) * 100.0
    else:
        precision = 100.0 if not references else 0.0

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "matched": len(matches),
        "reference_total": len(references),
        "predicted_total": len(predictions),
        "match_percent": rounded(recall),
        "recall_percent": rounded(recall),
        "precision_percent": rounded(precision),
        "f1_percent": rounded(f1),
        "weight_mae_percent": rounded(mean(weight_errors)),
        "matches": matches,
        "unmatched_reference": [
            references[index].name
            for index in range(len(references))
            if index not in matched_reference
        ],
        "unmatched_predicted": [
            predictions[index].name
            for index in range(len(predictions))
            if index not in matched_predicted
        ],
    }


def ingredient_match_metrics(match: dict) -> dict:
    """Компактная сводка по одному результату матчинга ингредиентов."""
    keys = (
        "matched",
        "reference_total",
        "predicted_total",
        "match_percent",
        "recall_percent",
        "precision_percent",
        "f1_percent",
        "weight_mae_percent",
    )
    return {key: match[key] for key in keys}


def extract_components(
    prediction: Optional[Prediction],
) -> Tuple[List[Ingredient], List[str]]:
    """Извлекает components[] из предсказания, пропуская некорректные элементы."""
    if prediction is None or not isinstance(prediction.data, dict):
        return [], []

    raw_components = prediction.data.get("components")
    if not isinstance(raw_components, list):
        return [], ["components: нет массива компонентов"]

    components: List[Ingredient] = []
    warnings: List[str] = []
    for index, item in enumerate(raw_components, start=1):
        if not isinstance(item, dict):
            warnings.append(f"components[{index}]: элемент не объект")
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            warnings.append(f"components[{index}]: пустое имя")
            continue
        weight = parse_number(item.get("weight_g"))
        if weight is not None and weight < 0:
            warnings.append(f"components[{index}]: отрицательный weight_g")
            weight = None
        components.append(Ingredient(name=name, weight_g=weight))

    return components, warnings


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

        raw_ingredients = item.get("ingredients")
        ingredients: List[Ingredient] = []
        if isinstance(raw_ingredients, list):
            for ingredient in raw_ingredients:
                if not isinstance(ingredient, dict):
                    continue
                ingredient_name = str(ingredient.get("name") or "").strip()
                if not ingredient_name:
                    continue
                ingredient_weight = parse_number(ingredient.get("weight_g"))
                if ingredient_weight is not None and ingredient_weight < 0:
                    ingredient_weight = None
                ingredients.append(
                    Ingredient(
                        name=ingredient_name,
                        weight_g=ingredient_weight,
                    )
                )

        dishes.append(
            CatalogDish(
                dish_id=dish_id,
                key=key,
                name=str(item.get("name") or "").strip(),
                total=values,
                ingredients=ingredients,
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
