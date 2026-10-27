import re
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CRITERIA = [
    "Price",
    "Ratings",
    "Battery",
    "Camera",
    "Processor / performance",
]


def _number(value: Any) -> Optional[float]:
    try:
        number = float(re.sub(r"[^\d.]", "", str(value)))
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "Unavailable"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def _battery_capacity(value: str) -> Optional[int]:
    match = re.search(r"\b(\d{3,5})\s*mAh\b", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_memory(name: str) -> Tuple[Optional[int], Optional[int]]:
    variant = name.split("|", maxsplit=1)[0]
    ram = re.search(r"\b(\d+)\s*GB\s*RAM\b", variant, re.IGNORECASE)
    storage = re.search(
        r"\b(\d+)\s*GB\s*(?:Storage|ROM)\b",
        variant,
        re.IGNORECASE,
    )
    values = re.findall(r"\b(\d+)\s*GB\b", variant, re.IGNORECASE)
    ram_value = int(ram.group(1)) if ram else None
    if ram_value is None and len(values) > 1:
        ram_value = int(values[0])
    return ram_value, int(storage.group(1)) if storage else None


def _extract_values(product: Dict[str, Any]) -> Dict[str, str]:
    name = str(product.get("name", ""))
    title = name.split("|", maxsplit=1)[0]
    battery_text = ""
    battery_capacity = re.search(
        r"\b\d{3,5}\s*mAh(?:\s+Battery)?\b",
        name,
        re.IGNORECASE,
    )
    battery_life = re.search(
        r"\b(?:up to|upto)\s+\d+\s*(?:hours?|hrs?|days?)\b",
        name,
        re.IGNORECASE,
    )
    if battery_capacity:
        battery_text = battery_capacity.group(0)
    if battery_life:
        battery_text = ", ".join(
            item for item in (battery_text, battery_life.group(0)) if item
        )

    camera = re.search(r"\b\d+(?:\.\d+)?\s*MP\b", name, re.IGNORECASE)

    processor_patterns = (
        r"(?:Qualcomm\s+)?Snapdragon\s+[\w.-]+(?:\s+[\w.-]+){0,2}",
        r"(?:MediaTek|Mediatek)\s+(?:Dimensity|Helio)\s+[\w.-]+",
        r"Apple\s+A\d+\s*(?:Pro|Bionic)?",
        r"Intel\s+Core\s+(?:Ultra\s+)?[iI]\d+(?:-[\w]+)?",
        r"AMD\s+Ryzen\s+\d+(?:\s+[\w-]+)?",
        r"\bExynos\s+[\w.-]+",
        r"\bTensor\s+[\w.-]+",
    )
    processor = next(
        (
            match.group(0).strip()
            for pattern in processor_patterns
            if (match := re.search(pattern, name, re.IGNORECASE))
        ),
        None,
    )
    ram, storage = _extract_memory(title)

    return {
        "price": _format_number(_number(product.get("discount_price"))),
        "actual_price": _format_number(_number(product.get("actual_price"))),
        "rating": _format_number(_number(product.get("ratings"))),
        "battery": battery_text or "Unavailable",
        "camera": camera.group(0) if camera else "Unavailable",
        "processor": processor or "Unavailable",
        "ram": f"{ram} GB" if ram is not None else "Unavailable",
        "storage": f"{storage} GB" if storage is not None else "Unavailable",
    }


def _criterion_key(criterion: str) -> str:
    value = criterion.lower()
    if "price" in value or "cost" in value:
        return "price"
    if "rating" in value:
        return "rating"
    if "battery" in value:
        return "battery"
    if "camera" in value:
        return "camera"
    if "processor" in value or "performance" in value:
        return "processor"
    if re.search(r"\bram\b", value):
        return "ram"
    if "storage" in value or re.search(r"\brom\b", value):
        return "storage"
    if "value" in value:
        return "value"
    return "unavailable"


def _requested_criteria(preferences: Dict[str, Any]) -> List[str]:
    criteria = preferences.get("comparison_criteria", [])
    if not isinstance(criteria, list):
        criteria = []
    cleaned = [str(criterion).strip() for criterion in criteria if str(criterion).strip()]
    return cleaned or DEFAULT_CRITERIA.copy()


def _assessment(
    criterion_key: str,
    products: List[Dict[str, Any]],
    values: List[Dict[str, str]],
) -> str:
    first_name = str(products[0].get("name", "Product 1"))
    second_name = str(products[1].get("name", "Product 2"))
    first_value = values[0].get(criterion_key, "Unavailable")
    second_value = values[1].get(criterion_key, "Unavailable")

    if criterion_key in {"price", "rating", "ram", "storage"}:
        numbers = [
            _number(products[0].get("discount_price"))
            if criterion_key == "price"
            else (
                _number(products[0].get("ratings"))
                if criterion_key == "rating"
                else _number(values[0][criterion_key])
            ),
            _number(products[1].get("discount_price"))
            if criterion_key == "price"
            else (
                _number(products[1].get("ratings"))
                if criterion_key == "rating"
                else _number(values[1][criterion_key])
            ),
        ]
        if None not in numbers and numbers[0] != numbers[1]:
            higher_is_better = criterion_key in {"rating", "ram", "storage"}
            winner_index = (
                0
                if (numbers[0] > numbers[1]) == higher_is_better
                else 1
            )
            winner = first_name if winner_index == 0 else second_name
            if criterion_key == "price":
                return f"{winner} has the lower catalog price."
            if criterion_key == "rating":
                return f"{winner} has the higher catalog rating."
            return f"{winner} has more {criterion_key.upper()} capacity."
        if first_value != "Unavailable" and first_value == second_value:
            return "The recorded values are the same."

    if criterion_key == "battery":
        if first_value != "Unavailable" and second_value != "Unavailable":
            capacities = [
                _battery_capacity(first_value),
                _battery_capacity(second_value),
            ]
            if None not in capacities and capacities[0] != capacities[1]:
                winner = first_name if capacities[0] > capacities[1] else second_name
                return (
                    f"{winner} has the greater listed battery capacity; "
                    "capacity alone does not establish real-world battery life."
                )
            return (
                "Battery specifications are listed; capacity alone does not "
                "establish real-world battery life."
            )
        return "Battery specifications are unavailable for at least one product."
    if criterion_key == "camera":
        if first_value != "Unavailable" and second_value != "Unavailable":
            resolutions = [_number(first_value), _number(second_value)]
            if None not in resolutions and resolutions[0] != resolutions[1]:
                higher_resolution_product = (
                    first_name
                    if resolutions[0] > resolutions[1]
                    else second_name
                )
                return (
                    f"{higher_resolution_product} lists the higher camera "
                    "resolution; resolution alone does not establish overall "
                    "camera quality."
                )
            return (
                "Camera information is listed, but the catalog does not provide "
                "enough evidence to rank overall camera quality."
            )
        return "Camera specifications are unavailable for at least one product."
    if criterion_key == "processor":
        if first_value != "Unavailable" and second_value != "Unavailable":
            return (
                "Processor names are listed, but no benchmark data is available "
                "to establish a performance winner."
            )
        return "Processor or performance data is unavailable for at least one product."
    if criterion_key == "value":
        return (
            "The catalog provides prices and ratings, but not enough evidence "
            "to determine overall value for money."
        )
    return "This specification is not available in the catalog."


def comparison_agent(state):
    """Build a comparison strictly from validated catalog records."""
    products = state.get("validated_comparison_products", [])

    if len(products) < 2:
        state["comparison_status"] = "insufficient_products"
        state["route"] = "fallback"
        return state

    preferences = state.get("preferences", {})
    criteria = _requested_criteria(preferences)
    product_values = [_extract_values(product) for product in products]
    comparisons = []

    for criterion in criteria:
        key = _criterion_key(criterion)
        comparisons.append(
            {
                "criterion": criterion,
                "values": [
                    values.get(key, "Unavailable")
                    for values in product_values
                ],
                "assessment": _assessment(key, products, product_values),
            }
        )

    missing_information = [
        comparison["criterion"]
        for comparison in comparisons
        if "unavailable" in comparison["assessment"].lower()
    ]
    differences = [
        f"{item['criterion']}: {item['assessment']}"
        for item in comparisons
    ]

    state["comparison_results"] = {
        "products": products,
        "criteria": criteria,
        "product_values": product_values,
        "comparisons": comparisons,
        "key_differences": differences,
        "advantages": [],
        "disadvantages": [],
        "best_value": None,
        "recommendation": (
            "No overall winner is declared; the catalog does not provide "
            "complete evidence across all requested criteria."
        ),
        "missing_information": missing_information,
    }
    state["comparison_status"] = "completed"
    state["route"] = "comparison_explanation"
    return state
