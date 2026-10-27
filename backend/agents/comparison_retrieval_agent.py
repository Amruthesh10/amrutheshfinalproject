import difflib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PRODUCTS_PATH = Path(__file__).resolve().parents[1] / "data" / "products_final.csv"
products_df = pd.read_csv(PRODUCTS_PATH)


def normalize(value: Any) -> str:
    """Normalize punctuation, spacing, and case for catalog matching."""
    text = str(value).lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def canonical_model_name(value: Any) -> str:
    """Return a model key without color, memory, or network suffix details."""
    name = str(value).split("|", maxsplit=1)[0].split("(", maxsplit=1)[0]
    name = re.sub(
        r"\b\d+\s*gb(?:\s*(?:ram|storage|rom))?\b",
        " ",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"\b5g\b", " ", name, flags=re.IGNORECASE)
    return normalize(name)


def _is_same_model(requested_name: Any, catalog_name: Any) -> bool:
    requested_key = canonical_model_name(requested_name)
    catalog_key = canonical_model_name(catalog_name)

    if not requested_key or not catalog_key:
        return False
    if requested_key == catalog_key:
        return True

    requested_tokens = requested_key.split()
    catalog_tokens = catalog_key.split()
    if not requested_tokens or len(requested_tokens) != len(catalog_tokens):
        return False

    for requested_token, catalog_token in zip(requested_tokens, catalog_tokens):
        if requested_token == catalog_token:
            continue
        if (
            any(character.isdigit() for character in requested_token + catalog_token)
            or difflib.SequenceMatcher(
                None,
                requested_token,
                catalog_token,
            ).ratio()
            < 0.8
        ):
            return False
    return True


def is_requested_product_match(requested_name: Any, product: Dict[str, Any]) -> bool:
    """Verify model identity and any explicitly requested memory variant."""
    actual_name = product.get("name", "")
    if not _is_same_model(requested_name, actual_name):
        return False

    requested_ram = _requested_ram(requested_name)
    requested_storage = _requested_storage(requested_name)
    actual_ram, actual_storage = _variant_memory(actual_name)

    return (
        (requested_ram is None or requested_ram == actual_ram)
        and (requested_storage is None or requested_storage == actual_storage)
    )


def _requested_ram(name: Any) -> Optional[int]:
    text = str(name)
    match = re.search(r"\b(\d+)\s*gb\s*ram\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    memory_values = re.findall(r"\b(\d+)\s*gb\b", text, re.IGNORECASE)
    if len(memory_values) == 1 and not _requested_storage(text):
        return int(memory_values[0])
    return None


def _requested_storage(name: Any) -> Optional[int]:
    match = re.search(
        r"\b(\d+)\s*gb\s*(?:storage|rom)\b",
        str(name),
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def _variant_memory(name: Any) -> Tuple[Optional[int], Optional[int]]:
    text = str(name).split("|", maxsplit=1)[0]
    ram = re.search(r"\b(\d+)\s*gb\s*ram\b", text, re.IGNORECASE)
    storage = re.search(
        r"\b(\d+)\s*gb\s*(?:storage|rom)\b",
        text,
        re.IGNORECASE,
    )
    if ram:
        ram_value = int(ram.group(1))
    else:
        values = re.findall(r"\b(\d+)\s*gb\b", text, re.IGNORECASE)
        ram_value = int(values[0]) if len(values) > 1 else None
    return (
        ram_value,
        int(storage.group(1)) if storage else None,
    )


def _requested_color(name: Any) -> Optional[str]:
    match = re.search(r"\(([^)]*)\)", str(name))
    if not match:
        return None
    color_text = re.sub(
        r"\b\d+\s*gb(?:\s*(?:ram|storage|rom))?\b",
        " ",
        match.group(1),
        flags=re.IGNORECASE,
    )
    color_text = re.sub(r"\b(?:ram|storage|rom)\b", " ", color_text, flags=re.IGNORECASE)
    color = normalize(color_text)
    return color or None


def _variant_key(product: Dict[str, Any]) -> Tuple[Any, ...]:
    name = str(product.get("name", ""))
    ram, storage = _variant_memory(name)
    return (
        ram,
        storage,
        normalize(product.get("discount_price", "")),
        _extract_specification_key(name),
    )


def _extract_specification_key(name: str) -> Tuple[str, ...]:
    return tuple(
        match.lower()
        for match in re.findall(
            r"\b\d{3,5}\s*mAh\b|\b\d+(?:\.\d+)?\s*MP\b|"
            r"\b(?:Snapdragon|Dimensity|Exynos|Tensor|Ryzen)\s*[\w-]+",
            name,
            flags=re.IGNORECASE,
        )
    )


class AmbiguousProductError(ValueError):
    """Raised when the requested model identifies materially different variants."""

    def __init__(self, candidates: List[Dict[str, Any]]):
        self.candidates = candidates
        super().__init__("The requested product has multiple catalog variants.")


def find_product_candidates(product_name: str) -> List[Dict[str, Any]]:
    """Resolve a model name to likely catalog records, without semantic-only matches."""
    requested = normalize(product_name)
    records = products_df.to_dict("records")

    # Prefer a complete catalog name before matching the broader model family.
    exact_matches = [
        product
        for product in records
        if normalize(product.get("name", "")) == requested
    ]
    if exact_matches:
        return exact_matches

    model_matches = [
        product
        for product in records
        if _is_same_model(product_name, product.get("name", ""))
    ]
    return model_matches


def retrieve_one_product(product_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a requested model and disambiguate materially different variants."""
    candidates = find_product_candidates(product_name)
    if not candidates:
        return None

    requested_ram = _requested_ram(product_name)
    requested_storage = _requested_storage(product_name)
    requested_color = _requested_color(product_name)

    if requested_ram is not None:
        candidates = [
            product
            for product in candidates
            if _variant_memory(product.get("name", ""))[0] == requested_ram
        ]
    if requested_storage is not None:
        candidates = [
            product
            for product in candidates
            if _variant_memory(product.get("name", ""))[1] == requested_storage
        ]
    if requested_color:
        color_matches = [
            product
            for product in candidates
            if requested_color in normalize(product.get("name", ""))
        ]
        if color_matches:
            candidates = color_matches

    if not candidates:
        return None

    variant_keys = {_variant_key(product) for product in candidates}
    if len(variant_keys) > 1:
        raise AmbiguousProductError(candidates)

    return candidates[0]


def comparison_retrieval_agent(state):
    """Retrieve each requested product independently from the product catalog."""
    preferences = state.get("preferences", {})
    requested_products = preferences.get("comparison_products", [])
    requested_brands = preferences.get("comparison_brands", [])

    state["comparison_retrieved_products"] = []
    state["comparison_missing_products"] = []
    state["comparison_status"] = "pending"

    if len(requested_products) >= 2:
        retrieved = []
        missing = []

        for product_name in requested_products:
            try:
                product = retrieve_one_product(product_name)
            except AmbiguousProductError as exc:
                state["comparison_status"] = (
                    f"ambiguous_product: {product_name}; candidates: "
                    + "; ".join(
                        candidate.get("name", "")
                        for candidate in exc.candidates[:4]
                    )
                )
                state["route"] = "fallback"
                state["comparison_retrieved_products"] = retrieved
                state["comparison_missing_products"] = missing
                return state

            if product is None:
                missing.append(product_name)
            else:
                retrieved.append(product)

        state["comparison_retrieved_products"] = retrieved
        if missing:
            state["comparison_missing_products"] = missing
            state["comparison_status"] = "missing_products"
            state["route"] = "fallback"
            return state

        state["comparison_status"] = "ready"
        state["route"] = "comparison_validation"
        return state

    if len(requested_brands) >= 2:
        state["comparison_status"] = "brand_discovery_required"
        state["route"] = "comparison_brand_discovery"
        return state

    state["comparison_status"] = "insufficient_products"
    state["route"] = "fallback"
    return state
