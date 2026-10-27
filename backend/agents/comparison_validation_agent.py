
from typing import Any, Dict, List

from agents.comparison_retrieval_agent import is_requested_product_match


def comparison_validation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    preferences = state.get("preferences", {})

    requested_products = preferences.get("comparison_products", [])
    retrieved_products = state.get("comparison_retrieved_products", [])

    state["validated_comparison_products"] = []

    # We need at least two explicitly requested product models.
    if len(requested_products) < 2:
        state["comparison_status"] = "insufficient_products"
        state["route"] = "fallback"
        return state

    if len(retrieved_products) < len(requested_products):
        state["comparison_status"] = "missing_products"
        state["route"] = "fallback"
        return state

    # Ensure every requested model has a corresponding product record.
    remaining_products = list(retrieved_products)
    validated_products: List[Dict[str, Any]] = []

    for requested_name in requested_products:
        matched_index = None

        for index, product in enumerate(remaining_products):
            if is_requested_product_match(requested_name, product):
                matched_index = index
                break

        if matched_index is None:
            state["comparison_status"] = (
                f"product_match_not_verified: {requested_name}"
            )
            state["route"] = "fallback"
            return state

        validated_products.append(
            remaining_products.pop(matched_index)
        )

    state["validated_comparison_products"] = validated_products
    state["comparison_status"] = "validated"
    state["route"] = "comparison"

    return state
