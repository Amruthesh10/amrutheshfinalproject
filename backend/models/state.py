
from typing import TypedDict, Dict, Any, List


class AgentState(TypedDict, total=False):

    # User input
    query: str

    # Extracted preferences and intent
    preferences: Dict[str, Any]

    # Recommendation pipeline
    retrieved_products: List[Dict[str, Any]]
    validated_products: List[Dict[str, Any]]
    analyzed_products: List[Dict[str, Any]]
    recommendations: str

    # Comparison pipeline
    comparison_products: List[str]
    comparison_brands: List[str]
    comparison_results: Dict[str, Any]
    comparison_status: str
    comparison_missing_products: List[str]
    comparison_retrieved_products: List[Dict[str, Any]]
    validated_comparison_products: List[Dict[str, Any]]

    # Final response and routing
    response: str
    route: str
