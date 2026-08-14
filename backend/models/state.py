from typing import TypedDict, Dict, Any, List


class AgentState(TypedDict):

    query: str

    preferences: Dict[str, Any]

    retrieved_products: List[Dict]

    validated_products: List[Dict]

    analyzed_products: List[Dict]

    recommendations: str

    response: str

    route: str