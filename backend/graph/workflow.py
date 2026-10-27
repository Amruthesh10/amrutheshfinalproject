
from langgraph.graph import StateGraph, END

from models.state import AgentState

from agents.preference_agent import preference_agent
from agents.supervisor import supervisor_agent
from agents.retrieval_agent import retrieval_agent
from agents.validation_agent import validation_agent
from agents.review_agent import review_agent
from agents.recommendation_agent import recommendation_agent
from agents.comparison_agent import comparison_agent
from agents.comparison_retrieval_agent import (
    comparison_retrieval_agent,
)
from agents.comparison_validation_agent import (
    comparison_validation_agent,
)
from agents.explanation_agent import explanation_agent
from agents.fallback_agent import fallback_agent


workflow = StateGraph(AgentState)


# =========================
# Nodes
# =========================

workflow.add_node("preference", preference_agent)

# Existing recommendation pipeline
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("retrieval", retrieval_agent)
workflow.add_node("validation", validation_agent)
workflow.add_node("review", review_agent)
workflow.add_node("recommendation", recommendation_agent)

# New comparison pipeline
workflow.add_node(
    "comparison_retrieval",
    comparison_retrieval_agent,
)
workflow.add_node(
    "comparison_validation",
    comparison_validation_agent,
)
workflow.add_node("comparison", comparison_agent)

# Shared final response and fallback
workflow.add_node("explanation", explanation_agent)
workflow.add_node("fallback", fallback_agent)


# =========================
# Entry Point
# =========================

workflow.set_entry_point("preference")


# =========================
# Intent Routing
# =========================

def route_after_preference(state):
    preferences = state.get("preferences", {})
    intent = preferences.get("intent", "general")

    if intent == "comparison":
        return "comparison"

    if intent == "recommendation":
        return "recommendation"

    return "fallback"


workflow.add_conditional_edges(
    "preference",
    route_after_preference,
    {
        "comparison": "comparison_retrieval",
        "recommendation": "supervisor",
        "fallback": "fallback",
    },
)


# =========================
# Existing Recommendation Flow
# =========================

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.get("route", "fallback"),
    {
        "retrieval": "retrieval",
        "fallback": "fallback",
    },
)

workflow.add_edge("retrieval", "validation")

workflow.add_conditional_edges(
    "validation",
    lambda state: state.get("route", "fallback"),
    {
        "review": "review",
        "fallback": "fallback",
    },
)

# Comparison requests have already branched off.
workflow.add_edge("review", "recommendation")

workflow.add_edge("recommendation", "explanation")


# =========================
# New Comparison Flow
# =========================

workflow.add_conditional_edges(
    "comparison_retrieval",
    lambda state: state.get("route", "fallback"),
    {
        "comparison_validation": "comparison_validation",
        "fallback": "fallback",

        # Brand-only discovery will be implemented separately.
        "comparison_brand_discovery": "fallback",
    },
)

workflow.add_conditional_edges(
    "comparison_validation",
    lambda state: state.get("route", "fallback"),
    {
        "comparison": "comparison",
        "fallback": "fallback",
    },
)

workflow.add_edge("comparison", "explanation")


# =========================
# Terminal Edges
# =========================

workflow.add_edge("explanation", END)
workflow.add_edge("fallback", END)


# =========================
# Compile Graph
# =========================

app = workflow.compile()
