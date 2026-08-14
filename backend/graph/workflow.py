from langgraph.graph import StateGraph, END

from models.state import AgentState

from agents.preference_agent import preference_agent
from agents.supervisor import supervisor_agent
from agents.retrieval_agent import retrieval_agent
from agents.validation_agent import validation_agent
from agents.review_agent import review_agent
from agents.recommendation_agent import recommendation_agent
from agents.explanation_agent import explanation_agent
from agents.fallback_agent import fallback_agent


workflow = StateGraph(AgentState)

# =========================
# Nodes
# =========================

workflow.add_node(
    "preference",
    preference_agent
)

workflow.add_node(
    "supervisor",
    supervisor_agent
)

workflow.add_node(
    "retrieval",
    retrieval_agent
)

workflow.add_node(
    "validation",
    validation_agent
)

workflow.add_node(
    "review",
    review_agent
)

workflow.add_node(
    "recommendation",
    recommendation_agent
)

workflow.add_node(
    "explanation",
    explanation_agent
)

workflow.add_node(
    "fallback",
    fallback_agent
)

# =========================
# Entry Point
# =========================

workflow.set_entry_point(
    "preference"
)

# =========================
# Basic Edges
# =========================

workflow.add_edge(
    "preference",
    "supervisor"
)

workflow.add_edge(
    "retrieval",
    "validation"
)

workflow.add_edge(
    "review",
    "recommendation"
)

workflow.add_edge(
    "recommendation",
    "explanation"
)

workflow.add_edge(
    "explanation",
    END
)

workflow.add_edge(
    "fallback",
    END
)

# =========================
# Supervisor Routing
# =========================

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["route"],
    {
        "retrieval": "retrieval",
        "fallback": "fallback"
    }
)

# =========================
# Validation Routing
# =========================

workflow.add_conditional_edges(
    "validation",
    lambda state: state["route"],
    {
        "review": "review",
        "fallback": "fallback"
    }
)

# =========================
# Compile Graph
# =========================

app = workflow.compile()