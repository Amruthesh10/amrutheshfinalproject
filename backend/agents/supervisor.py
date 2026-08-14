def supervisor_agent(state):

    preferences = state["preferences"]

    if preferences.get("recommendation_query", False):
        state["route"] = "retrieval"
    else:
        state["route"] = "fallback"

    print("\n=== SUPERVISOR AGENT ===")
    print("Route:", state["route"])

    return state