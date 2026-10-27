from config.groq_client import client, MODEL_NAME

SYSTEM_PROMPT = """
You are a helpful E-Commerce Assistant.

The requested products could not be found in the catalog.

Your task:

1. Explain politely why no suitable products were found.
2. Suggest possible alternatives.
3. Suggest increasing budget if needed.
4. Suggest related categories if appropriate.
5. Never invent products.
6. Be concise and helpful.
"""

def fallback_agent(state):
    preferences = state.get("preferences", {})
    if preferences.get("intent") == "comparison":
        missing_products = state.get("comparison_missing_products", [])
        status = state.get("comparison_status", "")

        if missing_products:
            missing = ", ".join(f"**{name}**" for name in missing_products)
            state["response"] = (
                f"I couldn't find {missing} in the product catalog, so I "
                "couldn't compare the requested products. Please check the "
                "product name or choose another model from the catalog."
            )
            return state

        if status.startswith("ambiguous_product:"):
            details = status.partition(":")[2].strip()
            product_name, _, candidates = details.partition("; candidates:")
            state["response"] = (
                f"I found multiple catalog variants for **{product_name.strip()}**"
            )
            if candidates.strip():
                state["response"] += f": {candidates.strip()}."
            else:
                state["response"] += "."
            state["response"] += (
                " Please specify the RAM/storage variant to compare the right "
                "product."
            )
            return state

        if status == "insufficient_products":
            state["response"] = (
                "Please name two products from the catalog so I can compare "
                "them."
            )
            return state

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
User Query:
{state['query']}

Extracted Preferences:
{preferences}

No matching products were found.
"""
            }
        ],
        temperature=0.3,
        max_completion_tokens=600,
    )

    state["response"] = (
        response.choices[0].message.content or
        "No matching products were found. Try changing your budget or product preferences."
    )

    return state
