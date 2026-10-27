
import json

from config.groq_client import client, MODEL_NAME


SYSTEM_PROMPT = """
You are a helpful E-Commerce Shopping Assistant.

Generate a clear, user-friendly response based on the supplied data.

For product recommendations:
- Explain why the product suits the user's requirements.
- Highlight important strengths.
- Mention alternatives when available.

For product comparisons:
- Compare the requested products in a Markdown table.
- Include product names, prices, ratings, and available features.
- Explain the key differences.
- Discuss advantages, disadvantages, and value for money.
- Give a practical verdict based on the user's needs.
- If evidence is insufficient to choose a winner, say so.

General rules:
- Be conversational and concise but informative.
- Use only the supplied information.
- Never invent prices, ratings, specifications, or features.
- Clearly identify missing information.
- Do not claim a product is superior without supporting evidence.
- Return the final response in Markdown.
"""


def explanation_agent(state):

    preferences = state.get("preferences", {})
    intent = preferences.get("intent", "recommendation")

    # Select the appropriate source data.
    if intent == "comparison":

        comparison_results = state.get("comparison_results", {})

        if not comparison_results:
            state["response"] = (
                "I couldn't generate a reliable comparison "
                "with the available product information. "
                "Please try again or provide the exact product models."
            )
            return state

        task_data = json.dumps(
            comparison_results,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        task_instruction = """
Generate the final product comparison.

Include:
1. A comparison table.
2. Key differences.
3. Advantages and disadvantages.
4. Best value for money, when supported by the data.
5. A final verdict based on the user's requirements.

Clearly mention unavailable information.
"""

    else:

        recommendations = state.get("recommendations", "")

        if not recommendations:
            state["response"] = (
                "I couldn't generate a product recommendation "
                "with the available information. "
                "Please try another product query."
            )
            return state

        task_data = str(recommendations)

        task_instruction = """
Generate the final product recommendation.

Explain why the product is suitable, highlight its strengths,
and mention alternatives when available.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""
User query:
{state.get("query", "")}

User preferences:
{json.dumps(preferences, ensure_ascii=False, default=str)}

Task data:
{task_data}

Instructions:
{task_instruction}
""",
                },
            ],
            temperature=0.2,
            max_completion_tokens=1500,
        )

        state["response"] = (
            response.choices[0].message.content
            or "Sorry, I couldn't generate a response."
        )

    except Exception as exc:
        print(f"Explanation agent error: {exc}")

        state["response"] = (
            "Sorry, I couldn't generate the final response "
            "right now. Please try again."
        )

    return state
