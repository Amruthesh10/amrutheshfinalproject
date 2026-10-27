
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


def _markdown_value(value):
    if value is None or str(value).strip().lower() in {"", "nan", "none"}:
        return "Unavailable"
    return str(value).replace("|", r"\|").replace("\n", " ")


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

        products = comparison_results.get(
            "products",
            state.get("validated_comparison_products", []),
        )
        comparisons = comparison_results.get("comparisons", [])
        if len(products) < 2:
            state["response"] = (
                "I couldn't generate a reliable comparison because both "
                "verified product records are not available."
            )
            return state

        product_names = [
            _markdown_value(product.get("name", f"Product {index + 1}"))
            for index, product in enumerate(products[:2])
        ]
        lines = [
            "### Product comparison",
            "",
            f"Products: **{product_names[0]}** and **{product_names[1]}**",
            "",
            "| Criterion | " + " | ".join(product_names) + " | Assessment |",
            "| --- | " + " | ".join("---" for _ in product_names) + " | --- |",
        ]

        for comparison in comparisons:
            values = comparison.get("values", [])
            left = _markdown_value(values[0] if values else None)
            right = _markdown_value(values[1] if len(values) > 1 else None)
            assessment = _markdown_value(comparison.get("assessment"))
            criterion = _markdown_value(comparison.get("criterion"))
            lines.append(
                f"| {criterion} | {left} | {right} | {assessment} |"
            )

        lines.extend(["", "**Catalog details**", ""])
        lines.extend(
            [
                "| Attribute | " + " | ".join(product_names) + " |",
                "| --- | " + " | ".join("---" for _ in product_names) + " |",
            ]
        )
        for label, field in (
            ("Catalog price", "discount_price"),
            ("List price", "actual_price"),
            ("Rating", "ratings"),
            ("Number of ratings", "no_of_ratings"),
            ("Category", "category"),
        ):
            row_values = [
                _markdown_value(product.get(field))
                for product in products[:2]
            ]
            lines.append(f"| {label} | {row_values[0]} | {row_values[1]} |")

        recommendation = comparison_results.get("recommendation")
        if recommendation:
            lines.extend(
                ["", f"**Verdict:** {_markdown_value(recommendation)}"]
            )
        missing_information = comparison_results.get("missing_information", [])
        if missing_information:
            missing = ", ".join(
                _markdown_value(item) for item in missing_information
            )
            lines.extend(
                [
                    "",
                    f"**Unavailable in the catalog:** {missing}.",
                ]
            )

        state["response"] = "\n".join(lines)
        return state

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
