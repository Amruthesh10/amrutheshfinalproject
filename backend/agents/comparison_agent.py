
import json
import re

from config.groq_client import client, MODEL_NAME


SYSTEM_PROMPT = """
You are an expert e-commerce product comparison analysis agent.

Analyze the validated products using the user's query.

Return ONLY valid JSON with this schema:

{
  "products": [],
  "key_differences": [],
  "advantages": [],
  "disadvantages": [],
  "best_value": null,
  "recommendation": "",
  "missing_information": []
}

Rules:
- Use only the supplied product data.
- Never invent prices, ratings, specifications, or features.
- Compare only the supplied validated products.
- Explain advantages and disadvantages only when supported by data.
- Consider the user's stated budget and use case.
- Mention missing information explicitly.
- Do not claim technical specifications are better unless data supports it.
- If the available data is insufficient to choose a winner,
  say so instead of guessing.
- Keep the recommendation practical and balanced.
- Return valid JSON only, without Markdown fences.
"""


def parse_json_response(content):
    """Parse JSON even if the model unexpectedly adds code fences."""
    content = (content or "").strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

    return json.loads(content)


def comparison_agent(state):
    # Use the products verified by the comparison validation agent.
    products = state.get("validated_comparison_products", [])

    if len(products) < 2:
        state["comparison_status"] = "insufficient_products"
        state["route"] = "fallback"
        return state

    query = state.get("query", "")
    preferences = state.get("preferences", {})

    product_data = json.dumps(
        products,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

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
{query}

User preferences:
{json.dumps(preferences, ensure_ascii=False, default=str)}

Validated products:
{product_data}

Analyze these products and return the required JSON object.
Include key differences, evidence-based advantages and
disadvantages, value for money, a practical recommendation,
and missing information.
""",
                },
            ],
            temperature=0.1,
            max_completion_tokens=1500,
        )

        content = response.choices[0].message.content or ""
        result = parse_json_response(content)

        if not isinstance(result, dict):
            raise ValueError("Comparison output must be a JSON object")

        # Keep the original validated records available downstream.
        result["products"] = products

        state["comparison_results"] = result
        state["comparison_status"] = "completed"
        state["route"] = "comparison_explanation"

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"Comparison output parsing error: {exc}")
        state["comparison_status"] = "analysis_failed"
        state["route"] = "fallback"

    except Exception as exc:
        print(f"Comparison agent error: {exc}")
        state["comparison_status"] = "analysis_failed"
        state["route"] = "fallback"

    return state
