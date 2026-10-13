import json
import re

from config.groq_client import client, MODEL_NAME
SYSTEM_PROMPT = """
You are a Product Validation Agent.

Return ONLY raw JSON.

Do NOT use markdown.
Do NOT use ```json.
Do NOT add explanations.

Example:

{
    "valid_products": [
        "product_name"
    ]
}
"""


def validation_agent(state):

    products = state["retrieved_products"]

    products_for_llm = []

    for p in products:

        products_for_llm.append(
            {
                "name": p["name"],
                "category": p["category"],
                "price": p["discount_price"],
                "rating": p["ratings"]
            }
        )

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
{state["query"]}

Retrieved Products:
{json.dumps(products_for_llm, indent=2)}

Return only valid products.
"""
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    print("\n=== VALIDATION RESPONSE ===")
    print(content)

    try:
        result = json.loads(content)

    except Exception:

        match = re.search(
            r"\{.*\}",
            content,
            re.DOTALL
        )

        if match:
            try:
                result = json.loads(match.group())
            except Exception:
                result = {"valid_products": []}
        else:
            result = {"valid_products": []}

    valid_names = set(
        result.get("valid_products", [])
    )

    validated_products = [
        p
        for p in products
        if p["name"] in valid_names
    ]

    state["validated_products"] = validated_products

    if len(validated_products) == 0:
        state["route"] = "fallback"
    else:
        state["route"] = "review"
        print("\n=== VALIDATION AGENT ===")
        print("Validated Products:", len(validated_products))
        print("Route:", state["route"])
    return state