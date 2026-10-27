
import json
import re

from config.groq_client import client, MODEL_NAME


SYSTEM_PROMPT = """
You are the Preference Extraction Agent for an Agentic AI
E-Commerce Recommendation System.

Analyze the user's query and return ONLY valid JSON.
Do not use Markdown or code fences.

Use this exact schema:

{
  "intent": "recommendation",
  "comparison_products": [],
  "comparison_brands": [],
  "comparison_criteria": [],
  "category": null,
  "budget_min": null,
  "budget_max": null,
  "brand": null,
  "use_case": null,
  "specifications": [],
  "sort_preference": null,
  "product_type": null,
  "recommendation_query": true
}

AVAILABLE CATEGORIES:
Phone, Laptop, Earbuds, Smartwatch, TV, Speaker,
Mouse, Keyboard, Storage, Power Bank.

RULES:

1. intent:
   - "comparison": The user explicitly asks to compare products,
     asks for differences, or asks which of named products is better.
   - "recommendation": The user wants product suggestions,
     searches for products, or asks for the best product.
   - "general": The query is unrelated to shopping.

2. comparison_products:
   - For comparisons, list every explicitly named product/model.
   - Preserve full model names.
   - Example: ["Samsung Galaxy S25", "Redmi Note 14 Pro"].
   - Use [] if exact product models are not specified.
   - Do not invent product names.

3. comparison_brands:
   - Extract explicitly mentioned brands in a comparison.
   - Example: ["Samsung", "Redmi"].
   - Use [] if none are identified.
   - Do not confuse a product model with a brand.

4. comparison_criteria:
   - Extract criteria explicitly requested, such as price,
     performance, camera, battery, ratings, or value for money.
   - If the user gives no criteria, use [].

5. category:
   - Infer a category when possible using the available categories.
   - Otherwise use null.

6. budget_min and budget_max:
   - Extract minimum and maximum budget when specified.
   - Use numeric values without currency symbols.
   - Use null when not specified.

7. brand:
   - For ordinary recommendations, extract the requested brand.
   - For comparisons between multiple brands, use null.

8. use_case:
   - Extract gaming, coding, photography, office work,
     studying, travel, or another stated purpose.
   - Otherwise use null.

9. specifications:
   - Extract technical requirements such as 8GB RAM,
     256GB storage, AMOLED, SSD, or 5000mAh battery.
   - For named comparison models, do NOT put the model names here;
     put them in comparison_products.

10. sort_preference:
    - "lowest_price", "highest_rating", "best",
      "most_popular", or null.

11. product_type:
    - Examples: smartphone, gaming laptop, wireless earbuds.
    - Otherwise null.

12. recommendation_query:
    - true for recommendation and comparison shopping requests.
    - false for unrelated general questions.

13. A query mentioning multiple brands is NOT automatically
    a comparison. The user must express comparison intent.

14. Recognize comparison expressions such as:
    "compare", "difference between", "versus", "vs",
    "which is better", and equivalent wording.

15. Do not invent products, specifications, budgets, or brands.

EXAMPLES:

User: Compare Samsung Galaxy S25 and Redmi Note 14 Pro

Output:
{
  "intent": "comparison",
  "comparison_products": [
    "Samsung Galaxy S25",
    "Redmi Note 14 Pro"
  ],
  "comparison_brands": ["Samsung", "Redmi"],
  "comparison_criteria": [],
  "category": "Phone",
  "budget_min": null,
  "budget_max": null,
  "brand": null,
  "use_case": null,
  "specifications": [],
  "sort_preference": null,
  "product_type": "smartphone",
  "recommendation_query": true
}

User: Compare Redmi and Samsung phones under 20000

Output:
{
  "intent": "comparison",
  "comparison_products": [],
  "comparison_brands": ["Redmi", "Samsung"],
  "comparison_criteria": [],
  "category": "Phone",
  "budget_min": null,
  "budget_max": 20000,
  "brand": null,
  "use_case": null,
  "specifications": [],
  "sort_preference": null,
  "product_type": "smartphone",
  "recommendation_query": true
}

User: Suggest a Samsung phone under 25000 with 8GB RAM

Output:
{
  "intent": "recommendation",
  "comparison_products": [],
  "comparison_brands": [],
  "comparison_criteria": [],
  "category": "Phone",
  "budget_min": null,
  "budget_max": 25000,
  "brand": "Samsung",
  "use_case": null,
  "specifications": ["8GB RAM"],
  "sort_preference": null,
  "product_type": "smartphone",
  "recommendation_query": true
}

User: What is Agentic AI?

Output:
{
  "intent": "general",
  "comparison_products": [],
  "comparison_brands": [],
  "comparison_criteria": [],
  "category": null,
  "budget_min": null,
  "budget_max": null,
  "brand": null,
  "use_case": null,
  "specifications": [],
  "sort_preference": null,
  "product_type": null,
  "recommendation_query": false
}
"""


DEFAULT_PREFERENCES = {
    "intent": "general",
    "comparison_products": [],
    "comparison_brands": [],
    "comparison_criteria": [],
    "category": None,
    "budget_min": None,
    "budget_max": None,
    "brand": None,
    "use_case": None,
    "specifications": [],
    "sort_preference": None,
    "product_type": None,
    "recommendation_query": False,
}


def _extract_json(content):
    """Parse JSON, including responses wrapped in Markdown fences."""
    content = (content or "").strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

    return json.loads(content)


def preference_agent(state):
    query = state.get("query", "").strip()

    if not query:
        state["preferences"] = DEFAULT_PREFERENCES.copy()
        return state

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_completion_tokens=800,
        )

        content = response.choices[0].message.content or ""
        extracted = _extract_json(content)

        if not isinstance(extracted, dict):
            raise ValueError("Preference response must be a JSON object")

        preferences = DEFAULT_PREFERENCES.copy()
        preferences.update(extracted)

        # Validate the intent.
        if preferences["intent"] not in {
            "comparison",
            "recommendation",
            "general",
        }:
            preferences["intent"] = "general"

        # Normalize list fields.
        for key in (
            "comparison_products",
            "comparison_brands",
            "comparison_criteria",
            "specifications",
        ):
            if not isinstance(preferences[key], list):
                preferences[key] = []

        # Keep the shopping flag consistent with the intent.
        preferences["recommendation_query"] = (
            preferences["intent"] in {"comparison", "recommendation"}
        )

        state["preferences"] = preferences

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Preference extraction parsing error: {exc}")
        state["preferences"] = DEFAULT_PREFERENCES.copy()

    except Exception as exc:
        print(f"Preference agent error: {exc}")
        state["preferences"] = DEFAULT_PREFERENCES.copy()

    return state
