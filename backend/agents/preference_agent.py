import json
from config.groq_client import client, MODEL_NAME

SYSTEM_PROMPT = """
You are the Preference Extraction Agent of an intelligent E-Commerce Recommendation System.

Analyze the user's request and extract shopping preferences.

Return ONLY valid JSON.

Schema:

{
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

Available Categories:
- Phone
- Laptop
- Earbuds
- Smartwatch
- TV
- Speaker
- Mouse
- Keyboard
- Storage
- Power Bank

Rules:

1. Return JSON only.
2. No markdown.
3. Missing fields must be null.
4. Infer category when possible.
5. Extract all technical specifications.
6. Extract brand if mentioned.
7. Extract budget constraints.
8. Detect the intended use case.
9. Detect ranking preference if mentioned.
10. If not a shopping query, set recommendation_query=false.

Examples:

User:
Suggest a gaming phone under 20000

Output:
{
  "category":"Phone",
  "budget_min":null,
  "budget_max":20000,
  "brand":null,
  "use_case":"gaming",
  "specifications":[],
  "sort_preference":null,
  "product_type":"smartphone",
  "recommendation_query":true
}

User:
Need Samsung phone under 25000 with 8GB RAM

Output:
{
  "category":"Phone",
  "budget_min":null,
  "budget_max":25000,
  "brand":"Samsung",
  "use_case":null,
  "specifications":["8GB RAM"],
  "sort_preference":null,
  "product_type":"smartphone",
  "recommendation_query":true
}

User:
Best laptop for coding with SSD and 16GB RAM around 60000

Output:
{
  "category":"Laptop",
  "budget_min":null,
  "budget_max":60000,
  "brand":null,
  "use_case":"coding",
  "specifications":["SSD","16GB RAM"],
  "sort_preference":"best",
  "product_type":"laptop",
  "recommendation_query":true
}

User:
Show me the cheapest earbuds

Output:
{
  "category":"Earbuds",
  "budget_min":null,
  "budget_max":null,
  "brand":null,
  "use_case":null,
  "specifications":[],
  "sort_preference":"lowest_price",
  "product_type":"earbuds",
  "recommendation_query":true
}

User:
What is Agentic AI?

Output:
{
  "category":null,
  "budget_min":null,
  "budget_max":null,
  "brand":null,
  "use_case":null,
  "specifications":[],
  "sort_preference":null,
  "product_type":null,
  "recommendation_query":false
}
"""

def preference_agent(state):
    query = state["query"]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_completion_tokens=600,
    )

    content = response.choices[0].message.content or ""

    try:
        # Handle JSON wrapped in markdown fences if returned
        content = content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0].strip()

        preferences = json.loads(content)

        # Ensure required keys exist
        defaults = {
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

        defaults.update(preferences)
        preferences = defaults

    except (json.JSONDecodeError, TypeError, ValueError):
        preferences = {
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

    state["preferences"] = preferences
    return state


    query = state["query"]

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    try:
        preferences = json.loads(content)

    except Exception:
        preferences = {
            "category": None,
            "budget_min": None,
            "budget_max": None,
            "brand": None,
            "use_case": None,
            "specifications": [],
            "sort_preference": None,
            "product_type": None,
            "recommendation_query": False
        }

    state["preferences"] = preferences

    return state