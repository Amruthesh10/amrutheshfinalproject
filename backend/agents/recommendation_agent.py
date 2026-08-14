from config.mistral_client import client


SYSTEM_PROMPT = """
You are an expert E-Commerce Product Recommendation Agent.

Your task is to recommend the best products based on:

1. User query
2. User preferences
3. Product specifications
4. Product ratings
5. Review analysis
6. Product popularity

Rules:

- Recommend only products matching the user's requirements.
- Prioritize user intent over raw ratings.
- Consider budget constraints.
- Consider use case.
- Explain reasoning clearly.
- Mention pros and cons.
- Suggest alternatives.
"""


def recommendation_agent(state):

    products = state["analyzed_products"][:5]

    response = client.chat.complete(
        model="mistral-small-latest",
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

Preferences:
{state['preferences']}

Top Ranked Products:
{products}

Recommend the best product and explain why.
"""
            }
        ]
    )

    state["recommendations"] = (
        response.choices[0].message.content
    )

    return state