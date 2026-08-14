from config.mistral_client import client


SYSTEM_PROMPT = """
You are a helpful E-Commerce Shopping Assistant.

Your task is to generate a final user-friendly response.

Rules:

1. Be conversational.
2. Explain why the product is recommended.
3. Mention important strengths.
4. Mention alternatives if available.
5. Do not sound robotic.
6. Keep the response concise but informative.
"""


def explanation_agent(state):

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

Recommendation:
{state['recommendations']}

Generate the final response for the customer.
"""
            }
        ]
    )

    state["response"] = (
        response.choices[0].message.content
    )

    return state