from config.mistral_client import client


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

Extracted Preferences:
{state.get('preferences', {})}

No matching products were found.
"""
            }
        ]
    )

    state["response"] = (
        response.choices[0].message.content
    )

    return state