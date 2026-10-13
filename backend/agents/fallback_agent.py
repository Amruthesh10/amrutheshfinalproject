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
{state.get('preferences', {})}

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
