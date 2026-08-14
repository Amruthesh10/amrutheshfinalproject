import pandas as pd

from rag.retriever import retrieve_products


products_df = pd.read_csv(
    "backend/data/products_final.csv"
)


def retrieval_agent(state):

    preferences = state["preferences"]

    # Non-shopping query
    if not preferences.get("recommendation_query", True):
        state["retrieved_products"] = []
        state["route"] = "fallback"
        return state

    category = preferences.get("category")
    budget_max = preferences.get("budget_max")
    brand = preferences.get("brand")

    # Semantic search from ChromaDB
    results = retrieve_products(
        state["query"],
        top_k=30
    )

    documents = results["documents"][0]

    filtered_products = []

    for doc in documents:

        try:
            product_name = (
                doc.split("\n")[0]
                .replace("Product Name:", "")
                .strip()
            )

            match = products_df[
                products_df["name"] == product_name
            ]

            if match.empty:
                continue

            product = match.iloc[0].to_dict()

            # Category filter
            if category:
                if product["category"] != category:
                    continue

            # Budget filter
            if budget_max:
                if product["discount_price"] > budget_max:
                    continue

            # Brand filter
            if brand:
                if brand.lower() not in product["name"].lower():
                    continue

            filtered_products.append(product)

        except Exception:
            continue

    state["retrieved_products"] = filtered_products

    if len(filtered_products) == 0:
        state["route"] = "fallback"
    else:
        state["route"] = "review"

    return state