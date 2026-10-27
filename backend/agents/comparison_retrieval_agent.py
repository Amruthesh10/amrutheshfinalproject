
import pandas as pd

from rag.retriever import retrieve_products


products_df = pd.read_csv("backend/data/products_final.csv")


def normalize(value):
    """Normalize text for case-insensitive matching."""
    return " ".join(str(value).lower().strip().split())


def get_product_name(document):
    """Extract the product name from a retrieved document."""
    first_line = document.split("\n")[0]
    return first_line.replace("Product Name:", "").strip()


def matches_category(product, category):
    """Check whether a product belongs to the requested category."""
    if not category:
        return True

    product_category = normalize(product.get("category", ""))
    requested_category = normalize(category)

    return (
        requested_category in product_category
        or product_category in requested_category
    )


def matches_budget(product, budget_max):
    """Check whether the product is within the maximum budget."""
    if budget_max is None:
        return True

    try:
        price = float(product.get("discount_price"))
        return price <= float(budget_max)
    except (TypeError, ValueError):
        return False


def find_exact_product(product_name, category=None, budget_max=None):
    """Find a product by its exact name in the CSV."""
    normalized_name = normalize(product_name)

    matches = products_df[
        products_df["name"].apply(normalize) == normalized_name
    ]

    if category:
        matches = matches[
            matches.apply(
                lambda product: matches_category(product, category),
                axis=1,
            )
        ]

    if budget_max is not None:
        matches = matches[
            matches.apply(
                lambda product: matches_budget(product, budget_max),
                axis=1,
            )
        ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def retrieve_one_product(product_name, category=None, budget_max=None):
    """Find a requested product using exact or semantic retrieval."""

    product = find_exact_product(
        product_name,
        category,
        budget_max,
    )

    if product is not None:
        return product

    results = retrieve_products(product_name, top_k=30)
    documents = results.get("documents", [[]])

    if not documents or not documents[0]:
        return None

    for document in documents[0]:
        retrieved_name = get_product_name(document)

        matches = products_df[
            products_df["name"].apply(normalize)
            == normalize(retrieved_name)
        ]

        if matches.empty:
            continue

        candidate = matches.iloc[0].to_dict()
        candidate_name = normalize(candidate.get("name", ""))
        requested_name = normalize(product_name)

        # Reject unrelated semantic-search results.
        if (
            requested_name not in candidate_name
            and candidate_name not in requested_name
        ):
            continue

        if not matches_category(candidate, category):
            continue

        if not matches_budget(candidate, budget_max):
            continue

        return candidate

    return None


def comparison_retrieval_agent(state):
    """Retrieve all requested products for comparison."""
    preferences = state.get("preferences", {})

    requested_products = preferences.get("comparison_products", [])
    requested_brands = preferences.get("comparison_brands", [])

    category = preferences.get("category")
    budget_max = preferences.get("budget_max")

    state["comparison_retrieved_products"] = []
    state["comparison_status"] = "pending"

    # Compare explicitly requested product names.
    if len(requested_products) >= 2:
        retrieved = []

        for product_name in requested_products:
            product = retrieve_one_product(
                product_name,
                category,
                budget_max,
            )

            if product is None:
                state["comparison_status"] = (
                    f"missing_product: {product_name}"
                )
                state["route"] = "fallback"
                return state

            retrieved.append(product)

        state["comparison_retrieved_products"] = retrieved
        state["comparison_status"] = "ready"
        state["route"] = "comparison_validation"
        return state

    # Brand-only comparisons require a separate brand-discovery agent.
    if len(requested_brands) >= 2:
        state["comparison_status"] = "brand_discovery_required"
        state["route"] = "comparison_brand_discovery"
        return state

    state["comparison_status"] = "insufficient_products"
    state["route"] = "fallback"

    return state
