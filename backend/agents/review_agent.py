import pandas as pd


reviews_df = pd.read_csv(
    "backend/data/reviews.csv"
)


def review_agent(state):

    query = state["query"].lower()

    preferences = state["preferences"]

    products = state["validated_products"]

    analyzed_products = []

    for product in products:

        product_reviews = reviews_df[
            reviews_df["product_id"] ==
            product["product_id"]
        ]

        total_reviews = len(product_reviews)

        positive_reviews = (
            product_reviews["rating"] >= 4
        ).sum()

        negative_reviews = (
            product_reviews["rating"] <= 2
        ).sum()

        avg_review_rating = (
            product_reviews["rating"].mean()
        )

        if pd.isna(avg_review_rating):
            avg_review_rating = 0

        product_name = product["name"].lower()

        bonus = 0

        # -----------------------------
        # RAM Bonus
        # -----------------------------

        if "12gb" in product_name:
            bonus += 8

        elif "8gb" in product_name:
            bonus += 6

        elif "6gb" in product_name:
            bonus += 4

        elif "4gb" in product_name:
            bonus += 2

        # -----------------------------
        # Storage Bonus
        # -----------------------------

        if "256gb" in product_name:
            bonus += 6

        elif "128gb" in product_name:
            bonus += 4

        elif "64gb" in product_name:
            bonus += 2

        # -----------------------------
        # Battery Bonus
        # -----------------------------

        if "6000mah" in product_name:
            bonus += 6

        elif "5000mah" in product_name:
            bonus += 4

        # -----------------------------
        # Gaming Use Case
        # -----------------------------

        if preferences.get("use_case") == "gaming":

            if "8gb" in product_name:
                bonus += 10

            if "12gb" in product_name:
                bonus += 15

            if "120hz" in product_name:
                bonus += 10

            if "6000mah" in product_name:
                bonus += 5

        # -----------------------------
        # Camera Use Case
        # -----------------------------

        if preferences.get("use_case") == "camera":

            if "108mp" in product_name:
                bonus += 15

            elif "64mp" in product_name:
                bonus += 10

            elif "50mp" in product_name:
                bonus += 6

        # -----------------------------
        # Battery Use Case
        # -----------------------------

        if preferences.get("use_case") == "battery":

            if "6000mah" in product_name:
                bonus += 15

            elif "5000mah" in product_name:
                bonus += 10

        # -----------------------------
        # Final Score
        # -----------------------------

        final_score = (

            product["ratings"] * 10 +

            avg_review_rating * 5 +

            positive_reviews * 0.5 -

            negative_reviews * 0.5 +

            (product["no_of_ratings"] / 1000) +

            bonus

        )

        analyzed_products.append(

            {

                **product,

                "positive_reviews":
                int(positive_reviews),

                "negative_reviews":
                int(negative_reviews),

                "avg_review_rating":
                round(avg_review_rating, 2),

                "total_reviews":
                total_reviews,

                "bonus":
                bonus,

                "final_score":
                round(final_score, 2)

            }

        )

    analyzed_products = sorted(

        analyzed_products,

        key=lambda x: x["final_score"],

        reverse=True

    )

    state["analyzed_products"] = analyzed_products

    return state