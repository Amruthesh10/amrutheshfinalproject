"""Review-related service layer."""

from __future__ import annotations

from typing import Any, Dict, List


class ReviewService:
    def __init__(self, reviews: List[Dict[str, Any]] | None = None):
        self.reviews = reviews or []

    def get_reviews_for_product(self, product_id: str) -> List[Dict[str, Any]]:
        return [
            review for review in self.reviews
            if str(review.get("product_id")) == str(product_id)
        ]
