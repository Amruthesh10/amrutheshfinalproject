"""Product-related service layer."""

from __future__ import annotations

from typing import Any, Dict, List


class ProductService:
    def __init__(self, products: List[Dict[str, Any]] | None = None):
        self.products = products or []

    def get_all_products(self) -> List[Dict[str, Any]]:
        return self.products

    def get_product_by_id(self, product_id: str) -> Dict[str, Any] | None:
        for product in self.products:
            if str(product.get("product_id")) == str(product_id):
                return product
        return None
