import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

mock_groq = types.ModuleType("config.groq_client")
mock_groq.client = SimpleNamespace(
    chat=SimpleNamespace(completions=SimpleNamespace(create=None))
)
mock_groq.MODEL_NAME = "test-model"
sys.modules.setdefault("config.groq_client", mock_groq)

from agents.comparison_agent import comparison_agent
from agents.comparison_retrieval_agent import (
    comparison_retrieval_agent,
    is_requested_product_match,
)
from agents.comparison_validation_agent import comparison_validation_agent
from agents.explanation_agent import explanation_agent
from agents.fallback_agent import fallback_agent
from agents.preference_agent import preference_agent
from agents.supervisor import supervisor_agent


def _llm_response(preferences):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(preferences))
            )
        ]
    )


class ComparisonFlowTests(unittest.TestCase):
    def setUp(self):
        self.llm = patch(
            "agents.preference_agent.client.chat.completions.create"
        )
        self.create = self.llm.start()
        self.addCleanup(self.llm.stop)

    def _preferences_response(self, intent="general", criteria=None):
        return {
            "intent": intent,
            "comparison_products": [],
            "comparison_criteria": criteria or [],
            "recommendation_query": intent == "recommendation",
        }

    def _extract(self, query, intent="general", criteria=None):
        self.create.return_value = _llm_response(
            self._preferences_response(intent, criteria)
        )
        return preference_agent({"query": query})["preferences"]

    def _retrieve_and_validate(self, query):
        preferences = self._extract(query)
        state = {"query": query, "preferences": preferences}
        comparison_retrieval_agent(state)
        if state.get("route") == "comparison_validation":
            comparison_validation_agent(state)
        return state

    def test_comparison_exists_and_generates_multi_criteria_response(self):
        query = (
            "Compare Redmi K50i 5G 6GB RAM 128GB Storage and "
            "OnePlus Nord CE 2 Lite 5G 6GB RAM 128GB Storage based on "
            "price, battery, camera, and performance."
        )
        state = self._retrieve_and_validate(query)

        self.assertEqual(state["preferences"]["intent"], "comparison")
        self.assertEqual(len(state["validated_comparison_products"]), 2)
        self.assertEqual(state["route"], "comparison")

        comparison_agent(state)
        self.assertEqual(state["route"], "comparison_explanation")
        explanation_agent(state)
        self.assertEqual(state["comparison_status"], "completed")
        self.assertIn("Price", state["response"])
        self.assertIn("Battery", state["response"])
        self.assertIn("Camera", state["response"])
        self.assertIn("performance", state["response"].lower())
        self.assertEqual(self.create.call_count, 1)

    def test_case_punctuation_and_optional_5g_suffix(self):
        query = (
            "Compare redmi-k50i 6gb ram 128gb storage and "
            "oneplus nord ce 2 lite 6gb ram 128gb storage."
        )
        state = self._retrieve_and_validate(query)
        self.assertEqual(state["comparison_status"], "validated")
        self.assertEqual(len(state["validated_comparison_products"]), 2)

    def test_difference_versus_and_which_is_better_queries(self):
        examples = (
            (
                "What is the difference between Redmi 10 Power and "
                "OnePlus Nord CE 2 Lite?",
                ["Redmi 10 Power", "OnePlus Nord CE 2 Lite"],
            ),
            (
                "Redmi 10 Power versus OnePlus Nord CE 2 Lite",
                ["Redmi 10 Power", "OnePlus Nord CE 2 Lite"],
            ),
            (
                "Which is better, Redmi 10 Power or OnePlus Nord CE 2 Lite?",
                ["Redmi 10 Power", "OnePlus Nord CE 2 Lite"],
            ),
        )
        for query, expected_products in examples:
            with self.subTest(query=query):
                preferences = self._extract(query)
                self.assertEqual(preferences["intent"], "comparison")
                self.assertEqual(
                    preferences["comparison_products"],
                    expected_products,
                )

    def test_comparison_detection_survives_invalid_llm_json(self):
        self.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))]
        )
        state = preference_agent(
            {
                "query": (
                    "What is the difference between Redmi 10 Power and "
                    "OnePlus Nord CE 2 Lite?"
                )
            }
        )
        self.assertEqual(state["preferences"]["intent"], "comparison")
        self.assertEqual(
            state["preferences"]["comparison_products"],
            ["Redmi 10 Power", "OnePlus Nord CE 2 Lite"],
        )

    def test_fuzzy_model_typo_does_not_require_exact_name(self):
        query = (
            "Compare Onepluss Nord CE 2 Lite 6GB RAM 128GB Storage and "
            "Redmi 10 Power."
        )
        state = self._retrieve_and_validate(query)
        self.assertEqual(state["comparison_status"], "validated")
        self.assertEqual(len(state["validated_comparison_products"]), 2)

    def test_ram_and_storage_select_requested_variants(self):
        query = (
            "Compare Redmi K50i 5G 8GB RAM 256GB Storage and "
            "Redmi K50i 5G 6GB RAM 128GB Storage."
        )
        state = self._retrieve_and_validate(query)
        self.assertEqual(state["comparison_status"], "validated")
        names = [
            product["name"]
            for product in state["validated_comparison_products"]
        ]
        self.assertIn("8GB RAM", names[0])
        self.assertIn("256GB Storage", names[0])
        self.assertIn("6GB RAM", names[1])
        self.assertIn("128GB Storage", names[1])

    def test_one_missing_product_is_reported(self):
        query = "Compare Unknown Model Z 900 and Redmi 10 Power."
        state = self._retrieve_and_validate(query)
        self.assertEqual(state["comparison_status"], "missing_products")
        self.assertEqual(state["comparison_missing_products"], ["Unknown Model Z 900"])
        fallback_agent(state)
        self.assertIn("Unknown Model Z 900", state["response"])

    def test_both_missing_products_are_reported(self):
        query = "Compare Unknown Model Z 900 and Imaginary Phone 77."
        state = self._retrieve_and_validate(query)
        self.assertEqual(state["comparison_status"], "missing_products")
        self.assertEqual(
            state["comparison_missing_products"],
            ["Unknown Model Z 900", "Imaginary Phone 77"],
        )
        fallback_agent(state)
        self.assertIn("Unknown Model Z 900", state["response"])
        self.assertIn("Imaginary Phone 77", state["response"])

    def test_ambiguous_model_requests_variant_clarification(self):
        query = "Compare OnePlus Nord CE 2 Lite and Redmi 10 Power."
        state = self._retrieve_and_validate(query)
        self.assertTrue(state["comparison_status"].startswith("ambiguous_product:"))
        fallback_agent(state)
        self.assertIn("multiple catalog variants", state["response"])
        self.assertIn("RAM/storage", state["response"])

    def test_validation_rejects_a_different_model(self):
        state = {
            "preferences": {
                "comparison_products": ["Redmi K50", "Redmi 10 Power"],
            },
            "comparison_retrieved_products": [
                {
                    "name": (
                        "Redmi K50i 5G "
                        "(Stealth Black, 6GB RAM, 128GB Storage)"
                    )
                },
                {"name": "Redmi 10 Power (Power Black, 8GB RAM, 128GB Storage)"},
            ],
        }
        comparison_validation_agent(state)
        self.assertEqual(
            state["comparison_status"],
            "product_match_not_verified: Redmi K50",
        )

    def test_recommendation_intent_is_preserved(self):
        preferences = self._extract(
            "Recommend a OnePlus phone under 25000 with 8GB RAM.",
            intent="recommendation",
        )
        self.assertEqual(preferences["intent"], "recommendation")
        self.assertTrue(preferences["recommendation_query"])
        state = {"preferences": preferences}
        supervisor_agent(state)
        self.assertEqual(state["route"], "retrieval")

    def test_non_shopping_query_keeps_general_fallback(self):
        query = "What is agentic AI?"
        preferences = self._extract(
            query,
            intent="general",
        )
        self.assertEqual(preferences["intent"], "general")
        self.assertFalse(preferences["recommendation_query"])
        self.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="General fallback response")
                )
            ]
        )
        state = {"query": query, "preferences": preferences}
        fallback_agent(state)
        self.assertEqual(state["response"], "General fallback response")

    def test_validator_accepts_dataset_variant_suffixes(self):
        product = {
            "name": (
                "Redmi K50i 5G (Stealth Black, 6GB RAM, 128GB Storage) | "
                "Flagship Mediatek Dimensity 8100 Processor"
            )
        }
        self.assertTrue(
            is_requested_product_match(
                "redmi-k50i 6gb ram 128gb storage",
                product,
            )
        )


if __name__ == "__main__":
    unittest.main()
