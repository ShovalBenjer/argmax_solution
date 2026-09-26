"""
DeepEval Pipeline Tests for Function-Calling RAG Architecture

This module provides comprehensive testing for the function-calling RAG pipeline
using the DeepEval framework. It evaluates the complete pipeline from ingredient
parsing to final classification, ensuring factual consistency and relevancy.

The DeepEval tests cover:
- Ingredient parsing accuracy and structured extraction
- Function calling handler query generation quality
- Query engine safety and execution validation
- Context retrieval relevancy assessment
- Classification factual consistency verification
- End-to-end pipeline coherence testing

Key Test Components:
1. FunctionCallingHandler: Converts ingredients to structured JSON queries
2. QueryEngine: Safely executes JSON queries against the database
3. ContextAwareDietClassifier: Orchestrates the full pipeline
4. Final judge model: Makes classifications based on retrieved facts

DeepEval Metrics Used:
- GEval: Custom evaluation criteria for specific tasks
- ContextualRelevancyMetric: Measures context relevance
- HallucinationMetric: Detects factual inconsistencies
- AnswerRelevancyMetric: Assesses answer quality

Test Features:
- Asynchronous testing for improved performance
- Comprehensive error handling and logging
- Mock context generation for controlled testing
- Factual consistency validation
- Pipeline coherence verification

Dependencies:
- pytest: Testing framework
- deepeval: RAG evaluation framework
- asyncio: Asynchronous testing support
- json: Data serialization
- logging: Test logging and debugging

Example:
    >>> pytest nb/src/tests/test_deepeval_pipeline.py -v
    >>> # Run specific DeepEval test
    >>> pytest nb/src/tests/test_deepeval_pipeline.py::test_ingredient_parser_accuracy -v
"""

import json
import logging
from pathlib import Path
from typing import Dict

import pytest

# Try to import deepeval, skip tests if not available
try:
    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        GEval,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

    # Create mock classes for when deepeval is not available
    class MockLLMTestCase:
        def __init__(self, input=None, actual_output=None, expected_output=None):
            self.input = input
            self.actual_output = actual_output
            self.expected_output = expected_output

    class MockGEval:
        def __init__(self, name=None, criteria=None, evaluation_params=None):
            self.name = name
            self.criteria = criteria
            self.evaluation_params = evaluation_params

    LLMTestCase = MockLLMTestCase
    GEval = MockGEval

    def assert_test(test_case, metrics):
        return None


# Add nb/src to path for local imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_aware_classifier import ContextAwareDietClassifier  # noqa: E402

# Import the components to be tested
from function_calling_handler import FunctionCallingHandler  # noqa: E402
from query_engine import QueryEngine  # noqa: E402
from unified_ingredient_parser import parse_ingredient  # noqa: E402

# Configure logging
logger = logging.getLogger(__name__)

# Skip all tests if deepeval is not available
pytestmark = pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="DeepEval not available")

# --- Helper Functions for Testing ---


def create_test_context(
    ingredient_name: str, nutrition_data: Dict = None, vegan_data: Dict = None
) -> str:
    """
    Create a mock context for testing purposes.

    This helper function generates mock context data for testing the RAG pipeline
    components. It creates structured context that mimics the format returned
    by the actual database queries.

    Args:
        ingredient_name (str): Name of the ingredient to create context for
        nutrition_data (Dict, optional): Mock nutrition data. Defaults to None.
        vegan_data (Dict, optional): Mock vegan classification data. Defaults to None.

    Returns:
        str: JSON-formatted string containing the mock context

    Example:
        >>> context = create_test_context("chicken breast",
        >>>                              {"calories": 165, "protein": 31},
        >>>                              {"is_explicitly_non_vegan": True})
        >>> print(context)
    """
    context = {
        "ingredient": ingredient_name,
        "nutrition_data": nutrition_data
        or {"calories": 100, "carbohydrates": 5, "protein": 20},
        "vegan_data": vegan_data
        or {"is_explicitly_non_vegan": False, "requires_contextual_check": True},
    }
    return json.dumps(context, indent=2)


# --- Test Suite ---


@pytest.mark.asyncio
async def test_ingredient_parser_accuracy():
    """
    Tests the accuracy of the ingredient-parser-nlp for structured extraction.

    This test validates that the ingredient parser can correctly extract
    structured information from complex ingredient strings. It uses DeepEval's
    GEval metric to assess parsing accuracy against expected outputs.

    Test Case:
        Input: "1 1/2 cups of sifted all-purpose flour, plus more for dusting"
        Expected: {"name": "all-purpose flour", "quantity": 1.5, "unit": "cup"}

    The test evaluates:
    - Ingredient name extraction accuracy
    - Quantity parsing and conversion
    - Unit identification and standardization
    - Handling of complex ingredient descriptions

    Raises:
        Exception: If parsing fails or produces unexpected results
        pytest.fail: If DeepEval evaluation fails
    """

    # Test case: Complex ingredient string
    test_input = "1 1/2 cups of sifted all-purpose flour, plus more for dusting"

    try:
        parsed_result = parse_ingredient(test_input)

        # Extract the actual components
        actual_output = {
            "name": (
                parsed_result.name.text
                if hasattr(parsed_result.name, "text")
                else str(parsed_result.name)
            ),
            "quantity": (
                float(parsed_result.amount[0].quantity)
                if parsed_result.amount and hasattr(parsed_result.amount[0], "quantity")
                else 1.0
            ),
            "unit": (
                str(parsed_result.amount[0].unit)
                if parsed_result.amount and hasattr(parsed_result.amount[0], "unit")
                else "unit"
            ),
        }

        # Define the evaluation metric
        parser_metric = GEval(
            name="Parsing Correctness",
            criteria=(
                "Evaluate if the 'actual_output' JSON correctly extracts the main ingredient "
                "'name', approximate 'quantity', and 'unit' from the 'input' ingredient string. "
                "The name should be the primary ingredient (e.g., 'flour'), quantity should be "
                "numeric, and unit should be "
                "appropriate."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
        )

        test_case = LLMTestCase(
            input=test_input,
            actual_output=json.dumps(actual_output),
            expected_output='{"name": "all-purpose flour", "quantity": 1.5, "unit": "cup"}',
        )

        assert_test(test_case, [parser_metric])
        logger.info("Ingredient parser accuracy test passed")

    except Exception as e:
        logger.error(f"Ingredient parser test failed: {e}")
        pytest.fail(f"Ingredient parser test failed: {e}")


@pytest.mark.asyncio
async def test_function_calling_handler_structure():
    """
    Tests that FunctionCallingHandler generates well-formed JSON queries.

    This test validates that the FunctionCallingHandler can generate properly
    structured JSON queries from ingredient inputs. It ensures the handler
    produces valid query structures that can be safely executed by the QueryEngine.

    Test Case:
        Input: "shredded sharp cheddar cheese"
        Expected: JSON with query_type and ingredient_name fields

    The test evaluates:
    - JSON structure validity
    - Required field presence (query_type, ingredient_name)
    - Query type appropriateness
    - Ingredient name extraction quality

    Raises:
        Exception: If handler fails to generate queries
        AssertionError: If query structure is invalid
        pytest.fail: If DeepEval evaluation fails
    """

    handler = FunctionCallingHandler()
    test_ingredient = "shredded sharp cheddar cheese"

    try:
        # Generate structured query
        result = await handler.generate_structured_query(test_ingredient)

        # Validate the structure
        assert isinstance(
            result, dict
        ), "Function calling handler should return a dictionary"
        assert "query_type" in result, "Result should contain query_type"
        assert "ingredient_name" in result, "Result should contain ingredient_name"

        # Define evaluation metric for structured output
        structure_metric = GEval(
            name="Structured Query Quality",
            criteria=(
                "Evaluate if the 'actual_output' is a well-formed JSON object with appropriate "
                "'query_type' and 'ingredient_name' fields for the given 'input' "
                "ingredient."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
        )

        test_case = LLMTestCase(
            input=test_ingredient,
            actual_output=json.dumps(result),
            expected_output='{"query_type": "nutrition", "ingredient_name": "cheddar cheese"}',
        )

        assert_test(test_case, [structure_metric])
        logger.info("Function calling handler structure test passed")

    except Exception as e:
        logger.error(f"Function calling handler test failed: {e}")
        pytest.fail(f"Function calling handler test failed: {e}")


@pytest.mark.asyncio
async def test_query_engine_safety_and_execution():
    """
    Tests that QueryEngine safely executes structured queries.

    This test validates the QueryEngine's ability to safely execute structured
    queries while protecting against SQL injection and other security threats.
    It tests both safe query execution and malicious query handling.

    Test Cases:
        Safe Query: {"query_type": "nutrition", "ingredient_name": "chicken breast"}
        Unsafe Query: {"query_type": "nutrition",
            "ingredient_name": "'; DROP TABLE nutrition_facts; --"}

    The test evaluates:
    - Safe query execution and result structure
    - Malicious query detection and handling
    - SQL injection prevention
    - Error handling for invalid queries

    Raises:
        Exception: If query execution fails unexpectedly
        AssertionError: If safety measures are not working
        pytest.fail: If security validation fails
    """

    engine = QueryEngine()

    # Test safe query execution
    safe_query = {"query_type": "nutrition", "ingredient_name": "chicken breast"}

    try:
        result = await engine.execute_structured_query(safe_query)

        # Validate result structure
        assert isinstance(result, dict), "Query engine should return a dictionary"

        # Test with potentially unsafe query (should be rejected)
        unsafe_query = {
            "query_type": "nutrition",
            "ingredient_name": "'; DROP TABLE nutrition_facts; --",
        }

        unsafe_result = await engine.execute_structured_query(unsafe_query)

        # The query should either be sanitized or return an error, not execute malicious SQL
        assert "error" in unsafe_result or "sanitized" in str(
            unsafe_result
        ), "Unsafe queries should be handled safely"

        logger.info("Query engine safety and execution test passed")

    except Exception as e:
        logger.error(f"Query engine test failed: {e}")
        pytest.fail(f"Query engine test failed: {e}")


@pytest.mark.asyncio
async def test_context_retrieval_relevancy():
    """
    Tests the relevancy of retrieved factual context.

    This test validates that the context retrieval system can provide
    relevant factual information for ingredient classification. It uses
    DeepEval's ContextualRelevancyMetric to assess context quality.

    Test Case:
        Input: "butter" ingredient
        Expected: Relevant context about butter's nutritional and dietary properties

    The test evaluates:
    - Context retrieval functionality
    - Context relevance to classification task
    - Factual accuracy of retrieved information
    - Context completeness and usefulness

    Raises:
        Exception: If context retrieval fails
        pytest.fail: If relevancy evaluation fails
    """

    classifier = ContextAwareDietClassifier()
    test_ingredient = "butter"

    try:
        # Get context for a known ingredient
        context_result = await classifier.get_ingredient_context(test_ingredient)

        if context_result and "context" in context_result:
            retrieved_context = context_result["context"]

            relevancy_metric = ContextualRelevancyMetric(threshold=0.7)
            test_case = LLMTestCase(
                input=f"Is {test_ingredient} vegan?",
                retrieval_context=[json.dumps(retrieved_context)],
            )

            assert_test(test_case, [relevancy_metric])
            logger.info("Context retrieval relevancy test passed")
        else:
            logger.warning(
                "No context retrieved for butter - may indicate database issue"
            )

    except Exception as e:
        logger.error(f"Context retrieval test failed: {e}")
        pytest.fail(f"Context retrieval test failed: {e}")


@pytest.mark.asyncio
async def test_classification_factual_consistency():
    """
    Tests that final classifications are factually consistent with retrieved context.

    This test validates that the final classification results are factually
    consistent with the retrieved context and known dietary facts. It uses
    DeepEval's HallucinationMetric to detect factual inconsistencies.

    Test Case:
        Input: ["chicken breast", "olive oil", "salt"]
        Expected: is_vegan=False, is_keto=True (chicken is not vegan but is keto-friendly)

    The test evaluates:
    - Factual consistency of classifications
    - Reasoning quality and accuracy
    - Absence of hallucinations or false claims
    - Alignment with known dietary facts

    Raises:
        Exception: If classification fails
        pytest.fail: If factual consistency evaluation fails
    """

    classifier = ContextAwareDietClassifier()

    # Test case: Recipe with obvious animal product
    test_ingredients = ["chicken breast", "olive oil", "salt"]

    try:
        result = await classifier.classify_recipe(test_ingredients)

        # Extract the reasoning and classification
        actual_output = {
            "is_vegan": result.get("is_vegan", False),
            "is_keto": result.get("is_keto", False),
            "reasoning": result.get("reasoning", "No reasoning provided"),
        }

        # Create mock retrieval context
        mock_context = create_test_context(
            "chicken breast",
            nutrition_data={"calories": 165, "carbohydrates": 0, "protein": 31},
            vegan_data={
                "is_explicitly_non_vegan": True,
                "requires_contextual_check": False,
            },
        )

        # Define metrics
        factual_consistency_metric = GEval(
            name="Factual Consistency",
            criteria=(
                "Based on the 'retrieval_context', is the classification in 'actual_output' "
                "factually correct? Chicken breast is not vegan but is "
                "keto-friendly."
            ),
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
        )

        hallucination_metric = HallucinationMetric(threshold=0.0)

        test_case = LLMTestCase(
            input=f"Classify recipe with ingredients: {', '.join(test_ingredients)}",
            actual_output=json.dumps(actual_output),
            retrieval_context=[mock_context],
            expected_output=(
                '{"is_vegan": false, "is_keto": true, "reasoning": "Contains chicken breast '
                "which is not vegan but is "
                'keto-friendly"}'
            ),
        )

        assert_test(test_case, [factual_consistency_metric, hallucination_metric])
        logger.info("Classification factual consistency test passed")

    except Exception as e:
        logger.error(f"Classification consistency test failed: {e}")
        pytest.fail(f"Classification consistency test failed: {e}")


@pytest.mark.asyncio
async def test_end_to_end_pipeline_coherence():
    """
    Tests the full pipeline from ingredient parsing to final classification.

    This test validates the complete end-to-end pipeline coherence, ensuring
    that all components work together seamlessly from ingredient input to
    final classification output. It tests the entire RAG pipeline workflow.

    Test Case:
        Input: ["1 cup almond flour", "2 tbsp coconut oil", "1 egg"]
        Expected: is_vegan=False (contains egg), is_keto=True (low carb ingredients)

    The test evaluates:
    - Complete pipeline functionality
    - Logical coherence of final classifications
    - Integration between all pipeline components
    - End-to-end workflow reliability

    Raises:
        Exception: If pipeline execution fails
        pytest.fail: If coherence evaluation fails
    """

    classifier = ContextAwareDietClassifier()

    # Test case: Mixed recipe with both vegan and non-vegan ingredients
    test_recipe = {
        "ingredients": ["1 cup almond flour", "2 tbsp coconut oil", "1 egg"],
        "expected_vegan": False,  # Contains egg
        "expected_keto": True,  # Low carb ingredients
    }

    try:
        result = await classifier.classify_recipe(test_recipe["ingredients"])

        # Validate pipeline coherence
        pipeline_metric = GEval(
            name="Pipeline Coherence",
            criteria=(
                "Evaluate if the 'actual_output' classification is logically coherent with the "
                "ingredients. The recipe contains egg (not vegan) but uses almond flour and "
                "coconut oil "
                "(keto-friendly)."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
        )

        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.8)

        test_case = LLMTestCase(
            input=f"Recipe with: {', '.join(test_recipe['ingredients'])}",
            actual_output=json.dumps(result),
            expected_output=(
                f'{{"is_vegan": {str(test_recipe["expected_vegan"]).lower()}, '
                f'"is_keto": {str(test_recipe["expected_keto"]).lower()}}}'
            ),
        )

        assert_test(test_case, [pipeline_metric, answer_relevancy_metric])
        logger.info("End-to-end pipeline coherence test passed")

    except Exception as e:
        logger.error(f"End-to-end pipeline test failed: {e}")
        pytest.fail(f"End-to-end pipeline test failed: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
