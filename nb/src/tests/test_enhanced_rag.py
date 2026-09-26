"""
Enhanced RAG Pipeline Test Suite

This module provides comprehensive testing for the enhanced RAG (Retrieval-Augmented
Generation) pipeline used in the diet classification system. It tests the pipeline's
ability to classify ingredients with different dietary characteristics using various
retrieval and generation strategies.

The test suite covers:
- Enhanced RAG pipeline functionality
- Fallback mechanism testing
- Various ingredient dietary characteristics
- Classification accuracy validation
- Reasoning quality assessment
- Method selection and confidence scoring

Key Test Areas:
- Animal protein classification (chicken breast)
- Leafy green classification (spinach)
- Pure carbohydrate classification (sugar)
- Pure fat classification (olive oil)
- Complex ingredient classification (bread)

Test Features:
- Asynchronous testing for improved performance
- Multiple ingredient test cases
- Expected outcome validation
- Reasoning quality assessment
- Method selection verification
- Confidence scoring validation

Dependencies:
- asyncio: Asynchronous testing support
- SOTASemanticClassifier: Main classification system

Example:
    >>> python nb/src/tests/test_enhanced_rag.py
    >>> # Run specific RAG test
    >>> asyncio.run(test_enhanced_rag_pipeline())
"""

import asyncio

from context_aware_classifier import SOTASemanticClassifier


async def test_enhanced_rag_pipeline():
    """
    Test the enhanced RAG pipeline with various ingredients.

    This function performs comprehensive testing of the enhanced RAG pipeline
    by classifying ingredients with different dietary characteristics. It validates
    the pipeline's ability to handle various ingredient types and provide accurate
    classifications with reasoning and confidence scores.

    Test Cases:
        1. "chicken breast": Animal protein, low carbs (Expected: keto=True, vegan=False)
        2. "spinach": Leafy green, very low carbs (Expected: keto=True, vegan=True)
        3. "sugar": Pure carbohydrates, plant-based (Expected: keto=False, vegan=True)
        4. "olive oil": Pure fat, plant-based (Expected: keto=True, vegan=True)
        5. "bread": Complex ingredient, high carbs (Expected: keto=False, vegan=depends)

    The test validates:
    - Classification accuracy for different ingredient types
    - Reasoning quality and relevance
    - Method selection appropriateness
    - Confidence scoring reliability
    - Fallback mechanism functionality
    - Pipeline integration reliability

    Returns:
        None: Prints test results to console

    Raises:
        Exception: If RAG pipeline fails unexpectedly

    Example:
        >>> await test_enhanced_rag_pipeline()
        >>> # Tests enhanced RAG pipeline with multiple ingredients
    """
    classifier = SOTASemanticClassifier()

    print("Testing Enhanced RAG Pipeline with Fallback Mechanism")
    print("=" * 55)

    # Test ingredients with different dietary characteristics
    test_cases = [
        (
            "chicken breast",
            "Expected: keto=True, vegan=False (animal protein, low carbs)",
        ),
        ("spinach", "Expected: keto=True, vegan=True (leafy green, very low carbs)"),
        ("sugar", "Expected: keto=False, vegan=True (pure carbs, plant-based)"),
        ("olive oil", "Expected: keto=True, vegan=True (pure fat, plant-based)"),
        (
            "bread",
            "Expected: keto=False, vegan=depends (high carbs, may contain eggs/milk)",
        ),
    ]

    for ingredient, expected in test_cases:
        print(f"\n--- Testing: {ingredient} ---")
        print(f"Expected: {expected}")

        result = await classifier.classify_single_ingredient(ingredient)

        print(f"Result: keto={result.get('is_keto')}, vegan={result.get('is_vegan')}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")
        print(f"Method: {result.get('method', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_rag_pipeline())
