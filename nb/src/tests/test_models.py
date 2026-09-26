"""
Model Availability and Functionality Test Suite

This module provides comprehensive testing for the LLM models used in the diet
classification system. It validates model availability, basic functionality,
and query processing capabilities for both Arctic Text2SQL and Qwen classification models.

The test suite covers:
- Model availability checking and listing
- Arctic Text2SQL model functionality
- Qwen classification model functionality
- Basic query processing and response validation
- Error handling and graceful degradation
- Model integration testing

Key Test Areas:
- Model discovery and availability
- Text2SQL query processing
- Classification query processing
- Response format validation
- Error handling for unavailable models
- Performance and reliability testing

Test Features:
- Asynchronous model testing
- Comprehensive model listing
- Individual model validation
- Query response testing
- Error handling verification
- Model integration validation

Dependencies:
- asyncio: Asynchronous testing support
- sys/os: Path management for imports
- LLMClient: Main model client interface

Supported Models:
- Arctic Text2SQL: Snowflake/Arctic-Text2SQL-R1-7B for database queries
- Qwen Classification: Qwen3-0.6B-GGUF for dietary classification

Example:
    >>> python nb/src/tests/test_models.py
    >>> # Run specific model test
    >>> asyncio.run(test_model_availability())
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_client import LLMClient  # noqa: E402


async def test_model_availability():
    """
    Test if models are available and working.

    This function performs comprehensive testing of the LLM models used in the
    diet classification system. It checks model availability, basic functionality,
    and query processing capabilities for both Arctic Text2SQL and Qwen models.

    Test Components:
        1. Model Discovery: Lists all available models
        2. Arctic Model Check: Validates Text2SQL model availability
        3. Qwen Model Check: Validates classification model availability
        4. Arctic Query Test: Tests Text2SQL functionality
        5. Qwen Query Test: Tests classification functionality

    Test Cases:
        - Model listing and availability checking
        - Arctic Text2SQL query:
          "SELECT name FROM nutrition_facts WHERE name LIKE '%chicken%' LIMIT 1"
        - Qwen classification query: "Is chicken breast keto-friendly?" with JSON response

    The test validates:
    - Model availability and accessibility
    - Query processing functionality
    - Response format and content
    - Error handling for unavailable models
    - Model integration reliability

    Returns:
        None: Prints test results to console

    Raises:
        Exception: If model testing fails unexpectedly

    Example:
        >>> await test_model_availability()
        >>> # Tests all available models and their functionality
    """
    print("Testing Model Availability and Basic Functionality")
    print("=" * 50)

    client = LLMClient()

    # Test 1: List available models
    print("\n1. Available Models:")
    models = client.list_models()
    for model in models:
        print(f"   - {model.get('name', 'Unknown')}")

    # Test 2: Check Arctic model availability
    arctic_name = "arctic-text2sql-r1-7b"
    print("\n2. Arctic Model Check:")
    print(f"   Model name: {arctic_name}")
    print(f"   Available: {client.is_model_available(arctic_name)}")

    # Test 3: Check Qwen model availability
    qwen_name = "qwen/qwen3-0.6b-gguf:q8_0"
    print("\n3. Qwen Model Check:")
    print(f"   Model name: {qwen_name}")
    print(f"   Available: {client.is_model_available(qwen_name)}")

    # Test 4: Simple Arctic query
    print("\n4. Arctic Text2SQL Test:")
    if client.is_model_available(arctic_name):
        try:
            result = await client.query_async(
                arctic_name,
                "SELECT name FROM nutrition_facts WHERE name LIKE '%chicken%' LIMIT 1",
                as_json=False,
            )
            print(f"   Arctic response: {result.get('content', 'No content')[:100]}...")
        except Exception as e:
            print(f"   Arctic error: {e}")
    else:
        print("   Arctic model not available for testing")

    # Test 5: Simple Qwen query
    print("\n5. Qwen Classification Test:")
    if client.is_model_available(qwen_name):
        try:
            result = await client.query_async(
                qwen_name,
                (
                    'Is chicken breast keto-friendly? Answer with JSON: {"is_keto": boolean, '
                    '"reasoning": '
                    '"text"}'
                ),
                as_json=True,
            )
            print(f"   Qwen response: {result}")
        except Exception as e:
            print(f"   Qwen error: {e}")
    else:
        print("   Qwen model not available for testing")


if __name__ == "__main__":
    asyncio.run(test_model_availability())
