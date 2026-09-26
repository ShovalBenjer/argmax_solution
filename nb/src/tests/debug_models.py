"""
Model Detection Debugging Utility

This module provides comprehensive debugging capabilities for LLM model
detection and availability issues in the diet classification system.
It helps diagnose problems with model loading, availability checking,
and client configuration.

The debug utility covers:
- Raw model list inspection and analysis
- Model availability testing and validation
- Client configuration verification
- Model name matching and comparison
- Detailed model information extraction

Key Debug Features:
- Comprehensive model list analysis
- Model availability validation
- Client configuration testing
- Model name matching verification
- Detailed error reporting and logging

Supported Model Types:
- Arctic Text2SQL models for database queries
- Qwen classification models for dietary analysis
- Various model name formats and versions

Dependencies:
- llm_client: LLM client for model interaction
- json: Data serialization and formatting
- sys/os: Path management and system access

Example:
    >>> python nb/src/debug_models.py
    >>> # Run comprehensive model debugging
    >>> debug_model_detection()
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from llm_client import LLMClient  # noqa: E402


def debug_model_detection():
    """
    Debug model detection issues and validate client configuration.

    This function performs comprehensive debugging of the LLM client's
    model detection capabilities. It analyzes the raw model list response,
    tests model availability checking, and validates client configuration.

    Debug Operations:
        1. Raw model list inspection and analysis
        2. Model response type and structure validation
        3. Individual model detail extraction
        4. Model availability testing for specific models
        5. Manual model name matching verification

    Test Models:
        - "arctic-text2sql-r1-7b"
        - "qwen/qwen3-0.6b-gguf:q8_0"
        - "qwen/qwen3-0.6b-gguf"

    The function validates:
    - Client initialization and configuration
    - Model list retrieval functionality
    - Model availability checking accuracy
    - Model name matching and comparison
    - Response format and structure consistency

    Returns:
        None: Prints comprehensive debug information to console

    Raises:
        Exception: If client operations fail unexpectedly

    Example:
        >>> debug_model_detection()
        >>> # Performs comprehensive model debugging and validation
    """
    print("Debugging Model Detection")
    print("=" * 30)

    client = LLMClient()

    # Get raw model list
    models = client.list_models()
    print(f"\nRaw models response: {models}")
    print(f"Type: {type(models)}")
    print(f"Length: {len(models) if models else 'None'}")

    if models:
        print("\nModel details:")
        for i, model in enumerate(models):
            print(f"  {i}: {model}")
            print(f"      Type: {type(model)}")
            if isinstance(model, dict):
                print(f"      Keys: {list(model.keys())}")
                print(f"      Name: {model.get('name', 'NO NAME')}")

    # Test specific model checks
    test_models = [
        "arctic-text2sql-r1-7b",
        "qwen/qwen3-0.6b-gguf:q8_0",
        "qwen/qwen3-0.6b-gguf",
    ]

    print("\nTesting model availability:")
    for model_name in test_models:
        available = client.is_model_available(model_name)
        print(f"  {model_name}: {available}")

        # Manual check
        if models:
            manual_check = any(model.get("name", "") == model_name for model in models)
            startswith_check = any(
                model.get("name", "").startswith(model_name) for model in models
            )
            print(f"    Manual exact match: {manual_check}")
            print(f"    Startswith check: {startswith_check}")


if __name__ == "__main__":
    debug_model_detection()
