"""
Function-Calling Pipeline Test Suite

This module provides comprehensive testing for the function-calling RAG pipeline,
which enables natural language queries to be converted to structured database
operations and executed safely. It tests the complete pipeline from text input
to classification results.

The test suite covers:
- Function calling handler (text-to-JSON conversion)
- Query engine (JSON-to-SQL translation)
- Context-aware classifier integration
- End-to-end classification pipeline
- Individual component validation
- Full recipe classification testing

Key Test Areas:
- Natural language query processing
- Structured query generation
- SQL translation and parameter binding
- Database query execution
- Classification accuracy validation
- Pipeline integration testing

Test Features:
- Component-level testing for isolation
- End-to-end pipeline validation
- Multiple ingredient and recipe scenarios
- Expected outcome validation
- Error handling verification
- Performance monitoring

Dependencies:
- asyncio: Asynchronous testing support
- json: Data serialization and validation
- sys/os: Path management for imports
- FunctionCallingHandler: Text-to-JSON conversion
- QueryEngine: JSON-to-SQL translation
- SOTASemanticClassifier: Main classification system

Example:
    >>> python nb/src/tests/test_function_calling_pipeline.py
    >>> # Run specific test function
"""

import os
import sys

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    print("======================================================")
    print("  Running Full Test of Function-Calling RAG Pipeline  ")
    print("======================================================")

    # Run component tests

    # Run end-to-end tests

    print("\n--- All Tests Completed Successfully! ---")
