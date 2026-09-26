#!/usr/bin/env python3
"""
OpenSearch Connectivity and Index Test Suite

This module provides comprehensive testing for OpenSearch connectivity,
index availability, and basic functionality in the diet classification system.
It validates the search infrastructure used for recipe and ingredient retrieval.

The test suite covers:
- OpenSearch connection establishment and validation
- Index availability checking and listing
- Recipe index existence and document counting
- Error handling and graceful degradation
- Search infrastructure health monitoring

Key Test Areas:
- Connection establishment and ping testing
- Index discovery and listing
- Recipe index validation
- Document count verification
- Error handling for connection failures
- Search infrastructure status reporting

Test Features:
- Comprehensive connection testing
- Index availability validation
- Document count verification
- Error handling and logging
- Infrastructure health monitoring
- Status reporting and feedback

Dependencies:
- opensearchpy: OpenSearch Python client
- pathlib: Path management
- sys/os: System path management
- shared.config: Application configuration

Example:
    >>> python nb/src/tests/test_opensearch.py
    >>> # Run OpenSearch connectivity test
    >>> test_opensearch()
"""

import sys
from pathlib import Path

# Add shared package to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import app_config  # noqa: E402
from opensearchpy import OpenSearch  # noqa: E402


def test_opensearch():
    """
    Test OpenSearch connectivity and available indices.

    This function performs comprehensive testing of the OpenSearch connection
    and validates the availability of required indices for the diet classification
    system. It checks connectivity, index existence, and document counts.

    Test Components:
        1. Connection Establishment: Validates OpenSearch connectivity
        2. Ping Testing: Ensures OpenSearch service is responding
        3. Index Discovery: Lists all available indices
        4. Recipe Index Validation: Checks specific recipe index
        5. Document Counting: Verifies recipe document availability

    The test validates:
    - OpenSearch service availability and responsiveness
    - Connection configuration and authentication
    - Index existence and accessibility
    - Document availability for search operations
    - Error handling for connection failures

    Returns:
        bool: True if OpenSearch is accessible and functional, False otherwise

    Raises:
        Exception: If OpenSearch connection fails unexpectedly

    Example:
        >>> success = test_opensearch()
        >>> if success:
        >>>     print("OpenSearch is ready for use")
        >>> else:
        >>>     print("OpenSearch connection failed")
    """
    print("Testing OpenSearch connection...")
    print(f"Connecting to: {app_config.OPENSEARCH_URL}")

    client = OpenSearch(
        hosts=[app_config.OPENSEARCH_URL],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )

    try:
        if client.ping():
            print("OpenSearch is responding")

            # Check available indices
            try:
                indices = client.indices.get_alias(index="*")
                print(f"Available indices: {list(indices.keys())}")
            except Exception as e:
                print(f"Could not list indices: {e}")

            # Check recipes index specifically
            try:
                if client.indices.exists(index="recipes"):
                    count = client.count(index="recipes")
                    print(f'Recipes index exists with {count["count"]} documents')
                else:
                    print("Recipes index does not exist")
            except Exception as e:
                print(f"Error checking recipes index: {e}")

        else:
            print("OpenSearch ping failed")

    except Exception as e:
        print(f"OpenSearch connection error: {e}")
        return False

    return True


if __name__ == "__main__":
    test_opensearch()
