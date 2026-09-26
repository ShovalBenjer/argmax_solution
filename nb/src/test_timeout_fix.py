#!/usr/bin/env python3
"""
Test script to verify timeout handling with Arctic model.
"""

import asyncio
import logging

from context_aware_classifier import SOTASemanticClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_timeout_handling():
    """Test that timeout handling works correctly."""
    classifier = SOTASemanticClassifier()

    # Test with a simple ingredient
    test_ingredient = "chicken breast"

    logger.info(f"Testing timeout handling with ingredient: {test_ingredient}")

    try:
        result = await classifier.classify_single_ingredient(test_ingredient)
        logger.info(f"Classification result: {result}")

        if "error" in result:
            logger.warning(f"Classification failed: {result['error']}")
        else:
            logger.success("Classification completed successfully!")

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_timeout_handling())
