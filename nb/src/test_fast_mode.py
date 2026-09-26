#!/usr/bin/env python3
"""
Test script to verify fast mode works correctly.
"""

import asyncio
import logging

from context_aware_classifier import SOTASemanticClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_fast_mode():
    """Test that fast mode works correctly."""
    # Test with fast mode enabled
    classifier_fast = SOTASemanticClassifier(fast_mode=True)

    # Test with a simple ingredient
    test_ingredient = "chicken breast"

    logger.info(f"Testing fast mode with ingredient: {test_ingredient}")

    try:
        result = await classifier_fast.classify_single_ingredient(test_ingredient)
        logger.info(f"Fast mode classification result: {result}")

        if "error" in result:
            logger.warning(f"Fast mode classification failed: {result['error']}")
        else:
            logger.success("Fast mode classification completed successfully!")

    except Exception as e:
        logger.error(f"Fast mode test failed with exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_fast_mode())
