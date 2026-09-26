#!/usr/bin/env python3
"""
Test script to verify ingredient parser functionality.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ingredient_parser():
    """Test the unified ingredient parser."""
    try:
        from unified_ingredient_parser import get_parser, parse_ingredient_simple

        logger.info("Unified ingredient parser imported successfully")

        get_parser()

        # Test cases
        test_cases = [
            ("100g chicken breast", "chicken_breast"),
            ("2 tbsp olive oil", "olive_oil"),
            ("1 cup all-purpose flour, sifted", "allpurpose_flour"),
            ("3 medium eggs", "eggs"),
            ("chopped onions", "onions"),
            ("1 lb ground beef, 80/20", "ground_beef"),
            ("2 cups whole milk", "whole_milk"),
        ]

        logger.info("Testing ingredient parsing:")
        all_passed = True

        for input_text, expected in test_cases:
            result = parse_ingredient_simple(input_text)
            status = "PASS" if result == expected else "FAIL"
            logger.info(
                f"  {status} '{input_text}' -> '{result}' (expected: '{expected}')"
            )
            if result != expected:
                all_passed = False

        if all_passed:
            logger.info("All ingredient parsing tests passed!")
        else:
            logger.error("Some ingredient parsing tests failed!")

        return all_passed

    except Exception as e:
        logger.error(f"Ingredient parser test failed: {e}")
        return False


def test_diet_classifiers():
    """Test that diet classifiers can use the unified parser."""
    try:
        from diet_classifiers import parse_ingredient_name, parse_ingredients_input

        logger.info("Diet classifiers imported successfully")

        # Test parse_ingredient_name
        result1 = parse_ingredient_name("100g chicken breast")
        logger.info(f"  parse_ingredient_name: '100g chicken breast' -> '{result1}'")

        # Test parse_ingredients_input
        result2 = parse_ingredients_input(["chicken breast", "spinach", "olive oil"])
        logger.info(f"  parse_ingredients_input: {result2}")

        logger.info("Diet classifier integration tests passed!")
        return True

    except Exception as e:
        logger.error(f"Diet classifier test failed: {e}")
        return False


def test_simple_classification():
    """Test a simple classification to see if the pipeline works."""
    try:
        from diet_classifiers import is_ingredient_keto, is_ingredient_vegan

        logger.info("Testing simple classification...")

        # Test a simple case
        keto_result = is_ingredient_keto("chicken breast")
        vegan_result = is_ingredient_vegan("chicken breast")

        logger.info(f"  chicken breast: keto={keto_result}, vegan={vegan_result}")

        logger.info("Simple classification test completed!")
        return True

    except Exception as e:
        logger.error(f"Simple classification test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Testing Ingredient Parser and Classification Pipeline")
    logger.info("=" * 60)

    success = True

    # Test ingredient parser
    if not test_ingredient_parser():
        success = False

    # Test diet classifier integration
    if not test_diet_classifiers():
        success = False

    # Test simple classification
    if not test_simple_classification():
        success = False

    if success:
        logger.info("\nAll tests passed! Ingredient parser is working correctly.")
        sys.exit(0)
    else:
        logger.error("\n💥 Some tests failed! Please check the implementation.")
        sys.exit(1)
