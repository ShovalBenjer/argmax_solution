#!/usr/bin/env python3
"""
Quick Test of Pre-computation Pipeline

Tests the SOTA semantic classifier on a small sample of ingredients
to verify the event loop fixes and performance.
"""

import sys
import time
from pathlib import Path

# Add nb/src to path for accessing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))

from diet_classifiers import is_keto, is_vegan  # noqa: E402


def test_precomputation():
    """Test pre-computation on sample ingredients."""
    print("Testing SOTA Semantic Classifier Pre-computation")
    print("=" * 60)

    # Test with a small sample of ingredients
    test_ingredients = [
        "spinach, raw",
        "chicken breast",
        "olive oil",
        "sugar",
        "butter",
    ]

    print(f"Testing with {len(test_ingredients)} sample ingredients...")

    results = []
    total_time = 0

    for i, ingredient in enumerate(test_ingredients, 1):
        print(f"\n{i}. Testing: {ingredient}")
        print("-" * 30)

        start_time = time.time()

        try:
            # Test both classifications
            keto_result = is_keto([ingredient])
            vegan_result = is_vegan([ingredient])

            end_time = time.time()
            classification_time = end_time - start_time
            total_time += classification_time

            print(f"   Keto: {keto_result}")
            print(f"   Vegan: {vegan_result}")
            print(f"   Time: {classification_time:.2f}s")

            results.append(
                {
                    "ingredient": ingredient,
                    "keto": keto_result,
                    "vegan": vegan_result,
                    "time": classification_time,
                    "status": "success",
                }
            )

        except Exception as e:
            end_time = time.time()
            classification_time = end_time - start_time
            total_time += classification_time

            print(f"   Error: {e}")
            print(f"   Time: {classification_time:.2f}s")

            results.append(
                {
                    "ingredient": ingredient,
                    "error": str(e),
                    "time": classification_time,
                    "status": "error",
                }
            )

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"Successful: {len(successful)}/{len(test_ingredients)}")
    print(f"Errors: {len(errors)}/{len(test_ingredients)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time / len(test_ingredients):.2f}s per ingredient")

    if successful:
        avg_success_time = sum(r["time"] for r in successful) / len(successful)
        print(f"Average success time: {avg_success_time:.2f}s")

        # Estimate full database processing time
        estimated_hours = (8789 * avg_success_time) / 3600
        print(f"Estimated time for 8,789 ingredients: {estimated_hours:.1f} hours")

    if errors:
        print("\nError details:")
        for error in errors:
            print(f"   • {error['ingredient']}: {error['error']}")

    print(f"\n{'SUCCESS' if len(successful) >= 3 else 'NEEDS WORK'}: Test completed")
    return len(successful), len(errors)


if __name__ == "__main__":
    successful, errors = test_precomputation()

    if successful >= 3:
        print("\nReady to run full pre-computation!")
        print("   Run: python scripts/05_precompute_classifications.py --batch-size 10")
    else:
        print("\nFix issues before running full pre-computation")

    sys.exit(0 if errors == 0 else 1)
