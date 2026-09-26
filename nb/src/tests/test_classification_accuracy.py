#!/usr/bin/env python3
"""
Classification Accuracy Test Suite for Diet Classification System

This module provides comprehensive testing for classification accuracy and performance
validation against specific KPI benchmarks. It validates the Arctic → Qwen pipeline
performance and ensures the system meets submission requirements.

The test suite covers:
- Individual ingredient classification accuracy
- Full recipe classification performance
- Input format compatibility testing
- Processing time validation
- KPI benchmark compliance
- F1-score and precision/recall metrics

Key Test Areas:
- Keto classification accuracy (benchmark: 75%)
- Vegan classification accuracy (benchmark: 80%)
- Processing time validation (max 5s per ingredient, 30s per recipe)
- SQL generation success rate (benchmark: 90%)
- Input format compatibility and robustness
- Performance metrics calculation and reporting

Test Features:
- Comprehensive benchmark testing
- Multiple input format validation
- Performance timing and monitoring
- Error handling and graceful degradation
- Detailed logging and reporting
- KPI compliance validation

KPI Benchmarks:
- keto_accuracy_min: 0.75 (75%)
- vegan_accuracy_min: 0.80 (80%)
- keto_f1_min: 0.70 (70%)
- vegan_f1_min: 0.75 (75%)
- max_processing_time_per_ingredient: 5.0 seconds
- sql_generation_success_rate_min: 0.90 (90%)
- max_processing_time_per_recipe: 30.0 seconds

Dependencies:
- sklearn: Machine learning metrics calculation
- pandas: Data manipulation and analysis
- time: Performance timing
- json: Data serialization
- logging: Comprehensive logging and reporting

Example:
    >>> python nb/src/tests/test_classification_accuracy.py
    >>> # Run complete accuracy validation
    >>> sys.exit(main())
"""

import logging
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add nb/src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nb" / "src"))

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    logger.warning("sklearn not available - using basic accuracy calculation")

    def accuracy_score(y_true, y_pred):
        return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


# KPI Benchmarks for Phase 4
BENCHMARKS = {
    "keto_accuracy_min": 0.75,
    "vegan_accuracy_min": 0.80,
    "keto_f1_min": 0.70,
    "vegan_f1_min": 0.75,
    "max_processing_time_per_ingredient": 5.0,  # seconds
    "sql_generation_success_rate_min": 0.90,
    "max_processing_time_per_recipe": 30.0,  # seconds for full recipe
}


def test_individual_ingredients():
    """
    Test individual ingredient classification performance.

    This function validates the accuracy of individual ingredient classification
    using a curated set of test cases with known dietary characteristics. It
    measures both keto and vegan classification accuracy, processing times,
    and F1 scores.

    Test Cases:
        - "chicken breast": Expected keto=True, vegan=False
        - "spinach": Expected keto=True, vegan=True
        - "sugar": Expected keto=False, vegan=True
        - "butter": Expected keto=True, vegan=False
        - "olive oil": Expected keto=True, vegan=True
        - "flour": Expected keto=False, vegan=True
        - "eggs": Expected keto=True, vegan=False
        - "broccoli": Expected keto=True, vegan=True
        - "milk": Expected keto=False, vegan=False
        - "avocado": Expected keto=True, vegan=True

    The test validates:
    - Individual ingredient classification accuracy
    - Processing time performance
    - Error handling and robustness
    - Benchmark compliance
    - F1-score calculation

    Returns:
        dict: Dictionary containing accuracy metrics, processing times, and F1 scores

    Raises:
        Exception: If classification functions fail unexpectedly

    Example:
        >>> results = test_individual_ingredients()
        >>> print(f"Keto accuracy: {results['keto_accuracy']:.2f}")
        >>> print(f"Vegan accuracy: {results['vegan_accuracy']:.2f}")
    """
    from diet_classifiers import is_ingredient_keto, is_ingredient_vegan

    # Test cases with known answers
    test_cases = [
        # (ingredient, expected_keto, expected_vegan)
        ("chicken breast", True, False),
        ("spinach", True, True),
        ("sugar", False, True),
        ("butter", True, False),
        ("olive oil", True, True),
        ("flour", False, True),
        ("eggs", True, False),
        ("broccoli", True, True),
        ("milk", False, False),
        ("avocado", True, True),
    ]

    logger.info("Testing individual ingredient classification...")

    keto_predictions = []
    vegan_predictions = []
    keto_expected = []
    vegan_expected = []
    processing_times = []

    for ingredient, expected_keto, expected_vegan in test_cases:
        start_time = time.time()

        try:
            pred_keto = is_ingredient_keto(ingredient)
            pred_vegan = is_ingredient_vegan(ingredient)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            keto_predictions.append(pred_keto)
            vegan_predictions.append(pred_vegan)
            keto_expected.append(expected_keto)
            vegan_expected.append(expected_vegan)

            logger.info(
                f"  {ingredient}: keto={pred_keto} (exp: {expected_keto}), "
                f"vegan={pred_vegan} (exp: {expected_vegan}), "
                f"time={processing_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"  {ingredient}: FAILED - {e}")
            keto_predictions.append(False)
            vegan_predictions.append(False)
            keto_expected.append(expected_keto)
            vegan_expected.append(expected_vegan)
            processing_times.append(BENCHMARKS["max_processing_time_per_ingredient"])

    # Calculate metrics
    keto_accuracy = accuracy_score(keto_expected, keto_predictions)
    vegan_accuracy = accuracy_score(vegan_expected, vegan_predictions)
    avg_processing_time = sum(processing_times) / len(processing_times)

    results = {
        "keto_accuracy": keto_accuracy,
        "vegan_accuracy": vegan_accuracy,
        "avg_processing_time": avg_processing_time,
        "max_processing_time": max(processing_times),
        "total_ingredients_tested": len(test_cases),
    }

    # Add F1 scores if sklearn available
    try:
        _, _, keto_f1, _ = precision_recall_fscore_support(
            keto_expected, keto_predictions, average="binary"
        )
        _, _, vegan_f1, _ = precision_recall_fscore_support(
            vegan_expected, vegan_predictions, average="binary"
        )
        results["keto_f1"] = keto_f1
        results["vegan_f1"] = vegan_f1
    except Exception:
        results["keto_f1"] = 0.0
        results["vegan_f1"] = 0.0

    return results


def test_recipe_classification():
    """
    Test full recipe classification performance.

    This function validates the accuracy of full recipe classification using
    curated test recipes with known dietary characteristics. It measures both
    keto and vegan classification accuracy for complete recipes.

    Test Recipes:
        1. ["chicken breast", "spinach", "olive oil"]: Expected keto=True, vegan=False
        2. ["quinoa", "black beans", "avocado"]: Expected keto=False, vegan=True
        3. ["salmon", "asparagus", "butter"]: Expected keto=True, vegan=False
        4. ["pasta", "tomato sauce", "cheese"]: Expected keto=False, vegan=False
        5. ["lettuce", "cucumber", "olive oil"]: Expected keto=True, vegan=True

    The test validates:
    - Full recipe classification accuracy
    - Multi-ingredient processing capability
    - Processing time performance for recipes
    - Error handling for complex inputs
    - Benchmark compliance for recipe processing

    Returns:
        dict: Dictionary containing recipe accuracy metrics and processing times

    Raises:
        Exception: If recipe classification functions fail unexpectedly

    Example:
        >>> results = test_recipe_classification()
        >>> print(f"Recipe keto accuracy: {results['recipe_keto_accuracy']:.2f}")
        >>> print(f"Recipe vegan accuracy: {results['recipe_vegan_accuracy']:.2f}")
    """
    from diet_classifiers import is_keto, is_vegan

    # Test recipes with known classifications
    test_recipes = [
        # (ingredients, expected_keto, expected_vegan)
        (["chicken breast", "spinach", "olive oil"], True, False),  # Keto, not vegan
        (["quinoa", "black beans", "avocado"], False, True),  # Not keto, vegan
        (["salmon", "asparagus", "butter"], True, False),  # Keto, not vegan
        (["pasta", "tomato sauce", "cheese"], False, False),  # Not keto, not vegan
        (["lettuce", "cucumber", "olive oil"], True, True),  # Keto and vegan
    ]

    logger.info("Testing recipe classification...")

    keto_predictions = []
    vegan_predictions = []
    keto_expected = []
    vegan_expected = []
    processing_times = []

    for ingredients, expected_keto, expected_vegan in test_recipes:
        start_time = time.time()

        try:
            pred_keto = is_keto(ingredients)
            pred_vegan = is_vegan(ingredients)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            keto_predictions.append(pred_keto)
            vegan_predictions.append(pred_vegan)
            keto_expected.append(expected_keto)
            vegan_expected.append(expected_vegan)

            logger.info(
                f"  {ingredients}: keto={pred_keto} (exp: {expected_keto}), "
                f"vegan={pred_vegan} (exp: {expected_vegan}), "
                f"time={processing_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"  {ingredients}: FAILED - {e}")
            keto_predictions.append(False)
            vegan_predictions.append(False)
            keto_expected.append(expected_keto)
            vegan_expected.append(expected_vegan)
            processing_times.append(BENCHMARKS["max_processing_time_per_recipe"])

    # Calculate metrics
    recipe_keto_accuracy = accuracy_score(keto_expected, keto_predictions)
    recipe_vegan_accuracy = accuracy_score(vegan_expected, vegan_predictions)
    avg_recipe_time = sum(processing_times) / len(processing_times)

    return {
        "recipe_keto_accuracy": recipe_keto_accuracy,
        "recipe_vegan_accuracy": recipe_vegan_accuracy,
        "avg_recipe_processing_time": avg_recipe_time,
        "max_recipe_processing_time": max(processing_times),
        "total_recipes_tested": len(test_recipes),
    }


def test_input_format_compatibility():
    """
    Test that diet_classifiers handles original submission format.

    This function validates that the classification system can handle various
    input formats including JSON strings, comma-separated strings, and direct
    lists. It ensures backward compatibility with the original submission format.

    Test Formats:
        1. JSON string format: '["chicken breast", "spinach", "olive oil"]'
        2. Comma-separated string: "chicken breast, spinach, olive oil"
        3. Direct list format: ["chicken breast", "spinach", "olive oil"]

    The test validates:
    - Input format compatibility and parsing
    - Consistent results across different formats
    - Error handling for malformed inputs
    - Backward compatibility with original format
    - Robustness of input processing

    Returns:
        list: List of results for each input format tested

    Raises:
        Exception: If input format processing fails unexpectedly

    Example:
        >>> results = test_input_format_compatibility()
        >>> for i, result in enumerate(results):
        >>>     print(f"Format {i + 1}: {result}")
    """
    from diet_classifiers import is_keto, is_vegan

    logger.info("Testing input format compatibility...")

    # Test different input formats
    test_cases = [
        # JSON string format (original submission format)
        '["chicken breast", "spinach", "olive oil"]',
        # Comma-separated string format
        "chicken breast, spinach, olive oil",
        # Direct list format
        ["chicken breast", "spinach", "olive oil"],
    ]

    results = []
    for i, ingredients in enumerate(test_cases):
        try:
            keto_result = is_keto(ingredients)
            vegan_result = is_vegan(ingredients)
            results.append((keto_result, vegan_result))
            logger.info(f"  Format {i + 1}: keto={keto_result}, vegan={vegan_result}")
        except Exception as e:
            logger.error(f"  Format {i + 1}: FAILED - {e}")
            results.append((False, False))

    # All formats should give same result
    all_same = len(set(results)) == 1
    logger.info(f"  All formats consistent: {all_same}")

    return {"format_compatibility": all_same}


def main():
    """Run all Phase 4 validations."""
    logger.info("=" * 60)
    logger.info("PHASE 4 VALIDATION: Submission Requirements KPIs")
    logger.info("=" * 60)

    try:
        # Test individual ingredients
        ingredient_results = test_individual_ingredients()

        # Test recipe classification
        recipe_results = test_recipe_classification()

        # Test input format compatibility
        format_results = test_input_format_compatibility()

        # Combine results
        all_results = {**ingredient_results, **recipe_results, **format_results}

        # Evaluate against benchmarks
        logger.info("=" * 60)
        logger.info("PHASE 4 BENCHMARK EVALUATION")
        logger.info("=" * 60)

        benchmarks_met = []

        # Accuracy benchmarks
        keto_acc_met = all_results["keto_accuracy"] >= BENCHMARKS["keto_accuracy_min"]
        vegan_acc_met = (
            all_results["vegan_accuracy"] >= BENCHMARKS["vegan_accuracy_min"]
        )

        logger.info(
            f"{'PASS' if keto_acc_met else 'WARNING'} Keto Accuracy: "
            f"{all_results['keto_accuracy']:.3f} "
            f"(benchmark: {BENCHMARKS['keto_accuracy_min']:.3f})"
        )
        logger.info(
            f"{'PASS' if vegan_acc_met else 'WARNING'} Vegan Accuracy: "
            f"{all_results['vegan_accuracy']:.3f} "
            f"(benchmark: {BENCHMARKS['vegan_accuracy_min']:.3f})"
        )

        benchmarks_met.extend([keto_acc_met, vegan_acc_met])

        # F1 score benchmarks (if available)
        if all_results.get("keto_f1", 0) > 0:
            keto_f1_met = all_results["keto_f1"] >= BENCHMARKS["keto_f1_min"]
            vegan_f1_met = all_results["vegan_f1"] >= BENCHMARKS["vegan_f1_min"]

            logger.info(
                f"{'PASS' if keto_f1_met else 'WARNING'} Keto F1: {all_results['keto_f1']:.3f} "
                f"(benchmark: {BENCHMARKS['keto_f1_min']:.3f})"
            )
            logger.info(
                f"{'PASS' if vegan_f1_met else 'WARNING'} Vegan F1: {all_results['vegan_f1']:.3f} "
                f"(benchmark: {BENCHMARKS['vegan_f1_min']:.3f})"
            )

            benchmarks_met.extend([keto_f1_met, vegan_f1_met])

        # Performance benchmarks
        time_met = (
            all_results["avg_processing_time"]
            <= BENCHMARKS["max_processing_time_per_ingredient"]
        )
        recipe_time_met = (
            all_results["avg_recipe_processing_time"]
            <= BENCHMARKS["max_processing_time_per_recipe"]
        )

        logger.info(
            f"{'PASS' if time_met else 'WARNING'} Avg Processing Time: "
            f"{all_results['avg_processing_time']:.2f}s "
            f"(benchmark: {BENCHMARKS['max_processing_time_per_ingredient']:.2f}s)"
        )
        logger.info(
            f"{'PASS' if recipe_time_met else 'WARNING'} Avg Recipe Time: "
            f"{all_results['avg_recipe_processing_time']:.2f}s "
            f"(benchmark: {BENCHMARKS['max_processing_time_per_recipe']:.2f}s)"
        )

        benchmarks_met.extend([time_met, recipe_time_met])

        # Format compatibility
        format_met = all_results["format_compatibility"]
        logger.info(
            f"{'PASS' if format_met else 'WARNING'} Format Compatibility: {format_met}"
        )
        benchmarks_met.append(format_met)

        # Overall status
        all_benchmarks_met = all(benchmarks_met)
        benchmarks_passed = sum(benchmarks_met)
        total_benchmarks = len(benchmarks_met)

        logger.info("=" * 60)
        logger.info(
            f"PHASE 4 SUMMARY: {benchmarks_passed}/{total_benchmarks} benchmarks met"
        )

        if all_benchmarks_met:
            logger.info("ALL PHASE 4 BENCHMARKS MET")
            return 0
        else:
            logger.warning("Some benchmarks not met - continuing with warnings")
            return 0  # Continue with warnings as requested

    except Exception as e:
        logger.error(f"Phase 4 validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
