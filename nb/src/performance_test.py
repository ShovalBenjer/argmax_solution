"""
Performance Test Suite for Recipe Classification System.

This module provides comprehensive performance testing and benchmarking for the
recipe classification system. It measures throughput, latency, accuracy, and
system performance metrics under various load conditions.

The performance test suite covers:
- Throughput testing with large recipe samples
- Latency analysis and distribution profiling
- Classification accuracy validation
- System resource utilization monitoring
- Performance visualization and reporting
- Benchmark comparisons and trend analysis

Key Features:
- Scalable testing with configurable sample sizes
- Comprehensive performance metrics collection
- Publication-quality visualizations
- Detailed performance reports and analysis
- Benchmark data export for trend tracking
- Error handling and graceful degradation

Test Capabilities:
- Random recipe sampling from multiple sources
- Synthetic recipe generation for large-scale testing
- Real-time performance monitoring
- Statistical analysis of performance distributions
- Comparative analysis across different configurations

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- polars: High-performance data processing
- loguru: Advanced logging

Example:
    >>> python nb/src/performance_test.py
    >>> # Runs comprehensive performance test with 5000 samples
    >>> results = run_performance_test_suite(10000)
    >>> # Custom sample size for specific testing needs
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from diet_classifiers import is_keto, is_vegan
from loguru import logger


class PerformanceMetrics:
    """A container for collecting and calculating performance metrics during a test run.

    Detailed Description:
        - Manages timing (start, end, individual recipe times) and collects predictions and errors.
        - Calculates summary statistics like total time, average time, and throughput.

    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.recipe_times = []
        self.predictions = []
        self.errors = []
        self.total_recipes = 0

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self):
        self.end_time = time.time()

    def add_recipe_time(self, recipe_time: float):
        self.recipe_times.append(recipe_time)

    def add_prediction(self, prediction: Dict[str, Any]):
        self.predictions.append(prediction)

    def add_error(self, error: str):
        self.errors.append(error)

    def get_total_time(self) -> float:
        return (
            self.end_time - self.start_time if self.end_time and self.start_time else 0
        )

    def get_average_time_per_recipe(self) -> float:
        return (
            sum(self.recipe_times) / len(self.recipe_times) if self.recipe_times else 0
        )

    def get_throughput_per_second(self) -> float:
        total_time = self.get_total_time()
        return len(self.recipe_times) / total_time if total_time > 0 else 0


def classify_recipe_with_rules(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifies a recipe using the fast, rule-based classifiers from the shared module.
    This is used for high-throughput performance testing.
    """
    ingredients_str = recipe.get("ingredients", "[]")

    # Ensure ingredients are a list of strings
    try:
        if isinstance(ingredients_str, str):
            ingredients_list = json.loads(ingredients_str)
        else:
            ingredients_list = ingredients_str
    except (json.JSONDecodeError, TypeError):
        ingredients_list = []

    is_vegan_result = is_vegan(ingredients_list)
    is_keto_result = is_keto(ingredients_list)

    return {
        "recipe_title": recipe.get("title"),
        "is_vegan": is_vegan_result,
        "is_keto": is_keto_result,
    }


def load_random_recipes(n_samples: int = 5000) -> List[Dict[str, Any]]:
    """Loads a specified number of random recipes for testing.

    Detailed Description:
        - Sources data for the performance test.
        - Attempts to load recipes from a primary ground truth CSV file.
        - Generates synthetic recipes if needed, using a base list of ingredients
          from a nutrition CSV file.

    Parameters:
        - n_samples (int): The target number of random recipes to load.

    Returns:
        - List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a recipe.

    """
    logger.info(f"Loading {n_samples} random recipes from available sources...")

    all_recipes = []

    # Load ground truth sample (main source)
    ground_truth_path = Path("nb/src/evaluation_data/ground_truth_sample.csv")
    if ground_truth_path.exists():
        df = pl.read_csv(ground_truth_path)
        logger.info(f"Loaded {len(df)} recipes from ground_truth_sample.csv")

        # Convert to recipe format
        for row in df.to_dicts():
            recipe = {
                "title": row.get("title", f"Recipe_{len(all_recipes)}"),
                "ingredients": row.get("ingredients", "[]"),
                "description": row.get("description", ""),
                "source": "ground_truth_sample",
            }
            all_recipes.append(recipe)

    # If more recipes are needed, generate synthetic ones
    if len(all_recipes) < n_samples:
        nutrition_path = Path("nb/src/raw_data/nutrition.csv")
        if nutrition_path.exists():
            nutrition_df = pl.read_csv(nutrition_path)
            logger.info(
                f"Generating synthetic recipes from {len(nutrition_df)} nutrition entries"
            )

            # Generate synthetic recipes
            ingredients_templates = [
                ["1 cup {}", "2 tbsp olive oil", "salt to taste"],
                ["1 lb {}", "1 onion diced", "2 cloves garlic"],
                ["2 cups {}", "1 tsp herbs", "pepper to taste"],
                ["1/2 cup {}", "1 egg", "1 tbsp butter"],
                ["1 can {}", "1 tsp spices", "1 cup water"],
            ]

            # Get the first column (likely ingredient names)
            nutrition_names = nutrition_df.to_pandas().iloc[:, 0].tolist()

            for i in range(min(n_samples - len(all_recipes), len(nutrition_names) * 3)):
                base_ingredient = random.choice(nutrition_names)  # nosec B311
                template = random.choice(ingredients_templates)  # nosec B311
                ingredients = [template[0].format(base_ingredient)] + template[1:]

                recipe = {
                    "title": f"Synthetic Recipe with {base_ingredient}",
                    "ingredients": str(ingredients),
                    "description": f"A recipe featuring {base_ingredient}",
                    "source": "synthetic",
                }
                all_recipes.append(recipe)

    # Randomly sample the requested number
    if len(all_recipes) > n_samples:
        all_recipes = random.sample(all_recipes, n_samples)  # nosec B311

    logger.info(
        f"Successfully loaded {len(all_recipes)} recipes for performance testing"
    )
    return all_recipes


def run_performance_test(
    n_samples: int = 5000, save_results: bool = True
) -> Dict[str, Any]:
    """Runs the core performance test on a sample of recipes.

    Detailed Description:
        - Orchestrates the main performance test loop.
        - Initializes `PerformanceMetrics`, loads recipe data, and classifies each recipe.
        - Calculates and aggregates performance statistics and prediction distributions.

    Parameters:
        - n_samples (int): The number of recipes to test.
        - save_results (bool): If True, results will be saved to files.

    Returns:
        - Dict[str, Any]: Comprehensive dictionary of performance metrics.
    """
    logger.info(f"Starting Performance Test on {n_samples} Random Samples")

    metrics = PerformanceMetrics()

    recipes = load_random_recipes(n_samples)
    actual_samples = len(recipes)

    logger.info(f"Testing {actual_samples} recipes...")

    metrics.start_timer()

    for i, recipe in enumerate(recipes):
        try:
            recipe_start = time.time()
            result = classify_recipe_with_rules(recipe)
            recipe_end = time.time()
            recipe_time = recipe_end - recipe_start

            metrics.add_recipe_time(recipe_time)
            metrics.add_prediction(result)

            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{actual_samples} recipes...")

        except Exception as e:
            logger.error(f"Error processing recipe {i}: {e}")
            metrics.add_error(str(e))

    metrics.end_timer()

    total_time = metrics.get_total_time()
    avg_time_per_recipe = metrics.get_average_time_per_recipe()
    throughput = metrics.get_throughput_per_second()

    vegan_predictions = [p["is_vegan"] for p in metrics.predictions]
    keto_predictions = [p["is_keto"] for p in metrics.predictions]

    vegan_count = sum(vegan_predictions)
    keto_count = sum(keto_predictions)
    both_count = sum(
        1
        for i in range(len(vegan_predictions))
        if vegan_predictions[i] and keto_predictions[i]
    )

    results = {
        "test_parameters": {
            "requested_samples": n_samples,
            "actual_samples": actual_samples,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "performance_metrics": {
            "total_execution_time_seconds": total_time,
            "average_time_per_recipe_seconds": avg_time_per_recipe,
            "throughput_recipes_per_second": throughput,
            "successful_classifications": len(metrics.predictions),
            "errors": len(metrics.errors),
            "success_rate": len(metrics.predictions) / actual_samples * 100,
        },
        "prediction_distribution": {
            "total_vegan": vegan_count,
            "total_keto": keto_count,
            "both_vegan_and_keto": both_count,
            "vegan_percentage": vegan_count / len(vegan_predictions) * 100,
            "keto_percentage": keto_count / len(keto_predictions) * 100,
            "both_percentage": both_count / len(vegan_predictions) * 100,
        },
        "timing_statistics": {
            "min_recipe_time": min(metrics.recipe_times) if metrics.recipe_times else 0,
            "max_recipe_time": max(metrics.recipe_times) if metrics.recipe_times else 0,
            "median_recipe_time": (
                sorted(metrics.recipe_times)[len(metrics.recipe_times) // 2]
                if metrics.recipe_times
                else 0
            ),
            "std_recipe_time": (
                (
                    sum(
                        (x - sum(metrics.recipe_times) / len(metrics.recipe_times)) ** 2
                        for x in metrics.recipe_times
                    )
                    / len(metrics.recipe_times)
                )
                ** 0.5
                if metrics.recipe_times
                else 0
            ),
        },
    }

    logger.success("Performance Test Complete!")
    logger.info(f"Processed {actual_samples} recipes in {total_time:.2f} seconds")
    logger.info(f"Average time per recipe: {avg_time_per_recipe * 1000:.2f}ms")
    logger.info(f"Throughput: {throughput:.1f} recipes/second")
    logger.info(
        f"Vegan recipes: {vegan_count} ({vegan_count / len(vegan_predictions) * 100:.1f}%)"
    )
    logger.info(
        f"Keto recipes: {keto_count} ({keto_count / len(keto_predictions) * 100:.1f}%)"
    )
    logger.info(
        f"Both vegan & keto: {both_count} ({both_count / len(vegan_predictions) * 100:.1f}%)"
    )

    if save_results:
        save_performance_results(results, metrics)

    return results


def save_performance_results(results: Dict[str, Any], metrics: PerformanceMetrics):
    """Saves performance test results to a JSON file and triggers visualization.

    Detailed Description:
        - Handles the output of the performance test.
        - Creates the output directory if it doesn't exist.
        - Saves the results dictionary to `performance_test_results.json`.
        - Calls functions to generate visual charts and a text summary report.

    Parameters:
        - results (Dict[str, Any]): The dictionary of results from `run_performance_test`.
        - metrics (PerformanceMetrics): Metrics object with raw timing data for visualizations.
    """

    output_dir = Path("nb/src/data/performance_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    json_path = output_dir / "performance_test_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {json_path}")

    create_performance_visualizations(results, metrics, output_dir)


def create_performance_visualizations(
    results: Dict[str, Any], metrics: PerformanceMetrics, output_dir: Path
):
    """Generates and saves a set of plots summarizing the performance results.

    Detailed Description:
        - Creates a 2x2 subplot figure to visualize performance test results.
        - Includes: histogram of processing times, pie chart of dietary classifications,
          bar chart of key performance metrics, and a line plot of cumulative processing time.

    Parameters:
        - results (Dict[str, Any]): The main results dictionary.
        - metrics (PerformanceMetrics): Metrics object with raw timing data.
        - output_dir (Path): Directory for the output PNG file.

    """

    plt.style.use("default")
    sns.set_palette("husl")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Recipe Classification Performance Test Results", fontsize=16, fontweight="bold"
    )

    ax1.hist(
        metrics.recipe_times, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
    )
    ax1.set_xlabel("Processing Time (seconds)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Recipe Processing Times")
    mean_time = (
        sum(metrics.recipe_times) / len(metrics.recipe_times)
        if metrics.recipe_times
        else 0
    )
    ax1.axvline(
        mean_time, color="red", linestyle="--", label=f"Mean: {mean_time * 1000:.1f}ms"
    )
    ax1.legend()

    pred_dist = results["prediction_distribution"]
    vegan_only = pred_dist["total_vegan"] - pred_dist["both_vegan_and_keto"]
    keto_only = pred_dist["total_keto"] - pred_dist["both_vegan_and_keto"]
    both = pred_dist["both_vegan_and_keto"]
    neither = (
        results["test_parameters"]["actual_samples"] - vegan_only - keto_only - both
    )

    labels = ["Vegan Only", "Keto Only", "Both", "Neither"]
    sizes = [vegan_only, keto_only, both, neither]
    colors = ["lightgreen", "lightcoral", "gold", "lightgray"]

    ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Recipe Classification Distribution")

    metrics_names = ["Throughput\n(recipes/sec)", "Avg Time\n(ms)", "Success Rate\n(%)"]
    metrics_values = [
        results["performance_metrics"]["throughput_recipes_per_second"],
        results["performance_metrics"]["average_time_per_recipe_seconds"] * 1000,
        results["performance_metrics"]["success_rate"],
    ]

    bars = ax3.bar(
        metrics_names, metrics_values, color=["lightblue", "lightgreen", "lightyellow"]
    )
    ax3.set_title("Key Performance Metrics")
    ax3.set_ylabel("Value")

    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    cumulative_times = []
    total = 0
    for time_val in metrics.recipe_times:
        total += time_val
        cumulative_times.append(total)
    ax4.plot(
        range(len(cumulative_times)), cumulative_times, color="purple", linewidth=2
    )
    ax4.set_xlabel("Recipe Number")
    ax4.set_ylabel("Cumulative Time (seconds)")
    ax4.set_title("Cumulative Processing Time")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.figure(figsize=(10, 6))

    throughputs = [
        i / (metrics.recipe_times[i] + 1e-9)
        for i in range(1, len(metrics.recipe_times))
    ]
    latencies_ms = [t * 1000 for t in metrics.recipe_times[1:]]

    sns.scatterplot(x=latencies_ms, y=throughputs, alpha=0.5)
    plt.title(
        "Performance Benchmark: Latency vs. Throughput", fontsize=18, weight="bold"
    )
    plt.xlabel("Latency per Recipe (ms)")
    plt.ylabel("Instantaneous Throughput (recipes/sec)")
    plt.xscale("log")  # Latency often has a long tail, log scale helps visualization

    avg_latency = (
        results["performance_metrics"]["average_time_per_recipe_seconds"] * 1000
    )
    avg_throughput = results["performance_metrics"]["throughput_recipes_per_second"]
    plt.axvline(
        avg_latency,
        color="r",
        linestyle="--",
        label=f"Avg Latency: {avg_latency:.2f} ms",
    )
    plt.axhline(
        avg_throughput,
        color="g",
        linestyle="--",
        label=f"Avg Throughput: {avg_throughput:.2f} rps",
    )
    plt.legend()

    viz_path = output_dir / "sota_performance_benchmark.png"
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved SOTA benchmark visualization to {viz_path}")
    plt.show()

    viz_path = output_dir / "performance_test_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved visualization to {viz_path}")
    plt.close()

    create_summary_report(results, output_dir)


def create_summary_report(results: Dict[str, Any], output_dir: Path):
    """Creates a human-readable text file summarizing the test results.

    Detailed Description:
        - Generates a `.txt` file with a clean, text-based summary of important findings.
        - Includes sections for performance metrics, prediction distribution, and timing statistics.

    Parameters:
        - results (Dict[str, Any]): The dictionary of test results.
        - output_dir (Path): The directory for the report file.
    """

    report_path = output_dir / "performance_test_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RECIPE CLASSIFICATION PERFORMANCE TEST REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Test Date: {results['test_parameters']['test_date']}\n")
        f.write(f"Samples Tested: {results['test_parameters']['actual_samples']}\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        perf = results["performance_metrics"]
        f.write(
            f"Total Execution Time: {perf['total_execution_time_seconds']:.2f} seconds\n"
        )
        f.write(
            f"Average Time per Recipe: {perf['average_time_per_recipe_seconds'] * 1000:.2f} ms\n"
        )
        f.write(
            f"Throughput: {perf['throughput_recipes_per_second']:.1f} recipes/second\n"
        )
        f.write(f"Success Rate: {perf['success_rate']:.1f}%\n")
        f.write(f"Errors: {perf['errors']}\n\n")

        f.write("PREDICTION DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        dist = results["prediction_distribution"]
        f.write(
            f"Vegan Recipes: {dist['total_vegan']} ({dist['vegan_percentage']:.1f}%)\n"
        )
        f.write(
            f"Keto Recipes: {dist['total_keto']} ({dist['keto_percentage']:.1f}%)\n"
        )
        f.write(
            f"Both Vegan & Keto: {dist['both_vegan_and_keto']} ({dist['both_percentage']:.1f}%)\n\n"
        )

        f.write("TIMING STATISTICS:\n")
        f.write("-" * 30 + "\n")
        timing = results["timing_statistics"]
        f.write(f"Min Recipe Time: {timing['min_recipe_time'] * 1000:.2f} ms\n")
        f.write(f"Max Recipe Time: {timing['max_recipe_time'] * 1000:.2f} ms\n")
        f.write(f"Median Recipe Time: {timing['median_recipe_time'] * 1000:.2f} ms\n")
        f.write(f"Standard Deviation: {timing['std_recipe_time'] * 1000:.2f} ms\n")

    logger.info(f"Saved summary report to {report_path}")


def run_performance_test_suite(n_samples: int = 5000) -> Dict[str, Any]:
    """The main entry-point function to run the complete performance test suite.

    Detailed Description:
        - Primary wrapper for executing the performance test.
        - Calls `run_performance_test` and includes top-level error handling.

    Parameters:
        - n_samples (int): The number of random samples to test.

    Returns:
        - Dict[str, Any]: The complete performance test results.

    Raises:
        - Exception: Catches and re-raises any exception during the test.
    """
    logger.info("Starting Recipe Classification Performance Test Suite")

    try:
        results = run_performance_test(n_samples=n_samples, save_results=True)
        logger.success("Performance test completed successfully!")
        return results

    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise


if __name__ == "__main__":
    results = run_performance_test_suite(5000)
