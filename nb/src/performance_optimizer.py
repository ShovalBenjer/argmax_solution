#!/usr/bin/env python3
"""
Performance Optimization Module for Diet Classification System

This module implements comprehensive performance optimizations:
1. Redis caching for LLM responses and classification results
2. Parallel processing for batch operations
3. GPU detection and utilization
4. Optimized prompts for faster LLM responses
5. Timeout handling and retry logic
6. Model switching based on performance requirements

Key Features:
- Intelligent caching with TTL and cache warming
- Async parallel processing with configurable concurrency
- GPU detection and CUDA optimization
- Prompt optimization for faster responses
- Comprehensive timeout and retry handling
- Performance monitoring and metrics

Example:
    >>> from performance_optimizer import PerformanceOptimizer
    >>> optimizer = PerformanceOptimizer()
    >>> results = await optimizer.process_ingredients_parallel(['chicken', 'beef', 'tofu'])
    >>> print(f"Processed {len(results)} ingredients in parallel")
"""

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Union

try:
    import torch
except ImportError:
    torch = None

from llm_client import LLMClient
from utils.cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Comprehensive performance optimization system for diet classification.

    This class implements multiple optimization strategies:
    - Redis caching for repeated queries
    - Parallel processing for batch operations
    - GPU acceleration when available
    - Optimized prompts for faster LLM responses
    - Timeout handling and retry logic
    - Model switching based on performance needs
    """

    def __init__(self):
        """Initialize the performance optimizer with all optimization features."""
        self.cache_manager = get_cache_manager()
        self.llm_client = LLMClient()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.gpu_available = self._detect_gpu()
        self.fast_model = "qwen/qwen3-0.6b-gguf:q8_0"
        self.accurate_model = "arctic-text2sql:latest"
        self.timeout_seconds = 30
        self.max_retries = 3

        logger.info(f"Performance optimizer initialized - GPU: {self.gpu_available}")

    def _detect_gpu(self) -> bool:
        """Detect if GPU is available and configure for optimal performance."""
        if torch is None:
            logger.info("PyTorch not installed, skipping GPU detection.")
            return False
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU detected: {gpu_name} (Count: {gpu_count})")

                # Set CUDA optimization flags
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

                return True
            else:
                logger.info("No GPU detected, using CPU")
                return False
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return False

    def _get_cache_key(self, data: Union[str, Dict], prefix: str = "") -> str:
        """Generate a cache key for the given data."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        key = f"{prefix}:{hashlib.md5(data_str.encode(), usedforsecurity=False).hexdigest()}"
        return key

    def _optimize_prompt(self, original_prompt: str, use_fast_mode: bool = True) -> str:
        """
        Optimize prompts for faster LLM responses.

        Args:
            original_prompt: The original prompt to optimize
            use_fast_mode: Whether to use fast mode optimizations

        Returns:
            str: Optimized prompt for faster response
        """
        if use_fast_mode:
            # Fast mode optimizations
            optimized = original_prompt.replace(
                "You are an expert at converting user questions into structured JSON queries.",
                "Generate JSON query:",
            )
            optimized = optimized.replace(
                (
                    "Based on the user's question, generate a JSON object that matches the "
                    "specified "
                    "format."
                ),
                "JSON:",
            )
            optimized = optimized.replace(
                "Respond with ONLY the generated JSON object, nothing else.",
                "JSON only:",
            )

            # Remove verbose instructions
            lines = optimized.split("\n")
            filtered_lines = []
            for line in lines:
                if any(
                    keyword in line.lower()
                    for keyword in ["json", "schema", "table", "operation"]
                ):
                    filtered_lines.append(line)
                elif line.strip() and not line.startswith("   "):
                    filtered_lines.append(line)

            optimized = "\n".join(filtered_lines)
        else:
            optimized = original_prompt

        return optimized

    async def _query_with_timeout_and_retry(
        self, model: str, prompt: str, use_fast_mode: bool = True, timeout: int = None
    ) -> Dict[str, Any]:
        """
        Execute LLM query with timeout, retry logic, and caching.

        Args:
            model: Model name to use
            prompt: Prompt to send
            use_fast_mode: Whether to use fast mode
            timeout: Timeout in seconds (defaults to self.timeout_seconds)

        Returns:
            Dict: Query result or error information
        """
        timeout = timeout or self.timeout_seconds
        cache_key = self._get_cache_key(f"{model}:{prompt}", "llm_query")

        # Check cache first
        cached_result = (
            self.cache_manager._redis_client.get(cache_key)
            if self.cache_manager.is_available()
            else None
        )
        if cached_result:
            logger.info(f"Cache hit for LLM query: {prompt[:50]}...")
            return json.loads(cached_result)

        # Optimize prompt for faster response
        optimized_prompt = self._optimize_prompt(prompt, use_fast_mode)

        for attempt in range(self.max_retries):
            try:
                # Use asyncio.wait_for for timeout handling
                result = await asyncio.wait_for(
                    self.llm_client.query_async(model, optimized_prompt, as_json=True),
                    timeout=timeout,
                )

                # Cache successful result
                if "error" not in result and self.cache_manager.is_available():
                    self.cache_manager._redis_client.setex(
                        cache_key, 3600, json.dumps(result)  # 1 hour TTL
                    )

                return result

            except asyncio.TimeoutError:
                logger.warning(
                    f"LLM query timeout (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt == self.max_retries - 1:
                    return {"error": f"Query timeout after {timeout}s"}

            except Exception as e:
                logger.error(
                    f"LLM query error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    return {"error": str(e)}

                # Exponential backoff
                await asyncio.sleep(2**attempt)

    async def process_ingredients_parallel(
        self,
        ingredients: List[str],
        use_fast_mode: bool = True,
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple ingredients in parallel with performance optimizations.

        Args:
            ingredients: List of ingredients to process
            use_fast_mode: Whether to use fast mode for speed
            max_concurrent: Maximum concurrent operations

        Returns:
            List: Processing results for all ingredients
        """
        logger.info(
            (
                f"Processing {len(ingredients)} ingredients in parallel (max_concurrent: "
                f"{max_concurrent})"
            )
        )

        # Choose model based on performance requirements
        model = self.fast_model if use_fast_mode else self.accurate_model

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_ingredient(ingredient: str) -> Dict[str, Any]:
            async with semaphore:
                start_time = time.time()

                # Create optimized prompt
                prompt = (
                    f"""Generate JSON query for: Search nutritional and vegan """
                    f"""information for '{ingredient}'.

                DATABASE SCHEMA:
                - nutrition_facts: name, calories, protein_g, total_fat_g, carbohydrate_g
                - vegan_ontology: term, is_explicitly_non_vegan, description

                JSON: """
                    f"""{{"operation": "search", "table": "nutrition_facts", """
                    f""""filters": [{{"field": "name", "operator": "LIKE", "value": """
                    f""""%{ingredient}%"}}]}}"""
                )

                result = await self._query_with_timeout_and_retry(
                    model, prompt, use_fast_mode, timeout=15 if use_fast_mode else 45
                )

                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                result["ingredient"] = ingredient

                logger.info(f"Processed '{ingredient}' in {processing_time:.2f}s")
                return result

        # Process all ingredients in parallel
        tasks = [process_single_ingredient(ingredient) for ingredient in ingredients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing '{ingredients[i]}': {result}")
                processed_results.append(
                    {
                        "error": str(result),
                        "ingredient": ingredients[i],
                        "processing_time": 0,
                    }
                )
            else:
                processed_results.append(result)

        total_time = sum(r.get("processing_time", 0) for r in processed_results)
        avg_time = total_time / len(processed_results) if processed_results else 0

        logger.info(
            f"Parallel processing completed: {len(processed_results)} ingredients, "
            f"avg time: {avg_time:.2f}s, total time: {total_time:.2f}s"
        )

        return processed_results

    async def batch_classify_diets(
        self, recipes: List[Dict[str, Any]], use_fast_mode: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch classify multiple recipes with parallel processing.

        Args:
            recipes: List of recipes to classify
            use_fast_mode: Whether to use fast mode

        Returns:
            List: Classification results for all recipes
        """
        logger.info(f"Batch classifying {len(recipes)} recipes")

        # Extract all unique ingredients
        all_ingredients = set()
        for recipe in recipes:
            ingredients = recipe.get("ingredients", [])
            all_ingredients.update(ingredients)

        # Process all ingredients in parallel
        ingredient_results = await self.process_ingredients_parallel(
            list(all_ingredients), use_fast_mode
        )

        # Create ingredient lookup
        ingredient_lookup = {
            result["ingredient"]: result
            for result in ingredient_results
            if "error" not in result
        }

        # Classify each recipe
        recipe_results = []
        for recipe in recipes:
            recipe_id = recipe.get("id", "unknown")
            ingredients = recipe.get("ingredients", [])

            # Look up ingredient results
            recipe_ingredients = []
            for ingredient in ingredients:
                if ingredient in ingredient_lookup:
                    recipe_ingredients.append(ingredient_lookup[ingredient])

            # Simple diet classification logic
            classification = self._classify_recipe_diets(recipe_ingredients)

            result = {
                "recipe_id": recipe_id,
                "ingredients": ingredients,
                "classification": classification,
                "ingredient_details": recipe_ingredients,
            }

            recipe_results.append(result)

        logger.info(f"Batch classification completed: {len(recipe_results)} recipes")
        return recipe_results

    def _classify_recipe_diets(
        self, ingredient_results: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        Classify recipe diets based on ingredient analysis.

        Args:
            ingredient_results: Results from ingredient processing

        Returns:
            Dict: Diet classification results
        """
        # Simple classification logic (can be enhanced)
        has_animal_products = False
        total_carbs = 0
        total_protein = 0
        total_fat = 0

        for result in ingredient_results:
            if "error" in result:
                continue

            # Check for animal products
            if result.get("is_explicitly_non_vegan", False):
                has_animal_products = True

            # Sum nutritional values
            total_carbs += result.get("carbohydrate_g", 0)
            total_protein += result.get("protein_g", 0)
            total_fat += result.get("total_fat_g", 0)

        # Diet classification
        is_vegan = not has_animal_products
        is_keto = total_carbs < 20 and total_fat > total_protein * 0.5

        return {
            "vegan": is_vegan,
            "keto": is_keto,
            "low_carb": total_carbs < 50,
            "high_protein": total_protein > 20,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and system information."""
        stats = {
            "gpu_available": self.gpu_available,
            "gpu_info": {
                "name": (
                    torch.cuda.get_device_name(0)
                    if self.gpu_available and torch
                    else None
                ),
                "memory": (
                    torch.cuda.get_device_properties(0).total_memory
                    if self.gpu_available and torch
                    else None
                ),
            },
            "cache_stats": (
                self.cache_manager.get_stats()
                if self.cache_manager.is_available()
                else {}
            ),
            "models": {"fast": self.fast_model, "accurate": self.accurate_model},
            "timeout_settings": {
                "fast_mode": 15,
                "accurate_mode": 45,
                "max_retries": self.max_retries,
            },
        }

        return stats


# Global instance for easy access
performance_optimizer = PerformanceOptimizer()


async def main():
    """Example usage and testing of the performance optimizer."""
    optimizer = PerformanceOptimizer()

    # Test parallel processing
    ingredients = ["chicken breast", "olive oil", "spinach", "eggs", "tofu"]

    print("Testing parallel ingredient processing...")
    results = await optimizer.process_ingredients_parallel(
        ingredients, use_fast_mode=True
    )

    for result in results:
        print(f"{result['ingredient']}: {result.get('processing_time', 0):.2f}s")

    # Test batch classification
    recipes = [
        {"id": "recipe1", "ingredients": ["chicken breast", "olive oil"]},
        {"id": "recipe2", "ingredients": ["tofu", "spinach"]},
    ]

    print("\nTesting batch classification...")
    classifications = await optimizer.batch_classify_diets(recipes, use_fast_mode=True)

    for classification in classifications:
        print(f"{classification['recipe_id']}: {classification['classification']}")

    # Show performance stats
    print("\nPerformance Statistics:")
    stats = optimizer.get_performance_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
