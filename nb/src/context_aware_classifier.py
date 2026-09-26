"""
Context-Aware Diet Classifier - SOTA Semantic Architecture

This module implements a state-of-the-art context-aware diet classification system
that combines multiple advanced techniques for accurate ingredient and recipe
classification. The system uses a sophisticated pipeline that leverages semantic
search, fuzzy matching, and machine learning for robust dietary classification.

PIPELINE: Recipe Ingredient → ingredient-parser → Arctic (Text2SQL) →
Safe Query Engine → Database → Qwen (Classifier)
"""

import asyncio
import json
from typing import Any, Dict, List

import polars as pl
from config import app_config
from database import db_manager
from function_calling_handler import FunctionCallingHandler
from llm_client import llm_client
from loguru import logger  # Use loguru for enhanced logging
from query_engine import translate_json_to_sql
from unified_ingredient_parser import parse_ingredient_simple
from utils.cache_manager import get_cache_manager

# Configure professional logging - using loguru directly
# logger = logging.getLogger(__name__)


def execute_sql_query(sql: str, params: List[Any]) -> str:
    """
    Executes a SQL query with parameters and returns the formatted results.
    """
    logger.info(f"Attempting to execute SQL query: {sql} with params: {params}")
    try:
        with db_manager.get_sqlite_connection() as conn:
            df = pl.read_database(
                query=sql, connection=conn, execute_options={"parameters": params}
            )
            if df.is_empty():
                logger.warning(
                    f"SQL query executed successfully but returned no data: {sql}"
                )
                return "No matching data found in the database."
            results = json.dumps(df.to_dicts(), indent=2)
            logger.success(f"SQL query executed successfully, returned {len(df)} rows.")
            return results
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return f"Database query error: {e}"


class SOTASemanticClassifier:
    """
    State-of-the-Art semantic diet classifier using the complete dual-LLM pipeline.
    """

    def __init__(self, fast_mode: bool = False):
        """
        Initialize the SOTA semantic classifier with all required components.

        Args:
            fast_mode: If True, bypasses Arctic and uses only fuzzy matching for speed
        """
        self.llm_client = llm_client
        self.function_handler = (
            FunctionCallingHandler()
        )  # Assumed to use Arctic for Text2SQL
        self.cache_manager = get_cache_manager()
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
        }
        self.fast_mode = fast_mode
        logger.info(
            f"SOTASemanticClassifier initialized with fast_mode={self.fast_mode}"
        )

    async def _get_semantic_context(self, clean_ingredient: str, diet: str) -> str:
        """
        Uses Arctic to generate a JSON query, which is then safely converted to SQL.
        """
        logger.info(
            f"Retrieving semantic context for '{clean_ingredient}' for {diet} diet."
        )
        # If in fast mode, skip Arctic entirely and use fuzzy matching
        if self.fast_mode:
            logger.info(
                f"Fast mode enabled: Skipping Arctic for {diet} context on '{clean_ingredient}'"
            )
            return await self._get_fuzzy_fallback_context(clean_ingredient, diet)

        if diet == "keto":
            table = "nutrition_facts"
            schema = (
                "nutrition_facts(name TEXT, carbohydrate_g REAL, fiber_g REAL, "
                "protein_g REAL, total_fat_g REAL)"
            )
            prompt = (
                f"Generate a SQL query to find nutrition data for '{clean_ingredient}' "
                f"in the '{table}' table. Prioritize raw or fresh forms. Schema: {schema}"
            )
        elif diet == "vegan":
            table = "vegan_ontology"
            schema = (
                "vegan_ontology(term TEXT, aliases TEXT, is_explicitly_non_vegan "
                "INTEGER, description TEXT)"
            )
            prompt = (
                f"Generate a SQL query to find vegan information for '{clean_ingredient}' "
                f"in the '{table}' table. Check both term and aliases. Schema: {schema}"
            )
        else:
            logger.error(f"Invalid diet type '{diet}' for semantic context retrieval.")
            return "Invalid diet type for context."

        logger.debug(
            f"Arctic prompt for {diet} context: {prompt[:100]}..."
        )  # Log truncated prompt
        # LLM 1 (Arctic) generates a structured JSON query with shorter timeout
        try:
            json_query = await asyncio.wait_for(
                self.function_handler.generate_json_query(prompt),
                timeout=app_config.ARCTIC_TIMEOUT,  # Use configured timeout
            )
            if "error" in json_query:
                logger.warning(
                    (
                        f"Arctic returned an error for {diet} query on '{clean_ingredient}': "
                        f"{json_query['error']}, falling back to fuzzy."
                    )
                )
                return await self._get_fuzzy_fallback_context(clean_ingredient, diet)
            logger.debug(
                f"Arctic successfully generated JSON query for {clean_ingredient}: {json_query}"
            )
        except asyncio.TimeoutError:
            logger.warning(
                (
                    f"Arctic timeout ({app_config.ARCTIC_TIMEOUT}s) for {diet} query on "
                    f"'{clean_ingredient}', using fuzzy fallback."
                )
            )
            return await self._get_fuzzy_fallback_context(clean_ingredient, diet)
        except Exception as e:
            logger.error(
                (
                    f"Unexpected error during Arctic query for {clean_ingredient}: {e}, "
                    f"falling back to fuzzy."
                )
            )
            return await self._get_fuzzy_fallback_context(clean_ingredient, diet)

        try:
            # The safe query engine translates JSON to parameterized SQL
            sql, params = translate_json_to_sql(json_query)
            logger.debug(
                f"Translated JSON to SQL for {diet} context: SQL='{sql}', Params={params}"
            )
            return execute_sql_query(sql, params)
        except ValueError as e:
            logger.error(
                (
                    f"Failed to translate semantic {diet} query to SQL for '{clean_ingredient}': "
                    f"{e}, using fuzzy fallback."
                )
            )
            return await self._get_fuzzy_fallback_context(clean_ingredient, diet)
        except Exception as e:
            logger.error(
                (
                    f"Error during SQL translation/execution for {clean_ingredient}: {e}, "
                    f"using fuzzy fallback."
                )
            )
            return await self._get_fuzzy_fallback_context(clean_ingredient, diet)

    async def _get_fuzzy_fallback_context(self, ingredient: str, diet: str) -> str:
        """Fuzzy matching fallback when semantic search fails."""
        logger.info(f"Executing fuzzy fallback for '{ingredient}' for {diet} diet.")
        try:
            if diet == "keto":
                data = db_manager.query_nutrition_data(ingredient)
                logger.debug(f"Fuzzy keto data for '{ingredient}': {data}")
            else:  # vegan
                data = db_manager.query_vegan_ontology(ingredient)
                logger.debug(f"Fuzzy vegan data for '{ingredient}': {data}")

            if data:
                return json.dumps([data], indent=2)
            logger.info(
                f"No matching data found in fuzzy fallback for '{ingredient}' ({diet} diet)."
            )
            return "No matching data found in fuzzy fallback."
        except Exception as e:
            logger.error(f"Fuzzy fallback failed for '{ingredient}' ({diet} diet): {e}")
            return f"Fuzzy fallback error: {e}"

    async def classify_single_ingredient(self, ingredient: str) -> Dict[str, Any]:
        """
        SOTA semantic ingredient classification pipeline.
        """
        logger.info(f"Classifying single ingredient: '{ingredient}'")
        self.performance_stats["total_requests"] += 1
        clean_ingredient = parse_ingredient_simple(ingredient)
        logger.debug(f"Ingredient '{ingredient}' parsed to '{clean_ingredient}'.")

        # Step 1: Check cache
        cached_result = self.cache_manager.get_ingredient_classification(
            clean_ingredient
        )
        if cached_result:
            self.performance_stats["cache_hits"] += 1
            logger.info(f"Cache hit for '{clean_ingredient}'.")
            return cached_result

        self.performance_stats["cache_misses"] += 1
        logger.info(
            f"Cache miss for '{clean_ingredient}', proceeding with classification."
        )

        # Step 2: Sequential semantic context retrieval (using Arctic)
        # Changed from parallel to sequential
        logger.debug(
            f"Initiating semantic context retrieval for '{clean_ingredient}'..."
        )
        keto_context = await self._get_semantic_context(clean_ingredient, "keto")
        vegan_context = await self._get_semantic_context(clean_ingredient, "vegan")
        logger.debug(f"Keto context for '{clean_ingredient}': {keto_context[:100]}...")
        logger.debug(
            f"Vegan context for '{clean_ingredient}': {vegan_context[:100]}..."
        )

        # Step 3: Final classification (using Qwen)
        MAX_CONTEXT_LENGTH = (
            app_config.QWEN_MAX_CONTEXT_TOKENS
        )  # Use configured max context
        truncated_keto_context = (
            (keto_context[:MAX_CONTEXT_LENGTH] + "...")
            if len(keto_context) > MAX_CONTEXT_LENGTH
            else keto_context
        )
        truncated_vegan_context = (
            (vegan_context[:MAX_CONTEXT_LENGTH] + "...")
            if len(vegan_context) > MAX_CONTEXT_LENGTH
            else vegan_context
        )

        prompt = (
            f"""Classify this ingredient for keto and vegan diets based on the provided data.
        Respond with ONLY a JSON object containing 'is_keto' and 'is_vegan' """
            f"""booleans. Do NOT include any other text, reasoning, or markdown.

        INGREDIENT: "{ingredient}" (parsed as "{clean_ingredient}")

        NUTRITION DATA (for Keto analysis, truncated to {MAX_CONTEXT_LENGTH} chars):
        {truncated_keto_context}

        VEGAN ONTOLOGY DATA (for Vegan analysis, truncated to {MAX_CONTEXT_LENGTH} chars):
        {truncated_vegan_context}

        OUTPUT JSON:
        {{
          "is_keto": boolean,
          "is_vegan": boolean
        }}"""
        )

        model_name = "qwen/qwen3-0.6b-gguf:q8_0"  # Or your preferred Qwen model
        logger.debug(f"Sending prompt to Qwen model '{model_name}' for classification.")
        if not self.llm_client.is_model_available(model_name):
            logger.error(
                (
                    f"Classification model '{model_name}' is not available. "
                    f"Aborting classification for '{ingredient}'."
                )
            )
            return {"error": "Classification model not available"}

        try:
            result = await asyncio.wait_for(
                self.llm_client.query_async(
                    model_name, prompt, as_json=True, timeout=app_config.QWEN_TIMEOUT
                ),
                timeout=app_config.QWEN_TIMEOUT,  # Use configured timeout
            )
            logger.debug(f"Qwen classification result for '{ingredient}': {result}")
            result["extracted_ingredient"] = clean_ingredient
        except asyncio.TimeoutError:
            logger.error(
                f"Qwen classification timeout ({app_config.QWEN_TIMEOUT}s) for '{ingredient}'."
            )
            result = {"error": "Classification timed out."}
        except Exception as e:
            logger.error(f"Error during Qwen classification for '{ingredient}': {e}")
            result = {"error": f"Classification failed: {e}"}

        # Step 4: Cache the result
        if "error" not in result:
            self.cache_manager.set_ingredient_classification(clean_ingredient, result)
            logger.debug(f"Cached classification result for '{clean_ingredient}'.")
        else:
            logger.warning(
                f"Not caching result for '{clean_ingredient}' due to error: {result['error']}"
            )

        return result

    async def classify_recipe(self, ingredients: List[str]) -> Dict[str, Any]:
        """
        SOTA recipe classification with comprehensive semantic analysis.
        """
        logger.info(f"Classifying recipe with {len(ingredients)} ingredients.")
        if not ingredients:
            logger.warning(
                "No ingredients provided for recipe classification. Returning default true values."
            )
            return {
                "recipe_is_keto": True,
                "recipe_is_vegan": True,
                "reasoning": "Recipe has no ingredients.",
            }

        # Parallel classification of all ingredients
        logger.debug("Starting parallel classification of ingredients...")
        classifications = await asyncio.gather(
            *(self.classify_single_ingredient(ing) for ing in ingredients)
        )
        logger.debug(
            f"Completed parallel classification for {len(classifications)} ingredients."
        )

        recipe_is_keto = all(
            c.get("is_keto", False) for c in classifications if "error" not in c
        )
        recipe_is_vegan = all(
            c.get("is_vegan", False) for c in classifications if "error" not in c
        )

        # Aggregate reasoning
        non_keto_reasons = [
            f"- {ing}: {c.get('reasoning', 'No reason')}"
            for ing, c in zip(ingredients, classifications)
            if not c.get("is_keto")
        ]
        non_vegan_reasons = [
            f"- {ing}: {c.get('reasoning', 'No reason')}"
            for ing, c in zip(ingredients, classifications)
            if not c.get("is_vegan")
        ]

        final_reasoning = []
        if not recipe_is_keto:
            final_reasoning.append(
                "Recipe is not keto because:\n" + "\n".join(non_keto_reasons)
            )
            logger.debug("Recipe classified as NOT keto.")
        if not recipe_is_vegan:
            final_reasoning.append(
                "Recipe is not vegan because:\n" + "\n".join(non_vegan_reasons)
            )
            logger.debug("Recipe classified as NOT vegan.")

        result = {
            "recipe_is_keto": recipe_is_keto,
            "recipe_is_vegan": recipe_is_vegan,
            "reasoning": "\n".join(final_reasoning) or "Recipe is both keto and vegan.",
            "ingredient_analysis": classifications,
        }
        logger.info(
            f"Recipe classification complete. Keto: {recipe_is_keto}, Vegan: {recipe_is_vegan}"
        )
        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        total = stats.get("total_requests", 0)
        hits = stats.get("cache_hits", 0)
        stats["cache_hit_rate"] = (hits / total) if total > 0 else 0
        stats["cache_available"] = self.cache_manager.is_available()
        logger.debug(f"Performance statistics: {stats}")
        return stats
