#!/usr/bin/env python3
"""
Ground Truth Generation for Diet Classification

This script generates high-quality ground truth data for diet classification
by using multiple AI models to classify recipes as vegan and keto-friendly.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- External API Imports ---
import google.generativeai as genai  # noqa: E402
import polars as pl  # noqa: E402

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None
    MLFLOW_AVAILABLE = False
import tqdm.asyncio as tqdm_asyncio  # noqa: E402

# --- Local Module Imports ---
# Assumes the script is run with the `src` directory in the Python path.
# from llm_handler.handler import LLMHandler
from context_aware_classifier import SOTASemanticClassifier  # noqa: E402
from context_aware_classifier import execute_sql_query  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from google.generativeai.types import HarmBlockThreshold, HarmCategory  # noqa: E402

# Removed unused import: get_context_with_rapidfuzz_fallback
from opensearchpy import OpenSearch  # noqa: E402
from tqdm import tqdm  # noqa: E402

load_dotenv()

# --- Configuration ---
CONFIG = {
    "OUTPUT_DIR": Path("nb / src / data"),
    "GROUND_TRUTH_FILENAME": "personas_ground_truth.csv",
    "SAMPLE_SIZE": 100,
    "BATCH_SIZE": 25,
    "PRIMARY_MODEL": "gemini - 2.5 - pro",  # Emphasize Gemini first
    "FALLBACK_MODEL": "gemma3:1b",  # Fallback to Gemma3 - 1B if Gemini unavailable
    "TEACHER_MODEL": os.getenv("TEACHER_MODEL", "gemini - 2.5 - pro"),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "OPENSEARCH_URL": os.getenv("OPENSEARCH_URL", "http://localhost:9200"),
    "RPM_LIMIT": int(os.getenv("RPM_LIMIT", 10)),
    "TPM_LIMIT": int(os.getenv("TPM_LIMIT", 200000)),
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
}

MODELS_QUOTAS = {
    "gemini - 2.5 - pro": (5, 250_000),
    "gemini - 2.5 - flash": (10, 250_000),
    "gemini - 2.5 - flash - lite - preview - 06 - 17": (15, 250_000),
    "gemma3:1b": (60, 1_000_000),  # Higher limits for local Ollama model
}

logging.basicConfig(
    level=CONFIG["LOG_LEVEL"],
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)


class AsyncTokenBucket:
    """Manages API rate limiting for asynchronous requests using a token bucket algorithm.

    Detailed Description:
        - Ensures API requests do not exceed specified rate limits (RPM and TPM).
        - Uses `asyncio.Lock` for safe concurrent access.
        - Calculates and applies asynchronous delays when limits are reached.

    Parameters:
        - rpm_rate (int): Maximum requests per minute.
        - tpm_rate (int): Maximum tokens per minute.
    """

    def __init__(self, rpm_rate: int, tpm_rate: int):
        self.rpm_rate = rpm_rate
        self.tpm_rate = tpm_rate
        self.last_reset_time = time.time()
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int):
        async with self.lock:
            now = time.time()
            if now - self.last_reset_time >= 60:
                self.last_reset_time = now
                self.requests_this_minute = 0
                self.tokens_this_minute = 0
            time_to_wait_rpm = (
                60 - (now - self.last_reset_time)
                if self.requests_this_minute >= self.rpm_rate
                else 0.0
            )
            time_to_wait_tpm = (
                60 - (now - self.last_reset_time)
                if self.tokens_this_minute + estimated_tokens > self.tpm_rate
                else 0.0
            )
            time_to_wait = max(time_to_wait_rpm, time_to_wait_tpm)
            if time_to_wait > 0.1:
                logging.warning(
                    f"Rate limit nearing. Pausing for {time_to_wait:.2f}s..."
                )
                await asyncio.sleep(time_to_wait)
                self.last_reset_time = time.time()
                self.requests_this_minute = 0
                self.tokens_this_minute = 0
            self.requests_this_minute += 1
            self.tokens_this_minute += estimated_tokens


def build_vegan_prompt(recipe_rows: List[Dict[str, Any]]) -> list:
    """Builds a prompt for the generative AI to classify recipes as vegan.

    Detailed Description:
        - Formats recipe data into a detailed prompt for a generative AI model.
        - Instructs the AI to act as a food scientist and classify recipes as strictly vegan.
        - Specifies a chain-of-thought process and required JSON output format.

    Parameters:
        - recipe_rows (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
        represents a recipe.

    Returns:
        - list: A list containing the user prompt and the desired JSON schema.
    """
    recipe_data_list = [
        {
            "recipe_id": r.get("_id"),
            "title": r.get("title"),
            "ingredients": json.loads(r.get("ingredients", "[]")),
        }
        for r in recipe_rows
    ]
    user_prompt = f"""
    You are a meticulous food scientist. Your task is to determine if each recipe in the following
    list is strictly vegan.
    A recipe is **strictly vegan** if it contains absolutely no animal products (no meat, poultry,
    fish, dairy, eggs, honey, etc.).
    **Analysis Process:**
    1. For each recipe, carefully examine the ingredients.
    2. In your reasoning, first list any and all ingredients that are or could be derived from
    animals.
    3. After listing the evidence, make a final "Verdict" in your reasoning.
    4. Based on your verdict, set `is_vegan` to `true` or `false`.
    **RESPONSE INSTRUCTIONS:**
    - You must respond with ONLY a single, valid JSON array. Each object must correspond to a
    recipe.
    - If you identify ANY potential animal product, `is_vegan` MUST be `false`.
    **RECIPE DATA:**
    ```json
    {json.dumps(recipe_data_list)}
    ```
    """
    schema = [
        {
            "recipe_id": "string",
            "is_vegan": "boolean",
            "vegan_reasoning": "string",
            "animal_ingredients_found": ["list", "of", "strings"],
        }
    ]
    return [
        user_prompt,
        "Provide your analysis in this JSON format:",
        json.dumps(schema),
    ]


def build_keto_prompt(recipe_rows: List[Dict[str, Any]]) -> list:
    """Builds a prompt for the generative AI to classify recipes as keto-friendly.

    Detailed Description:
        - Constructs a prompt for a generative AI model to determine if recipes are keto-friendly.
        - Defines "strictly keto" based on carbohydrate content.
        - Instructs the AI to use its own knowledge to override incorrect information.
        - Specifies a JSON output format.

    Parameters:
        - recipe_rows (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
        represents a recipe.

    Returns:
        - list: A list containing the user prompt and the desired JSON schema.
    """
    recipe_data_list = []
    for row in recipe_rows:
        rag_summary = row.get("rag_summary", "No analysis available")
        sanitized_summary = rag_summary.replace('"', "'")
        recipe_data_list.append(
            {
                "recipe_id": row.get("_id"),
                "title": row.get("title"),
                "ingredients": json.loads(row.get("ingredients", "[]")),
                "rag_summary": sanitized_summary,
            }
        )
    user_prompt = f"""
    You are a meticulous nutritionist. Your task is to determine if each recipe is strictly
    keto-friendly.
    A recipe is **strictly keto** if it contains NO ingredients with more than 10g of carbohydrates
    per 100g serving.
    **CRITICAL INSTRUCTION FOR MISSING DATA:**
    - The `rag_summary` may be incomplete or incorrect. You MUST use your own expert knowledge to
    override it.
    - For example, you know that 'pizza crust', 'pasta', 'bread', 'sugar', 'potatoes', 'flour', and
    'rice' are ALWAYS high in carbohydrates (>10g/100g).
    Classify them as high-carb even if the summary says '0.0g'.
    **Analysis Process:**
    1. For each recipe, identify ingredients with >10g of carbs per 100g, using the `rag_summary`
    AND your own knowledge.
    2. In your reasoning, first list all high-carbohydrate ingredients you identified.
    3. After listing the evidence, make a final "Verdict" in your reasoning.
    4. Based on your verdict, set `is_keto` to `true` or `false`.
    **RESPONSE INSTRUCTIONS:**
    - You must respond with ONLY a single, valid JSON array. Each object must correspond to a
    recipe.
    - If you identify ANY high-carbohydrate ingredient, `is_keto` MUST be `false`.
    **RECIPE DATA:**
    ```json
    {json.dumps(recipe_data_list)}
    ```
    """
    schema = [
        {
            "recipe_id": "string",
            "is_keto": "boolean",
            "keto_reasoning": "string",
            "high_carb_ingredients_found": ["list", "of", "strings"],
        }
    ]
    return [
        user_prompt,
        "Provide your analysis in this JSON format:",
        json.dumps(schema),
    ]


async def classify_recipes(
    recipes: List[Dict[str, Any]],
    model_name: str,
    rate_limiter: AsyncTokenBucket,
    prompt_builder: Callable[[List[Dict[str, Any]]], list],
) -> Optional[List[Dict[str, Any]]]:
    """Classifies a batch of recipes using a generative AI model.

    Detailed Description:
        - Sends a request to the configured generative AI model.
        - Acquires a slot from the `AsyncTokenBucket` to respect rate limits.
        - Builds the prompt, sends it to the model, and parses the JSON response.
        - Includes error handling for API calls.

    Parameters:
        - recipes (List[Dict[str, Any]]): The list of recipes to classify.
        - model_name (str): The name of the generative AI model to use.
        - rate_limiter (AsyncTokenBucket): The rate limiter instance.
        - prompt_builder (Callable): The function for building the classification prompt.

    Returns:
        - Optional[List[Dict[str, Any]]]: A list of classification results, or None if the API call
        fails.

    Raises:
        - Exception: Catches and logs any exception during the API call, returning None.
    """
    if not recipes:
        return []
    estimated_tokens = (800 + 200) * len(recipes)
    await rate_limiter.acquire(estimated_tokens)
    prompt = prompt_builder(recipes)
    try:
        gemini_model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json", max_output_tokens=8192
        )
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        response = await gemini_model.generate_content_async(
            prompt, generation_config=generation_config, safety_settings=safety_settings
        )
        return json.loads(response.text)
    except Exception as e:
        logging.warning(
            f"API call failed for model {model_name} with {len(recipes)} recipes: {e}"
        )
        return None


# Add Arctic integration for nutritional queries
async def enrich_with_sql_context(recipe: Dict, sql_handler) -> Dict:
    """Enrich recipe with SQL-generated nutritional context."""
    enriched_ingredients = []

    for ingredient in recipe["ingredients"]:
        # Query nutrition data
        nutrition_q = f"SELECT * FROM nutrition_facts WHERE name LIKE '%{ingredient}%'"
        vegan_q = f"SELECT * FROM vegan_ontology WHERE term LIKE '%{ingredient}%'"

        nutrition_data = execute_sql_query(nutrition_q, ())
        vegan_data = execute_sql_query(vegan_q, ())

        enriched_ingredients.append(
            {"name": ingredient, "nutrition": nutrition_data, "vegan_info": vegan_data}
        )

    recipe["enriched_context"] = enriched_ingredients
    return recipe


async def worker(
    name: str,
    model_name: str,
    rate_limiter: AsyncTokenBucket,
    batch_queue: asyncio.Queue,
    results_list: List[Dict[str, Any]],
    pbar: tqdm,
    prompt_builder: Callable[[List[Dict[str, Any]]], list],
):
    """An asynchronous worker that processes recipe classification tasks from a queue.

    Detailed Description:
        - Processes batches of recipes from an `asyncio.Queue`.
        - Calls `classify_recipes` for each batch, retrying individual recipes on failure.
        - Updates a `tqdm` progress bar.

    Parameters:
        - name (str): Worker name (for logging).
        - model_name (str): Model to use for classification.
        - rate_limiter (AsyncTokenBucket): API rate limiter.
        - batch_queue (asyncio.Queue): Queue to fetch recipe batches from.
        - results_list (List[Dict[str, Any]]): List to store classification results.
        - pbar (tqdm): Progress bar to update.
        - prompt_builder (Callable): Function for building prompts.
    """
    while True:
        batch_id, batch_recipes_gen = await batch_queue.get()
        if batch_recipes_gen is None:
            batch_queue.task_done()
            break
        batch_recipes = list(batch_recipes_gen)
        classified_results = await classify_recipes(
            batch_recipes, model_name, rate_limiter, prompt_builder
        )
        if classified_results:
            if not isinstance(classified_results, list):
                classified_results = [classified_results]
            results_list.extend(classified_results)
            pbar.update(len(batch_recipes))
        else:
            logging.warning(
                f"Worker '{name}': Batch {batch_id} failed. Retrying recipes individually..."
            )
            for recipe in batch_recipes:
                individual_result = await classify_recipes(
                    [recipe], model_name, rate_limiter, prompt_builder
                )
                if individual_result:
                    if not isinstance(individual_result, list):
                        individual_result = [individual_result]
                    results_list.extend(individual_result)
                else:
                    logging.error(
                        (
                            f"Worker '{name}': Failed to process recipe {recipe.get('_id')} even "
                            f"individually."
                        )
                    )
                pbar.update(1)
        batch_queue.task_done()


async def run_classification_stage(
    df: pl.DataFrame, prompt_builder: Callable, stage_name: str
) -> pl.DataFrame:
    """Runs a complete classification stage for a given diet.

    Detailed Description:
        - Orchestrates a classification stage (e.g., "Vegan").
        - Creates a queue of recipe batches and a pool of worker tasks.
        - Workers use different models with varying rate limits for parallel processing.
        - Gathers results from all workers and returns them as a Polars DataFrame.

    Parameters:
        - df (pl.DataFrame): Input DataFrame of recipes.
        - prompt_builder (Callable): Function to build the prompt for this stage.
        - stage_name (str): Name of the stage (e.g., "Vegan", "Keto") for logging.

    Returns:
        - pl.DataFrame: DataFrame containing classification results.
    """
    logging.info(f"Starting Stage: {stage_name} Classification...")
    work_queue = asyncio.Queue()
    all_results = []
    for i in range(0, len(df), CONFIG["BATCH_SIZE"]):
        batch_id = i // CONFIG["BATCH_SIZE"]
        batch_df = df.slice(i, CONFIG["BATCH_SIZE"])
        work_queue.put_nowait((batch_id, batch_df.iter_rows(named=True)))
    logging.info(f"Prepared {work_queue.qsize()} batches for {stage_name} stage.")
    worker_tasks = []
    with tqdm(total=df.shape[0], desc=f"Classifying ({stage_name})") as pbar:
        for model_api_name, (rpm_limit, tpm_limit) in MODELS_QUOTAS.items():
            rate_limiter = AsyncTokenBucket(rpm_rate=rpm_limit, tpm_rate=tpm_limit)
            task = asyncio.create_task(
                worker(
                    f"Worker-{model_api_name}",
                    model_api_name,
                    rate_limiter,
                    work_queue,
                    all_results,
                    pbar,
                    prompt_builder,
                )
            )
            worker_tasks.append(task)
            await work_queue.put((None, None))
        await work_queue.join()
        await asyncio.gather(*worker_tasks)
    if not all_results:
        logging.error(f"No results were generated for stage {stage_name}.")
        return pl.DataFrame()
    return pl.DataFrame(all_results)


def wait_for_index(
    client: OpenSearch, index_name: str, max_retries: int = 20, retry_interval: int = 10
) -> bool:
    """Waits for a specific OpenSearch index to exist."""
    logging.info(f"Waiting for OpenSearch index '{index_name}' to be created...")
    for i in range(max_retries):
        try:
            if client.indices.exists(index=index_name):
                logging.info(f"Index '{index_name}' found.")
                return True
        except Exception as e:
            logging.warning(f"Error checking for index: {e}")
        logging.info(
            f"Retrying in {retry_interval} seconds... (attempt {i + 1}/{max_retries})"
        )
        time.sleep(retry_interval)
    logging.error(f"Index '{index_name}' not found after maximum retries.")
    return False


def fetch_recipes_synchronously(os_client: OpenSearch) -> List[Dict]:
    """Fetches a random sample of recipes using the synchronous client."""
    logging.info(f"Fetching {CONFIG['SAMPLE_SIZE']} random recipes from OpenSearch...")
    query = {
        "size": CONFIG["SAMPLE_SIZE"],
        "query": {"function_score": {"functions": [{"random_score": {}}]}},
    }
    response = os_client.search(index="recipes", body=query)
    return response["hits"]["hits"]


async def classify_batch(
    batch: List[Dict],
    classifier: SOTASemanticClassifier,
    rate_limiter: AsyncTokenBucket,
) -> List[Dict]:
    """Classifies a batch of recipes using the LLM handler for both vegan and keto
    classifications."""
    try:
        results = []
        for recipe in batch:
            recipe_id = recipe.get("recipe_id")
            ingredients = json.loads(recipe.get("ingredients", "[]"))
            title = recipe.get("title", "Unknown Recipe")

            # Use the new context-aware classifier
            llm_classification_result = await classifier.classify_with_context(
                ingredients, title
            )

            merged_result = {
                "recipe_id": recipe_id,
                "title": title,
                # Keep ingredients as a JSON string for consistency in output
                "ingredients": ingredients,
            }

            # Add classification results
            if llm_classification_result:
                merged_result["is_vegan"] = llm_classification_result.get(
                    "is_vegan", False
                )
                merged_result["vegan_reasoning"] = llm_classification_result.get(
                    "reasoning", "No specific vegan reasoning provided."
                )
                merged_result["animal_ingredients_found"] = (
                    []
                )  # The new model integrates this into reasoning

                merged_result["is_keto"] = llm_classification_result.get(
                    "is_keto", False
                )
                merged_result["keto_reasoning"] = llm_classification_result.get(
                    "reasoning", "No specific keto reasoning provided."
                )
                merged_result["high_carb_ingredients_found"] = (
                    []
                )  # The new model integrates this into reasoning
            else:
                merged_result.update(
                    {
                        "is_vegan": False,
                        "vegan_reasoning": "Classification failed",
                        "animal_ingredients_found": [],
                    }
                )
                merged_result.update(
                    {
                        "is_keto": False,
                        "keto_reasoning": "Classification failed",
                        "high_carb_ingredients_found": [],
                    }
                )

            results.append(merged_result)

        return results

    except Exception as e:
        logging.error(f"Batch classification failed: {e}")
        return []


async def classify_and_log_async(processed_recipes: List[Dict]):
    """Runs the asynchronous classification and MLflow logging pipeline."""
    classifier = SOTASemanticClassifier()
    rate_limiter = AsyncTokenBucket(CONFIG["RPM_LIMIT"], CONFIG["TPM_LIMIT"])

    # Run Classification
    tasks = []
    for i in range(0, len(processed_recipes), CONFIG["BATCH_SIZE"]):
        batch = processed_recipes[i : i + CONFIG["BATCH_SIZE"]]
        tasks.append(classify_batch(batch, classifier, rate_limiter))

    all_results = []
    for result in await tqdm_asyncio.gather(*tasks, desc="Classifying Batches"):
        all_results.extend(result)

    # MLflow Logging & Saving
    if not all_results:
        logging.error("Classification failed for all batches. No data to save.")
        return

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(CONFIG["MLFLOW_TRACKING_URI"])
        with mlflow.start_run(run_name="SOTA Personas Ground Truth Generation") as run:
            logging.info(f"MLflow run started: {run.info.run_id}")
            mlflow.log_params(CONFIG)

            final_data = []
            for res in all_results:
                # Flatten lists before saving to CSV
                flat_res = {
                    "recipe_id": res.get("recipe_id"),
                    "title": res.get("title"),
                }
                for k, v in res.items():
                    if k not in ["recipe_id", "title"]:
                        if isinstance(v, list):
                            flat_res[k] = ", ".join(map(str, v))
                        else:
                            flat_res[k] = v
                final_data.append(flat_res)

            final_df = pl.DataFrame(final_data)

            if not final_df.is_empty():
                output_path = CONFIG["OUTPUT_DIR"] / CONFIG["GROUND_TRUTH_FILENAME"]
                final_df.write_csv(output_path)
                mlflow.log_artifact(str(output_path), "generated_datasets")
                logging.info(
                    f"Ground truth generation complete. Saved to {output_path}"
                )
            else:
                logging.warning("No results were generated. Skipping artifact logging.")


def main():
    """Main execution block, orchestrating synchronous and asynchronous parts."""
    logging.info("Starting SOTA Personas Ground Truth Generation Pipeline...")

    # Connect, Wait, and Fetch Data
    os_client = OpenSearch(hosts=[CONFIG["OPENSEARCH_URL"]])
    if not os_client.ping() or not wait_for_index(os_client, "recipes"):
        logging.error("OpenSearch is not available. Exiting.")
        return

    recipes_hits = fetch_recipes_synchronously(os_client)
    recipes_df = pl.DataFrame([hit["_source"] for hit in recipes_hits])
    recipes_df = recipes_df.with_columns(
        pl.Series(name="recipe_id", values=[hit["_id"] for hit in recipes_hits])
    )

    async def run_stages():
        # Run Teacher Model Classification Stages
        vegan_results_df = await run_classification_stage(
            recipes_df, build_vegan_prompt, "Vegan"
        )

        keto_results_df = await run_classification_stage(
            recipes_df, build_keto_prompt, "Keto"
        )

        # Merge results and save
        if not vegan_results_df.is_empty() and not keto_results_df.is_empty():
            # Merge results on recipe_id
            final_df = vegan_results_df.join(
                keto_results_df, on="recipe_id", how="outer"
            )

            # Flatten list columns for CSV writing
            for col in final_df.columns:
                if final_df[col].dtype == pl.List:
                    final_df = final_df.with_columns(
                        pl.col(col).list.join(", ").alias(col)
                    )

            output_path = CONFIG["OUTPUT_DIR"] / CONFIG["GROUND_TRUTH_FILENAME"]
            final_df.write_csv(output_path)
            logging.info(f"Ground truth generation complete. Saved to {output_path}")

            # MLflow Logging
            if MLFLOW_AVAILABLE:
                mlflow.set_tracking_uri(CONFIG["MLFLOW_TRACKING_URI"])
                with mlflow.start_run(
                    run_name="SOTA Personas Ground Truth Generation"
                ) as run:
                    logging.info(f"MLflow run started: {run.info.run_id}")
                    mlflow.log_params(CONFIG)
                    mlflow.log_artifact(str(output_path), "generated_datasets")
        else:
            logging.error("Classification stages failed. No data to save.")

    # Run the async stages
    asyncio.run(run_stages())


if __name__ == "__main__":
    logging.basicConfig(
        level=CONFIG["LOG_LEVEL"],
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    main()
