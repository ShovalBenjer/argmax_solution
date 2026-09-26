"""
Final Evaluation Script for Diet Classification System.

This script performs comprehensive evaluation of the SOTA, context-aware
dietary classification system against Gemini-generated persona ground truth.
"""

import ast
import asyncio
import sys
import time
from pathlib import Path

import polars as pl
from loguru import logger

try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass

from config import app_config
from context_aware_classifier import SOTASemanticClassifier
from llm_client import llm_client
from sklearn.metrics import classification_report


def check_system_health():
    """Check system health before starting evaluation."""
    logger.info("Checking system health...")

    # List all available models for debugging
    all_models = llm_client.list_models()
    if all_models:
        # Extract model names, handling both 'model' attribute for Model objects
        # and 'name' for dicts
        model_names = []
        for m in all_models:
            if hasattr(m, "model"):  # Ollama Model object
                model_names.append(m.model)
            elif isinstance(m, dict) and m.get(
                "name"
            ):  # Fallback for dict-like responses
                model_names.append(m.get("name"))
        logger.info(f"Available models: {model_names}")
    else:
        logger.warning("No models found in Ollama")

    # Check for Qwen model (classification)
    qwen_model = "qwen/qwen3-0.6b-gguf:q8_0"
    logger.info(f"Verifying availability of classification model: '{qwen_model}'")
    if not llm_client.is_model_available(qwen_model):
        logger.error(
            f"Qwen model '{qwen_model}' is required for classification but was not found."
        )
        return False
    logger.success(f"Classification model '{qwen_model}' is available.")

    # Check for Arctic model (Text2SQL)
    arctic_model_primary = "arctic-text2sql:latest"
    arctic_model_fallback = "arctic-text2sql-r1-7b"
    logger.info(
        (
            f"Verifying availability of Text2SQL model: '{arctic_model_primary}' or "
            f"'{arctic_model_fallback}'"
        )
    )
    if not llm_client.is_model_available(arctic_model_primary):
        if not llm_client.is_model_available(arctic_model_fallback):
            logger.error(
                (
                    f"Neither Arctic model '{arctic_model_primary}' nor '{arctic_model_fallback}' "
                    f"was found. One is required for "
                    f"Text2SQL."
                )
            )
            return False
        else:
            logger.success(
                f"Arctic model '{arctic_model_fallback}' is available (fallback)."
            )
    else:
        logger.success(f"Arctic model '{arctic_model_primary}' is available.")

    logger.success("All required LLM models are available.")
    return True


def load_gold_label_personas(eval_data_dir: Path) -> pl.DataFrame:
    """
    Loads and robustly standardizes all gold label persona tables for evaluation.
    Handles inconsistent schemas and column names across different CSV files.
    """
    logger.info(f"Attempting to load gold label personas from: {eval_data_dir}")
    if not eval_data_dir.exists():
        logger.critical(
            f"Evaluation data directory not found: {eval_data_dir}. Exiting."
        )
        sys.exit(1)

    persona_files = list(eval_data_dir.glob("*.csv"))
    if not persona_files:
        logger.critical(
            f"No CSV files found in evaluation directory: {eval_data_dir}. Exiting."
        )
        sys.exit(1)

    logger.info(
        f"Found {len(persona_files)} evaluation files: {[f.name for f in persona_files]}"
    )

    # Define the final, unified schema we want all DataFrames to conform to.
    final_schema = {
        "recipe_id": pl.Utf8,
        "title": pl.Utf8,
        "ingredients": pl.Utf8,
        "is_keto": pl.Boolean,
        "is_vegan": pl.Boolean,
        "persona": pl.Utf8,
        "reasoning": pl.Utf8,
        "confidence": pl.Float64,
        "description": pl.Utf8,
        "instructions": pl.Utf8,
        "photo_url": pl.Utf8,
    }
    logger.debug(f"Defined final schema for persona data: {final_schema.keys()}")

    all_dfs = []
    for file_path in persona_files:
        logger.info(f"Processing file: {file_path.name}")
        try:
            df = pl.read_csv(file_path)
            logger.debug(
                f"Successfully read {file_path.name}. Initial columns: {df.columns}"
            )

            # Handle inconsistent column names for keto/vegan flags
            if "vegan" in df.columns and "is_vegan" not in df.columns:
                df = df.rename({"vegan": "is_vegan"})
                logger.debug(f"Renamed 'vegan' to 'is_vegan' in {file_path.name}.")
            if "keto" in df.columns and "is_keto" not in df.columns:
                df = df.rename({"keto": "is_keto"})
                logger.debug(f"Renamed 'keto' to 'is_keto' in {file_path.name}.")

            # Ensure all columns from the final schema exist, adding them with nulls if not.
            for col, dtype in final_schema.items():
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
                    logger.debug(f"Added missing column '{col}' to {file_path.name}.")

            # Select only the columns in our final schema, in the correct order.
            df = df.select(list(final_schema.keys()))
            all_dfs.append(df)
            logger.info(
                f"Successfully loaded and standardized {file_path.name}. Rows: {len(df)}"
            )

        except Exception as e:
            logger.warning(
                f"Could not load or process {file_path.name}: {e}. Skipping this file."
            )

    if not all_dfs:
        logger.critical(
            "No data could be loaded for evaluation after processing all files. Exiting."
        )
        sys.exit(1)

    # Concatenate all standardized DataFrames.
    df_all = pl.concat(all_dfs, how="vertical_relaxed")
    logger.info(
        f"Loaded a total of {len(df_all)} records for evaluation after concatenation."
    )

    # Safely parse the ingredients string into a list.
    def safe_eval_ingredients(val: str) -> list:
        if isinstance(val, str) and val.startswith("["):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError) as e:
                logger.warning(
                    f"Failed to parse ingredient string '{val}': {e}. Returning empty list."
                )
                return []
        logger.debug(
            (
                f"Ingredient value '{val}' is not a string or does not start with '['. Returning "
                f"empty "
                f"list."
            )
        )
        return []

    df_all = df_all.with_columns(
        pl.col("ingredients")
        .map_elements(safe_eval_ingredients, return_dtype=pl.List(pl.Utf8))
        .alias("ingredients_list")
    )
    logger.info(f"Parsed ingredients into list for {len(df_all)} records.")

    # Filter out records with empty ingredient lists.
    initial_rows = len(df_all)
    df_filtered = df_all.filter(pl.col("ingredients_list").list.len() > 0)
    filtered_rows = len(df_filtered)
    if filtered_rows < initial_rows:
        logger.warning(
            f"Filtered out {initial_rows - filtered_rows} records with empty ingredient lists."
        )
    logger.info(
        f"Returning {filtered_rows} records after filtering for non-empty ingredient lists."
    )
    return df_filtered


async def run_classification_for_evaluation(df: pl.DataFrame) -> pl.DataFrame:
    """Runs the SOTA classification for the entire evaluation dataset."""
    # Use fast_mode=True to avoid Arctic timeouts as it relies on fuzzy matching.
    # The timeout for classify_recipe will be handled by app_config.ARCTIC_TIMEOUT
    # and app_config.QWEN_TIMEOUT
    classifier = SOTASemanticClassifier(fast_mode=True)
    predictions = []

    logger.info(f"Starting SOTA classification for {len(df)} recipes.")
    for i, row in enumerate(df.to_dicts()):
        recipe_id = row.get("recipe_id", f"unknown_id_{i}")
        ingredients = row.get("ingredients_list", [])
        title = row.get("title", "No Title")

        logger.info(f"Processing recipe {i + 1}/{len(df)}: '{title}' (ID: {recipe_id})")

        if not ingredients:
            logger.warning(
                f"Skipping recipe '{recipe_id}' due to empty ingredient list."
            )
            predictions.append(
                {
                    "recipe_id": recipe_id,
                    "title": title,
                    "true_is_keto": row.get("is_keto"),
                    "true_is_vegan": row.get("is_vegan"),
                    "pred_is_keto": None,
                    "pred_is_vegan": None,
                    "reasoning": "Skipped: Empty ingredient list.",
                }
            )
            continue

        try:
            # Use a combined timeout for the entire recipe classification
            classification_timeout = (
                app_config.ARCTIC_TIMEOUT + app_config.QWEN_TIMEOUT + 5.0
            )  # Add buffer
            logger.debug(
                (
                    f"Setting classification timeout for recipe {recipe_id} to "
                    f"{classification_timeout}s."
                )
            )
            result = await asyncio.wait_for(
                classifier.classify_recipe(ingredients), timeout=classification_timeout
            )
            logger.debug(
                (
                    f"Successfully classified recipe {recipe_id}. Result: "
                    f"{result.get('recipe_is_keto')}, "
                    f"{result.get('recipe_is_vegan')}"
                )
            )
            predictions.append(
                {
                    "recipe_id": recipe_id,
                    "title": title,
                    "true_is_keto": row.get("is_keto"),
                    "true_is_vegan": row.get("is_vegan"),
                    "pred_is_keto": result.get("recipe_is_keto", False),
                    "pred_is_vegan": result.get("recipe_is_vegan", False),
                    "reasoning": result.get("reasoning", "No reasoning provided."),
                }
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout classifying recipe {recipe_id} after {classification_timeout}s."
            )
            predictions.append(
                {
                    "recipe_id": recipe_id,
                    "title": title,
                    "true_is_keto": row.get("is_keto"),
                    "true_is_vegan": row.get("is_vegan"),
                    "pred_is_keto": None,
                    "pred_is_vegan": None,
                    "reasoning": f"Classification timed out after {classification_timeout}s.",
                }
            )
        except Exception as e:
            logger.error(
                f"Error classifying recipe {recipe_id}: {e}. Appending error result."
            )
            predictions.append(
                {
                    "recipe_id": recipe_id,
                    "title": title,
                    "true_is_keto": row.get("is_keto"),
                    "true_is_vegan": row.get("is_vegan"),
                    "pred_is_keto": None,
                    "pred_is_vegan": None,
                    "reasoning": f"Classification failed: {e}",
                }
            )

    logger.info(
        f"Finished SOTA classification. Generated {len(predictions)} predictions."
    )
    return pl.DataFrame(predictions)


def calculate_and_report_metrics(eval_df: pl.DataFrame):
    """Calculates and prints comprehensive classification metrics."""
    logger.info("Calculating and reporting classification metrics...")
    # Filter out rows where prediction failed (pred_is_keto or pred_is_vegan is None)
    valid_predictions = eval_df.filter(
        pl.col("pred_is_keto").is_not_null() & pl.col("pred_is_vegan").is_not_null()
    )

    if valid_predictions.is_empty():
        logger.error(
            "No valid predictions were made after filtering. Cannot calculate metrics."
        )
        return

    logger.info(
        f"Calculating metrics based on {len(valid_predictions)} valid classifications."
    )
    eval_pandas = valid_predictions.to_pandas()

    logger.info("Generating Keto Classification Report.")
    keto_report = classification_report(
        eval_pandas["true_is_keto"],
        eval_pandas["pred_is_keto"],
        target_names=["Not Keto", "Keto"],
        zero_division=0,
        output_dict=True,  # Get report as dict to log easily
    )
    print("\n" + "=" * 20 + " Keto Classification Evaluation " + "=" * 20)
    print(
        classification_report(
            eval_pandas["true_is_keto"],
            eval_pandas["pred_is_keto"],
            target_names=["Not Keto", "Keto"],
            zero_division=0,
        )
    )
    logger.debug(f"Keto Classification Report: {keto_report}")

    logger.info("Generating Vegan Classification Report.")
    vegan_report = classification_report(
        eval_pandas["true_is_vegan"],
        eval_pandas["pred_is_vegan"],
        target_names=["Not Vegan", "Vegan"],
        zero_division=0,
        output_dict=True,  # Get report as dict to log easily
    )
    print("\n" + "=" * 20 + " Vegan Classification Evaluation " + "=" * 20)
    print(
        classification_report(
            eval_pandas["true_is_vegan"],
            eval_pandas["pred_is_vegan"],
            target_names=["Not Vegan", "Vegan"],
            zero_division=0,
        )
    )
    logger.debug(f"Vegan Classification Report: {vegan_report}")

    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "final_evaluation_predictions.csv"
    eval_df.write_csv(output_file)
    logger.success(f"Detailed predictions saved to {output_file.resolve()}")


async def main():
    """Main execution function for the final evaluation pipeline."""
    logger.info("Starting Final Evaluation of SOTA Classifier...")

    if not check_system_health():
        logger.critical("System health check failed. Aborting evaluation.")
        return -1

    eval_data_dir = app_config.EVAL_DATA_DIR
    ground_truth_df = load_gold_label_personas(eval_data_dir)

    if ground_truth_df.is_empty():
        logger.error("Ground truth data is empty after processing. Aborting.")
        return -1

    logger.info(f"Beginning classification for {len(ground_truth_df)} recipes.")
    start_time = time.time()
    evaluation_predictions_df = await run_classification_for_evaluation(ground_truth_df)
    end_time = time.time()

    if evaluation_predictions_df.is_empty():
        logger.error(
            "Classification resulted in an empty predictions DataFrame. Aborting."
        )
        return -1

    logger.info("--- Evaluation Results Summary ---")
    calculate_and_report_metrics(evaluation_predictions_df)

    total_time = end_time - start_time
    num_predictions = len(evaluation_predictions_df)
    avg_time = total_time / num_predictions if num_predictions > 0 else 0
    logger.success(
        (
            f"Evaluation finished in {total_time:.2f}s (Avg: {avg_time * 1000:.2f} ms/recipe for "
            f"{num_predictions} "
            f"recipes)."
        )
    )
    return 0


if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        logger.info("Existing asyncio loop found.")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("No existing asyncio loop found, created a new one.")

    logger.info("Running main evaluation pipeline.")
    sys.exit(loop.run_until_complete(main()))
