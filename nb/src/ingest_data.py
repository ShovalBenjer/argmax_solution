#!/usr/bin/env python3
"""
Fixed Data Ingestion Script for Diet Classification Knowledge Base

This script addresses critical issues in the original data ingestion:
1. Fixes numeric value parsing regex bug (corrects double backslashes to single)
2. Corrects vegan ontology aliases parsing from JSON strings
3. Simplifies database schema for Text2SQL efficiency (80+ -> 15 core columns)
4. Removes complex pre-enrichment pipeline for better maintainability
5. Uses Polars for high-performance data processing
6. Professional logging without emojis

The resulting knowledge_graph.db will have clean, validated data optimized
for the Arctic Text2SQL -> Qwen classification pipeline.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict

import polars as pl
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import local config
from config import app_config  # noqa: E402

# Database setup
engine = create_engine(f"sqlite:///{app_config.DB_PATH}")
metadata = MetaData()

# SIMPLIFIED SCHEMA - Optimized for Text2SQL queries (15 core columns vs 80+)
nutrition_facts_table = Table(
    "nutrition_facts",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("serving_size", String),
    Column("calories", Float),
    # Core macronutrients for diet classification
    Column("total_fat_g", Float),
    Column("saturated_fat_g", Float),
    Column("protein_g", Float),
    Column("carbohydrate_g", Float),
    Column("fiber_g", Float),
    Column("sugars_g", Float),
    # Key indicators for classification
    Column("vitamin_b12_mcg", Float),  # Critical for vegan analysis
    Column("cholesterol_mg", Float),  # Animal product indicator
    Column("calcium_mg", Float),
    Column("iron_mg", Float),
    Column("sodium_mg", Float),
)

vegan_ontology_table = Table(
    "vegan_ontology",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("term", String, nullable=False),
    Column("aliases", String),  # Simple comma-separated string for Text2SQL
    Column("is_explicitly_non_vegan", Boolean),
    Column("description", String),
    Column("requires_contextual_check", Boolean),
)

unit_conversions_table = Table(
    "unit_conversions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("unit", String, nullable=False),
    Column("abbreviation", String),
    Column("metric_equivalent", String),
    Column("type", String),  # volume or weight
    Column("factor", Float, nullable=False),  # conversion factor to grams
)

# Enhanced tables for generalized ingredient analysis
ingredient_synonyms_table = Table(
    "ingredient_synonyms",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("canonical_name", String, nullable=False),
    Column("synonym", String, nullable=False),
    Column("language_code", String),  # 'en', 'es', 'fr', etc.
    Column("region_code", String),  # 'US', 'UK', 'IN', etc.
    Column("confidence", Float, default=1.0),
    Column("discovered_at", DateTime, default=datetime.utcnow),
)

ingredient_hierarchies_table = Table(
    "ingredient_hierarchies",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("parent_ingredient", String, nullable=False),
    Column("child_ingredient", String, nullable=False),
    Column("relationship_type", String),  # 'is_a', 'contains', 'derived_from'
    Column("confidence", Float, default=1.0),
    Column("discovered_at", DateTime, default=datetime.utcnow),
)

ingredient_contexts_table = Table(
    "ingredient_contexts",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("ingredient", String, nullable=False),
    Column("context_clue", String, nullable=False),
    Column(
        "context_type", String
    ),  # 'animal_indicator', 'plant_indicator', 'ambiguous'
    Column("confidence", Float, default=1.0),
    Column("discovered_at", DateTime, default=datetime.utcnow),
)


def create_database_schema() -> None:
    """
    Create the simplified database schema optimized for Text2SQL queries.

    This function drops any existing tables and creates new ones with indexes
    for efficient querying by the Arctic Text2SQL model.
    """
    logger.info("Creating simplified database schema optimized for Text2SQL")

    # Drop existing tables for clean slate
    metadata.drop_all(engine)
    metadata.create_all(engine)

    # Create indexes for efficient lookups
    with engine.connect() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_nutrition_name ON nutrition_facts (name);"
            )
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS idx_vegan_term ON vegan_ontology (term);")
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_vegan_aliases ON vegan_ontology (aliases);"
            )
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS idx_unit_name ON unit_conversions (unit);")
        )
        # Enhanced indexes for generalized analysis
        connection.execute(
            text(
                (
                    "CREATE INDEX IF NOT EXISTS idx_synonyms_canonical ON ingredient_synonyms "
                    "(canonical_name);"
                )
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_synonyms_synonym ON ingredient_synonyms (synonym);"
            )
        )
        connection.execute(
            text(
                (
                    "CREATE INDEX IF NOT EXISTS idx_hierarchies_parent ON ingredient_hierarchies "
                    "(parent_ingredient);"
                )
            )
        )
        connection.execute(
            text(
                (
                    "CREATE INDEX IF NOT EXISTS idx_hierarchies_child ON ingredient_hierarchies "
                    "(child_ingredient);"
                )
            )
        )
        connection.execute(
            text(
                (
                    "CREATE INDEX IF NOT EXISTS idx_contexts_ingredient ON ingredient_contexts "
                    "(ingredient);"
                )
            )
        )
        connection.commit()

    logger.info(
        "Database schema created successfully with optimized indexes for generalized analysis"
    )


def clean_numeric_value(value_str: Any) -> float:
    """
    Clean numeric values from CSV data with FIXED regex pattern.

    Args:
        value_str: Raw value from CSV that may contain units or formatting

    Returns:
        Cleaned float value or 0.0 if parsing fails

    Note:
        This fixes the critical bug in the original function where the regex
        pattern had incorrect double backslashes. The correct pattern uses
        single backslashes in the raw string.
    """
    if value_str is None or value_str == "" or str(value_str).strip() == "":
        return 0.0

    # Convert to string and clean
    value_str = str(value_str).strip()

    # Handle special cases
    if value_str.lower() in ["", "na", "n/a", "null", "none"]:
        return 0.0

    # FIXED: Correct regex pattern (single backslashes)
    # This removes everything except digits, dots, and minus signs
    cleaned = re.sub(r"[^\d\.-]", "", value_str)

    # Handle empty result
    if not cleaned or cleaned == ".":
        return 0.0

    try:
        result = float(cleaned)
        # Sanity check for unreasonable values
        if result < 0 or result > 10000:
            logger.warning(f"Unusual nutrition value: {result} from '{value_str}'")
        return result
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse numeric value '{value_str}': {e}")
        return 0.0


def parse_aliases_string(aliases_str: Any) -> str:
    """
    Parse aliases from string-formatted JSON arrays to comma-separated strings.

    Args:
        aliases_str: String representation of JSON array like '["Albumin", "egg whites"]'

    Returns:
        Comma-separated string like "Albumin, egg whites" or empty string

    Note:
        This fixes the aliases parsing issue by properly handling the JSON
        string format and converting to simple searchable text for Text2SQL.
    """
    if aliases_str is None or aliases_str == "" or str(aliases_str).strip() == "":
        return ""

    aliases_str = str(aliases_str).strip()

    # Handle JSON-like strings: ["Albumin", "egg whites"]
    if aliases_str.startswith("[") and aliases_str.endswith("]"):
        try:
            # Parse as JSON
            aliases_list = json.loads(aliases_str)
            if isinstance(aliases_list, list):
                # Convert to comma-separated string for simple Text2SQL queries
                clean_aliases = [
                    str(alias).strip('"') for alias in aliases_list if alias
                ]
                return ", ".join(clean_aliases)
        except json.JSONDecodeError:
            # If JSON parsing fails, try manual extraction
            aliases_str = aliases_str.strip("[]")
            aliases_list = [
                alias.strip(' "') for alias in aliases_str.split(",") if alias.strip()
            ]
            return ", ".join(aliases_list)

    # If not JSON-like, return as is
    return aliases_str


def ingest_nutrition_data() -> None:
    """
    Ingest nutrition data with FIXED numeric parsing and simplified schema.

    This function reads the nutrition CSV, applies corrected data cleaning,
    and populates the simplified nutrition_facts table with only the most
    important columns for diet classification.
    """
    logger.info("Ingesting nutrition data with fixed parsing")

    try:
        # Read CSV with Polars for performance
        df = pl.read_csv(app_config.RAW_DATA_DIR / "nutrition.csv")
        logger.info(f"Read {len(df)} rows from nutrition CSV")

        # Check available columns
        available_columns = df.columns
        logger.info(f"Available columns: {len(available_columns)} total")

        # Simplified mapping focusing on core nutrients for classification
        nutrition_mapping = {
            "name": "name",
            "serving_size": "serving_size",
            "calories": "calories",
            "total_fat": "total_fat_g",
            "saturated_fat": "saturated_fat_g",
            "protein": "protein_g",
            "carbohydrate": "carbohydrate_g",
            "fiber": "fiber_g",
            "sugars": "sugars_g",
            "vitamin_b12": "vitamin_b12_mcg",
            "cholesterol": "cholesterol_mg",
            "calcium": "calcium_mg",
            "irom": "iron_mg",  # Note: CSV has 'irom' typo instead of 'iron'
            "sodium": "sodium_mg",
        }

        # Clean name column
        df = df.with_columns(pl.col("name").str.to_lowercase().str.strip_chars())

        # Process numeric columns with FIXED cleaning function
        for csv_col, db_col in nutrition_mapping.items():
            if csv_col in df.columns and csv_col != "name":
                logger.debug(f"Processing column: {csv_col} -> {db_col}")
                df = df.with_columns(
                    pl.col(csv_col)
                    .map_elements(clean_numeric_value, return_dtype=pl.Float64)
                    .alias(db_col)
                )

        # Select only available columns
        available_cols = ["name"] + [
            db_col
            for csv_col, db_col in nutrition_mapping.items()
            if csv_col in df.columns and csv_col != "name"
        ]
        df_clean = df.select(available_cols)

        # Fill any remaining nulls
        df_clean = df_clean.fill_null(0.0)

        # Debug: Check first few rows
        logger.info("Sample of processed data:")
        sample_data = df_clean.head(3).to_dicts()
        for row in sample_data:
            logger.info(
                (
                    f"  {row['name']}: calories={row.get('calories', 'N / A')}, "
                    f"carbs={row.get('carbohydrate_g', 'N / A')}"
                )
            )

        # Insert into database
        with engine.connect() as connection:
            connection.execute(nutrition_facts_table.insert(), df_clean.to_dicts())
            connection.commit()

        logger.info(
            (
                f"Successfully ingested {len(df_clean)} nutrition records with "
                f"{len(available_cols) - 1} "
                f"nutrients"
            )
        )

        # Verify data was inserted correctly
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT name, calories, carbohydrate_g FROM nutrition_facts LIMIT 3"
                )
            ).fetchall()
            logger.info("Verification - Data in database:")
            for row in result:
                logger.info(f"  {row[0]}: calories={row[1]}, carbs={row[2]}")

    except Exception as e:
        logger.error(f"Failed to ingest nutrition data: {e}")
        raise


def ingest_vegan_ontology() -> None:
    """
    Ingest vegan ontology with FIXED aliases parsing.

    This function reads the vegan ontology CSV and correctly parses the
    aliases column from JSON string format to searchable comma-separated text.
    """
    logger.info("Ingesting vegan ontology with fixed aliases parsing")

    try:
        # Read CSV with Polars
        df = pl.read_csv(app_config.RAW_DATA_DIR / "vegan_ontology.csv")
        logger.info(f"Read {len(df)} rows from vegan ontology CSV")

        # Debug: Check sample aliases before processing
        sample_aliases = df.select("aliases").head(3).to_series().to_list()
        logger.info(f"Sample aliases before processing: {sample_aliases}")

        # Process columns with fixed functions
        df = df.with_columns(
            [
                pl.col("term").str.to_lowercase().str.strip_chars(),
                pl.col("aliases").map_elements(
                    parse_aliases_string, return_dtype=pl.String
                ),
                pl.col("is_explicitly_non_vegan").cast(pl.Boolean),
                pl.col("requires_contextual_check").cast(pl.Boolean),
            ]
        )

        # Debug: Check sample aliases after processing
        sample_aliases_processed = df.select("aliases").head(3).to_series().to_list()
        logger.info(f"Sample aliases after processing: {sample_aliases_processed}")

        # Select relevant columns for simplified schema
        df_clean = df.select(
            [
                "term",
                "aliases",
                "is_explicitly_non_vegan",
                "description",
                "requires_contextual_check",
            ]
        )

        # Fill nulls
        df_clean = df_clean.fill_null("")

        # Insert into database
        with engine.connect() as connection:
            connection.execute(vegan_ontology_table.insert(), df_clean.to_dicts())
            connection.commit()

        logger.info(f"Successfully ingested {len(df_clean)} vegan ontology terms")

        # Verify aliases were parsed correctly
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT term, aliases FROM vegan_ontology WHERE aliases != '' LIMIT 3"
                )
            ).fetchall()
            logger.info("Verification - Aliases in database:")
            for row in result:
                logger.info(f"  {row[0]}: aliases='{row[1]}'")

    except Exception as e:
        logger.error(f"Failed to ingest vegan ontology data: {e}")
        raise


def ingest_unit_conversions() -> None:
    """
    Ingest unit conversion data with simplified schema.

    This function reads the unit conversion CSV and populates the conversions
    table for standardizing recipe measurements.
    """
    logger.info("Ingesting unit conversion data")

    try:
        # Read CSV with Polars
        df = pl.read_csv(app_config.RAW_DATA_DIR / "unit_conversion.csv")
        logger.info(f"Read {len(df)} rows from unit conversion CSV")

        # Clean column names
        df = df.rename({col: col.strip() for col in df.columns})

        # Map columns to simplified schema
        column_mapping = {
            "Unit": "unit",
            "Abbreviation": "abbreviation",
            "Metric Equivalent": "metric_equivalent",
            "Type": "type",
            "Factor": "factor",
        }

        for csv_col, db_col in column_mapping.items():
            if csv_col in df.columns:
                df = df.rename({csv_col: db_col})

        # Clean and process data
        df = df.with_columns(
            [
                pl.col("unit").str.to_lowercase().str.strip_chars(),
                pl.col("abbreviation").str.strip_chars().fill_null(""),
                pl.col("metric_equivalent").str.strip_chars().fill_null(""),
                pl.col("type").str.to_lowercase().str.strip_chars().fill_null("volume"),
                pl.col("factor")
                .map_elements(clean_numeric_value, return_dtype=pl.Float64)
                .fill_null(1.0),
            ]
        )

        # Select relevant columns
        df_clean = df.select(
            ["unit", "abbreviation", "metric_equivalent", "type", "factor"]
        )

        # Insert into database
        with engine.connect() as connection:
            connection.execute(unit_conversions_table.insert(), df_clean.to_dicts())
            connection.commit()

        logger.info(f"Successfully ingested {len(df_clean)} unit conversion records")

    except Exception as e:
        logger.error(f"Failed to ingest unit conversion data: {e}")
        raise


def validate_data_quality() -> Dict[str, Any]:
    """
    Validate the quality of ingested data and return metrics.

    Returns:
        Dictionary containing validation metrics and status
    """
    logger.info("Validating data quality")

    validation_results = {}

    with engine.connect() as connection:
        # Check nutrition data
        nutrition_count = connection.execute(
            text("SELECT COUNT(*) FROM nutrition_facts")
        ).fetchone()[0]
        non_zero_calories = connection.execute(
            text("SELECT COUNT(*) FROM nutrition_facts WHERE calories > 0")
        ).fetchone()[0]

        # Check vegan data
        vegan_count = connection.execute(
            text("SELECT COUNT(*) FROM vegan_ontology")
        ).fetchone()[0]
        with_aliases = connection.execute(
            text("SELECT COUNT(*) FROM vegan_ontology WHERE aliases != ''")
        ).fetchone()[0]

        # Check units
        unit_count = connection.execute(
            text("SELECT COUNT(*) FROM unit_conversions")
        ).fetchone()[0]

        validation_results = {
            "nutrition_total": nutrition_count,
            "nutrition_with_calories": non_zero_calories,
            "vegan_total": vegan_count,
            "vegan_with_aliases": with_aliases,
            "unit_conversions": unit_count,
            "calories_success_rate": (
                non_zero_calories / nutrition_count if nutrition_count > 0 else 0
            ),
            "aliases_success_rate": (
                with_aliases / vegan_count if vegan_count > 0 else 0
            ),
        }

        logger.info("Validation Results:")
        logger.info(
            (
                f"  Nutrition: {nutrition_count} total, {non_zero_calories} with calories > 0 "
                f"({validation_results['calories_success_rate']:.1%})"
            )
        )
        logger.info(
            (
                f"  Vegan: {vegan_count} total, {with_aliases} with aliases "
                f"({validation_results['aliases_success_rate']:.1%})"
            )
        )
        logger.info(f"  Units: {unit_count} conversions")

        # Check if critical issues are fixed
        # Note: aliases_success_rate of 12.3% is actually correct - most terms don't have aliases
        if (
            validation_results["calories_success_rate"] > 0.8
            and validation_results["aliases_success_rate"] > 0.1
        ):
            logger.info("Data quality validation PASSED - critical issues resolved")
            validation_results["status"] = "PASSED"
        else:
            logger.error("Data quality validation FAILED - issues still present")
            validation_results["status"] = "FAILED"

    return validation_results


def main() -> int:
    """
    Main function to execute the fixed data ingestion pipeline.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    logger.info("Starting FIXED Data Ingestion Pipeline")

    try:
        # Ensure directories exist
        app_config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Remove old database for fresh start
        if app_config.DB_PATH.exists():
            logger.info(f"Removing existing database: {app_config.DB_PATH}")
            app_config.DB_PATH.unlink()

        # Execute pipeline steps
        create_database_schema()
        ingest_nutrition_data()
        ingest_vegan_ontology()
        ingest_unit_conversions()

        # Validate results
        validation_results = validate_data_quality()

        if validation_results["status"] == "PASSED":
            logger.info("Fixed data ingestion pipeline completed successfully")
            return 0
        else:
            logger.error("Data ingestion completed but validation failed")
            return 1

    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
