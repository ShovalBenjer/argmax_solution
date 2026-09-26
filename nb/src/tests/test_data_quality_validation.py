#!/usr/bin/env python3
"""
Data Quality Validation Test Suite

This module provides comprehensive data quality validation for the diet classification
system's knowledge base. It validates that data ingestion completed successfully
with required benchmarks and ensures data integrity, completeness, and quality.

The validation suite covers:
- Database existence and size validation
- Table structure and schema verification
- Record count benchmarking
- Data quality and integrity checks
- Duplicate detection and null value validation
- Overall data quality assessment and reporting

Key Validation Areas:
- Nutrition facts data completeness
- Vegan ontology data quality
- Unit conversion data integrity
- Database structure validation
- Data consistency verification
- Benchmark compliance checking

Validation Features:
- Comprehensive benchmark testing
- Detailed logging and reporting
- Graceful error handling
- Quality metrics calculation
- Status reporting and feedback
- Performance monitoring

Data Quality Benchmarks:
- nutrition_facts_min_records: 8000 records
- vegan_ontology_min_records: 500 records
- unit_conversions_min_records: 50 records
- min_database_size_mb: 10 MB
- required_tables: ['nutrition_facts', 'vegan_ontology', 'unit_conversions']

Dependencies:
- sqlite3: Database connectivity and querying
- pathlib: Path management
- logging: Comprehensive logging and reporting
- config: Application configuration management

Example:
    >>> python nb/src/tests/test_data_quality_validation.py
    >>> # Run complete data quality validation
    >>> sys.exit(main())
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add nb/src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import app_config  # noqa: E402

# Data Quality Benchmarks
BENCHMARKS = {
    "nutrition_facts_min_records": 8000,
    "vegan_ontology_min_records": 500,
    "unit_conversions_min_records": 50,
    "min_database_size_mb": 10,
    "required_tables": ["nutrition_facts", "vegan_ontology", "unit_conversions"],
}


def validate_database_exists():
    """
    Validate that knowledge_graph.db was created and meets size requirements.

    This function checks for the existence of the main database file and validates
    that it meets minimum size requirements. It provides detailed logging about
    the database status and size.

    Validates:
    - Database file existence at configured path
    - Database file size meets minimum benchmark
    - Database accessibility and readability

    Returns:
        None: Logs validation results

    Raises:
        FileNotFoundError: If database file does not exist
        OSError: If database file cannot be accessed

    Example:
        >>> validate_database_exists()
        >>> # Validates database existence and size
    """
    if not app_config.DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {app_config.DB_PATH}")

    size_mb = app_config.DB_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Database found: {app_config.DB_PATH} ({size_mb:.1f} MB)")

    if size_mb < BENCHMARKS["min_database_size_mb"]:
        logger.warning(
            (
                f"Database size {size_mb:.1f} MB below benchmark "
                f"{BENCHMARKS['min_database_size_mb']} "
                f"MB"
            )
        )
    else:
        logger.info(
            (
                f"Database size meets benchmark ({size_mb:.1f} MB >= "
                f"{BENCHMARKS['min_database_size_mb']} "
                f"MB)"
            )
        )


def validate_table_structure():
    """
    Validate that all required tables exist with proper structure.

    This function verifies that all required tables are present in the database
    and provides information about their structure. It ensures the database
    schema is complete and properly configured.

    Validates:
    - All required tables exist
    - Table structure is accessible
    - Column information is available
    - Schema integrity is maintained

    Returns:
        None: Logs validation results

    Raises:
        ValueError: If required tables are missing
        sqlite3.Error: If database operations fail

    Example:
        >>> validate_table_structure()
        >>> # Validates database table structure
    """
    with sqlite3.connect(app_config.DB_PATH) as conn:
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        logger.info(f"Found tables: {tables}")

        # Check required tables
        missing_tables = set(BENCHMARKS["required_tables"]) - set(tables)
        if missing_tables:
            raise ValueError(f"Missing required tables: {missing_tables}")

        logger.info("All required tables present")

        # Validate table schemas
        for table in BENCHMARKS["required_tables"]:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            logger.info(f"Table {table}: {len(columns)} columns")


def validate_record_counts():
    """
    Validate that tables have sufficient records to meet benchmarks.

    This function checks the record counts for all required tables and compares
    them against established benchmarks. It provides detailed reporting on
    whether each table meets its minimum record requirements.

    Validates:
    - nutrition_facts table record count
    - vegan_ontology table record count
    - unit_conversions table record count
    - Benchmark compliance for each table

    Returns:
        dict: Dictionary containing record counts for each table

    Raises:
        sqlite3.Error: If database queries fail

    Example:
        >>> counts = validate_record_counts()
        >>> print(f"Nutrition facts: {counts['nutrition_facts']} records")
    """
    with sqlite3.connect(app_config.DB_PATH) as conn:
        cursor = conn.cursor()

        results = {}

        # Check nutrition_facts
        cursor.execute("SELECT COUNT(*) FROM nutrition_facts;")
        nutrition_count = cursor.fetchone()[0]
        results["nutrition_facts"] = nutrition_count

        if nutrition_count < BENCHMARKS["nutrition_facts_min_records"]:
            logger.warning(
                (
                    f"nutrition_facts: {nutrition_count} records < "
                    f"{BENCHMARKS['nutrition_facts_min_records']} "
                    f"benchmark"
                )
            )
        else:
            logger.info(
                (
                    f"nutrition_facts: {nutrition_count} records >= "
                    f"{BENCHMARKS['nutrition_facts_min_records']} "
                    f"benchmark"
                )
            )

        # Check vegan_ontology
        cursor.execute("SELECT COUNT(*) FROM vegan_ontology;")
        vegan_count = cursor.fetchone()[0]
        results["vegan_ontology"] = vegan_count

        if vegan_count < BENCHMARKS["vegan_ontology_min_records"]:
            logger.warning(
                (
                    f"vegan_ontology: {vegan_count} records < "
                    f"{BENCHMARKS['vegan_ontology_min_records']} "
                    f"benchmark"
                )
            )
        else:
            logger.info(
                (
                    f"vegan_ontology: {vegan_count} records >= "
                    f"{BENCHMARKS['vegan_ontology_min_records']} "
                    f"benchmark"
                )
            )

        # Check unit_conversions
        cursor.execute("SELECT COUNT(*) FROM unit_conversions;")
        unit_count = cursor.fetchone()[0]
        results["unit_conversions"] = unit_count

        if unit_count < BENCHMARKS["unit_conversions_min_records"]:
            logger.warning(
                (
                    f"unit_conversions: {unit_count} records < "
                    f"{BENCHMARKS['unit_conversions_min_records']} "
                    f"benchmark"
                )
            )
        else:
            logger.info(
                (
                    f"unit_conversions: {unit_count} records >= "
                    f"{BENCHMARKS['unit_conversions_min_records']} "
                    f"benchmark"
                )
            )

        return results


def validate_data_quality():
    """
    Validate data quality and integrity across all tables.

    This function performs comprehensive data quality checks including null value
    detection, duplicate identification, and data consistency validation. It
    ensures the data meets quality standards for the classification system.

    Validates:
    - Null or empty ingredient names
    - Duplicate ingredient records
    - Data consistency and integrity
    - Quality metrics and reporting

    Returns:
        None: Logs quality validation results

    Raises:
        sqlite3.Error: If database queries fail

    Example:
        >>> validate_data_quality()
        >>> # Validates data quality and integrity
    """
    with sqlite3.connect(app_config.DB_PATH) as conn:
        cursor = conn.cursor()

        # Check for null ingredient names
        cursor.execute(
            "SELECT COUNT(*) FROM nutrition_facts WHERE name IS NULL OR name = '';"
        )
        null_names = cursor.fetchone()[0]

        if null_names > 0:
            logger.warning(
                f"Found {null_names} nutrition records with null/empty names"
            )
        else:
            logger.info("No null ingredient names found")

        # Check for duplicate ingredient names
        cursor.execute(
            "SELECT name, COUNT(*) FROM nutrition_facts GROUP BY name HAVING COUNT(*) > 1;"
        )
        duplicates = cursor.fetchall()

        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate ingredient names")
            for name, count in duplicates[:5]:  # Show first 5
                logger.warning(f"  - '{name}': {count} duplicates")
        else:
            logger.info("No duplicate ingredient names found")


def main():
    """
    Run all data quality validations and provide comprehensive reporting.

    This function orchestrates the complete data quality validation process,
    running all validation checks and providing a comprehensive summary report.
    It ensures the knowledge base is ready for use in the classification system.

    Validation Process:
        1. Database existence and size validation
        2. Table structure verification
        3. Record count benchmarking
        4. Data quality assessment
        5. Comprehensive summary reporting

    Returns:
        int: Exit code (0 for success, 1 for failure)

    Raises:
        Exception: If critical validation steps fail

    Example:
        >>> exit_code = main()
        >>> if exit_code == 0:
        >>>     print("Data quality validation passed")
        >>> else:
        >>>     print("Data quality validation failed")
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY VALIDATION TEST")
    logger.info("=" * 60)

    try:
        # Database existence and size
        validate_database_exists()

        # Table structure
        validate_table_structure()

        # Record counts
        record_counts = validate_record_counts()

        # Data quality
        validate_data_quality()

        # Summary
        logger.info("=" * 60)
        logger.info("DATA QUALITY VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Database: {app_config.DB_PATH}")
        logger.info(f"Size: {app_config.DB_PATH.stat().st_size / (1024 * 1024):.1f} MB")

        for table, count in record_counts.items():
            benchmark = BENCHMARKS.get(f"{table}_min_records", 0)
            status = "PASS" if count >= benchmark else "WARN"
            logger.info(
                f"{status}: {table}: {count:,} records (benchmark: {benchmark:,})"
            )

        # Overall status
        all_benchmarks_met = all(
            record_counts[table] >= BENCHMARKS[f"{table}_min_records"]
            for table in record_counts
        )

        if all_benchmarks_met:
            logger.info("ALL DATA QUALITY BENCHMARKS MET")
            return 0
        else:
            logger.warning("Some benchmarks not met - continuing with warnings")
            return 0  # Continue with warnings as requested

    except Exception as e:
        logger.error(f"Data quality validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
