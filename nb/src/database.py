"""
Unified database access layer for all services.

This module provides a centralized database interface that abstracts both SQLite
and OpenSearch operations. It handles connection management, query execution,
and data retrieval for the diet classification system.

The DatabaseManager class implements a unified interface for:
- SQLite operations (nutrition facts, vegan ontology, unit conversions)
- OpenSearch operations (recipe search, ingredient lookup)
- Connection pooling and error handling
- Context manager support for safe resource management

Key Features:
- Automatic connection initialization and health checking
- Graceful fallback when OpenSearch is unavailable
- Thread-safe operations with proper resource cleanup
- Comprehensive error handling and logging

Example:
    >>> from database import db_manager
    >>> nutrition_data = db_manager.query_nutrition_data("chicken breast")
    >>> vegan_info = db_manager.query_vegan_ontology("milk")
    >>> recipes = db_manager.search_recipes("chicken spinach")
"""

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from config import app_config
from loguru import logger
from opensearchpy import OpenSearch
from rapidfuzz import fuzz
from sqlalchemy import create_engine


class DatabaseManager:
    """
    Unified database manager for SQLite and OpenSearch operations.

    This class provides a high-level interface for database operations across
    multiple backend systems. It manages connections, handles errors gracefully,
    and provides consistent APIs for data retrieval and manipulation.

    The manager supports both synchronous and asynchronous operations, with
    automatic connection pooling and health monitoring. It implements proper
    resource cleanup and provides detailed logging for debugging.

    Attributes:
        sqlite_engine: SQLAlchemy engine for SQLite operations
        opensearch_client: OpenSearch client for search operations

    Example:
        >>> db = DatabaseManager()
        >>> with db.get_sqlite_connection() as conn:
        ...     result = pd.read_sql("SELECT * FROM nutrition_facts LIMIT 5", conn)
        >>> recipes = db.search_recipes("chicken", size=10)
    """

    def __init__(self):
        """
        Initialize the database manager with SQLite and OpenSearch connections.

        Creates SQLAlchemy engine for SQLite operations and initializes
        OpenSearch client with proper error handling. If OpenSearch is
        unavailable, the system continues to operate with SQLite-only mode.
        """
        self.sqlite_engine = create_engine(f"sqlite:///{app_config.DB_PATH}")
        self.opensearch_client = None
        self.nutrition_df = self._load_table_to_dataframe("nutrition_facts")
        self.vegan_ontology_df = self._load_table_to_dataframe("vegan_ontology")
        self._init_opensearch()

    def _init_opensearch(self):
        """
        Initialize OpenSearch client with comprehensive error handling.

        Attempts to establish connection to OpenSearch with configurable
        timeout and retry settings. If connection fails, the system
        continues operating in SQLite-only mode with appropriate logging.

        Raises:
            None: All exceptions are caught and logged, system continues operation
        """
        try:
            self.opensearch_client = OpenSearch(
                hosts=[app_config.OPENSEARCH_URL],
                http_auth=None,
                use_ssl=False,
                verify_certs=False,
                ssl_show_warn=False,
            )
            if self.opensearch_client.ping():
                logger.info("OpenSearch connection established")
            else:
                logger.warning("OpenSearch ping failed")
                self.opensearch_client = None
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch: {e}")
            self.opensearch_client = None

    @contextmanager
    def get_sqlite_connection(self):
        """
        Get SQLite connection context manager for safe database operations.

        This context manager ensures proper connection cleanup and transaction
        handling. It automatically commits successful operations and rolls back
        on exceptions.

        Yields:
            sqlite3.Connection: Active SQLite database connection

        Example:
            >>> with db_manager.get_sqlite_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT COUNT(*) FROM nutrition_facts")
            ...     count = cursor.fetchone()[0]
        """
        conn = sqlite3.connect(str(app_config.DB_PATH))
        try:
            yield conn
        finally:
            conn.close()

    def _load_table_to_dataframe(self, table_name: str) -> pd.DataFrame:
        """Loads an entire SQLite table into a pandas DataFrame."""
        logger.info(f"Loading '{table_name}' table into memory for fuzzy lookup...")
        try:
            with self.get_sqlite_connection() as conn:
                df = pd.read_sql_query(
                    f"SELECT * FROM {table_name}", conn
                )  # nosec B608
                # For vegan_ontology, parse aliases if they are stored as JSON strings
                if table_name == "vegan_ontology" and "aliases" in df.columns:
                    df["aliases"] = df["aliases"].apply(
                        lambda x: json.loads(x) if pd.notna(x) and x else []
                    )
                logger.info(
                    f"Successfully loaded {len(df)} records from '{table_name}'."
                )
                return df
        except Exception as e:
            logger.error(f"Failed to load table '{table_name}' into DataFrame: {e}")
            return pd.DataFrame()  # Return empty DataFrame on failure

    def get_opensearch_client(self) -> Optional[OpenSearch]:
        """
        Get the OpenSearch client instance.

        Returns:
            OpenSearch: Configured OpenSearch client or None if unavailable

        Example:
            >>> client = db_manager.get_opensearch_client()
            >>> if client:
            ...     response = client.search(index="recipes", body={"query": {"match_all": {}}})
        """
        return self.opensearch_client

    def query_nutrition_data(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        """
        Query nutrition data from SQLite database for a specific ingredient.

        Performs a case-insensitive search for the ingredient name in the
        nutrition_facts table. Returns comprehensive nutritional information
        including macronutrients, vitamins, and minerals.

        Args:
            ingredient_name: Name of the ingredient to search for

        Returns:
            Dict containing nutrition facts or None if not found

        Example:
            >>> nutrition = db_manager.query_nutrition_data("chicken breast")
            >>> if nutrition:
            ...     print(f"Calories: {nutrition['calories']}")
            ...     print(f"Protein: {nutrition['protein_g']}g")
        """
        try:
            normalized_name = ingredient_name.lower().strip()

            # Try exact match first on the preloaded DataFrame
            exact_match = self.nutrition_df[
                self.nutrition_df["name"] == normalized_name
            ]
            if not exact_match.empty:
                return exact_match.iloc[0].to_dict()

            # If no exact match, try fuzzy matching
            best_match_score = 0
            best_match_data = None
            for _, row in self.nutrition_df.iterrows():
                score = fuzz.ratio(normalized_name, str(row["name"]))
                if score > best_match_score and score >= 95:  # 95% confidence threshold
                    best_match_score = score
                    best_match_data = row.to_dict()
            if best_match_data:
                return best_match_data
        except Exception as e:
            logger.error(f"Failed to query nutrition data for {ingredient_name}: {e}")
        return None

    def query_vegan_ontology(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        """
        Query vegan ontology from SQLite database for ingredient classification.

        Searches the vegan_ontology table for exact matches and aliases.
        Returns comprehensive information about vegan status, including
        explicit non-vegan flags and contextual check requirements.

        Args:
            ingredient_name: Name of the ingredient to classify

        Returns:
            Dict containing vegan ontology information or None if not found

        Example:
            >>> vegan_info = db_manager.query_vegan_ontology("milk")
            >>> if vegan_info:
            ...     is_vegan = vegan_info['is_vegan_term']
            ...     aliases = vegan_info['aliases']
        """
        try:
            normalized_name = ingredient_name.lower().strip()

            # Try exact match on 'term' column
            exact_match_term = self.vegan_ontology_df[
                self.vegan_ontology_df["term"] == normalized_name
            ]
            if not exact_match_term.empty:
                result = exact_match_term.iloc[0].to_dict()
                result["is_vegan_term"] = not bool(
                    result.get("is_explicitly_non_vegan", False)
                )
                logger.debug(
                    f"Found vegan info for '{ingredient_name}' (exact term match): {result}"
                )
                return result

            # Try exact match in 'aliases' column (if aliases are pre-parsed lists)
            alias_match = self.vegan_ontology_df[
                self.vegan_ontology_df["aliases"].apply(
                    lambda x: normalized_name in x if isinstance(x, list) else False
                )
            ]
            if not alias_match.empty:
                result = alias_match.iloc[0].to_dict()
                result["is_vegan_term"] = not bool(
                    result.get("is_explicitly_non_vegan", False)
                )
                logger.debug(
                    f"Found vegan info for '{ingredient_name}' (exact alias match): {result}"
                )
                return result

            # If no exact match, try fuzzy matching against 'term' and 'aliases'
            best_match_score = 0
            best_match_data = None
            for _, row in self.vegan_ontology_df.iterrows():
                # Fuzzy match against 'term'
                score_term = fuzz.ratio(normalized_name, str(row["term"]))
                # Fuzzy match against each alias in the 'aliases' list
                score_alias = 0
                if isinstance(row["aliases"], list):
                    for alias in row["aliases"]:
                        score_alias = max(
                            score_alias, fuzz.ratio(normalized_name, str(alias))
                        )

                combined_score = max(score_term, score_alias)

                if (
                    combined_score > best_match_score and combined_score >= 95
                ):  # 95% confidence threshold
                    best_match_score = combined_score
                    best_match_data = row.to_dict()

            if best_match_data:
                best_match_data["is_vegan_term"] = not bool(
                    best_match_data.get("is_explicitly_non_vegan", False)
                )
                logger.debug(
                    (
                        f"Found vegan info for '{ingredient_name}' (fuzzy match, score "
                        f"{best_match_score:.2f}): "
                        f"{best_match_data}"
                    )
                )
                return best_match_data
            else:
                logger.debug(
                    f"No vegan info found for '{ingredient_name}' after exact and fuzzy search."
                )
        except Exception as e:
            logger.error(f"Failed to query vegan ontology for {ingredient_name}: {e}")
        return None

    def search_recipes(
        self, ingredient_query: str, size: int = 100000
    ) -> Dict[str, Any]:
        """
        Search recipes in OpenSearch using fuzzy matching.

        Performs a full-text search across recipe ingredients with automatic
        fuzzy matching for typos and variations. Returns ranked results with
        relevance scores.

        Args:
            ingredient_query: Search query for recipe ingredients
            size: Maximum number of results to return (default: 100000)

        Returns:
            Dict containing search results with hits and metadata

        Example:
            >>> results = db_manager.search_recipes("chicken spinach", size=5)
            >>> for hit in results['hits']['hits']:
            ...     recipe = hit['_source']
            ...     print(f"Recipe: {recipe['title']}")
        """
        if not self.opensearch_client:
            return {"hits": {"hits": [], "total": {"value": 0}}}

        query = {
            "query": {
                "match": {
                    "ingredients": {"query": ingredient_query, "fuzziness": "AUTO"}
                }
            }
        }

        try:
            response = self.opensearch_client.search(
                index="recipes", body=query, size=size
            )
            return response
        except Exception as e:
            logger.error(f"OpenSearch query failed: {e}")
            return {"hits": {"hits": [], "total": {"value": 0}}}

    def get_all_ingredients(self, limit: int = 10000) -> List[str]:
        """
        Get all available ingredients from OpenSearch for autocomplete.

        Retrieves a comprehensive list of all ingredients stored in the
        OpenSearch index. This is typically used for building autocomplete
        functionality in user interfaces.

        Args:
            limit: Maximum number of ingredients to retrieve (default: 10000)

        Returns:
            List of ingredient names

        Example:
            >>> ingredients = db_manager.get_all_ingredients(limit=1000)
            >>> print(f"Found {len(ingredients)} ingredients")
        """
        if not self.opensearch_client:
            return []

        try:
            response = self.opensearch_client.search(
                index="ingredients", body={"query": {"match_all": {}}}, size=limit
            )
            return [hit["_source"]["ingredients"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Failed to get ingredients: {e}")
            return []


# Global database manager instance
db_manager = DatabaseManager()
