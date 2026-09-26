"""
Query Engine for Translating Structured JSON to SQL

This module provides a secure and robust system for translating structured
JSON queries into executable SQL statements. It serves as a bridge between
function-calling LLMs and database operations, ensuring type safety and
preventing SQL injection attacks.

The module implements a two-stage validation process:
1. Structure validation: Ensures required fields are present
2. Content validation: Validates operators, table names, and parameter types

Key Features:
- SQL injection prevention through parameterized queries and whitelisting
- Support for complex WHERE clauses with multiple conditions
- Comprehensive error handling with descriptive messages
- Type-safe parameter binding
"""

import logging
from typing import Any, Dict, List, Optional

# Configure professional logging
logger = logging.getLogger(__name__)

# --- Security Whitelists ---
# Prevents injection by ensuring only known tables and columns are used.
ALLOWED_TABLES = {
    "nutrition_facts": [
        "name",
        "carbohydrate_g",
        "fiber_g",
        "protein_g",
        "total_fat_g",
        "*",
    ],
    "vegan_ontology": [
        "term",
        "aliases",
        "is_explicitly_non_vegan",
        "description",
        "*",
    ],
}

ALLOWED_OPERATORS = ["=", "!=", ">", "<", ">=", "<=", "LIKE", "IN"]
ALLOWED_AGGREGATIONS = ["AVG", "COUNT", "SUM", "MIN", "MAX"]


class QueryEngine:
    """
    Query Engine for translating structured JSON to SQL.

    This class provides a secure interface for converting high-level JSON
    query specifications into executable SQL statements with proper
    parameter binding to prevent SQL injection attacks.
    """

    def __init__(self):
        """Initialize the QueryEngine."""
        logger.info("QueryEngine initialized successfully")

    def translate_json_to_sql(self, query: Dict[str, Any]) -> tuple[str, List[Any]]:
        """
        Translate a structured JSON query object into an executable SQL query.
        """
        return translate_json_to_sql(query)

    def validate_query(self, query: Dict[str, Any]) -> bool:
        """
        Validate a JSON query structure without executing it.
        """
        return _validate_query_structure(query)


def _validate_query_structure(query: Dict[str, Any]) -> bool:
    """
    Validate the basic structure of the incoming JSON query.
    """
    required_keys = ["operation", "table"]
    if not all(key in query for key in required_keys):
        raise ValueError("Query must contain 'operation' and 'table' keys")

    if query["operation"] not in ["search", "aggregate"]:
        raise ValueError(
            "Invalid operation specified. Must be 'search' or 'aggregate'."
        )

    table = query["table"]
    if table not in ALLOWED_TABLES:
        raise ValueError(
            f"Disallowed table: '{table}'. Must be one of {list(ALLOWED_TABLES.keys())}"
        )

    return True


def _build_where_clause(
    filters: Optional[List[Dict[str, Any]]], table_name: str
) -> tuple[str, List[Any]]:
    """
    Build the WHERE clause and parameters from a list of filters.
    """
    if not filters:
        return "", []

    conditions = []
    params = []
    allowed_columns = ALLOWED_TABLES[table_name]

    for f in filters:
        field = f.get("field")
        operator = f.get("operator", "=").upper()
        value = f.get("value")

        if not field or value is None:
            continue

        # Security: Validate field and operator against whitelists
        if field not in allowed_columns:
            raise ValueError(f"Disallowed field '{field}' for table '{table_name}'")
        if operator not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator}")

        if operator == "IN":
            if not isinstance(value, list):
                raise ValueError("IN operator requires a list value")
            placeholders = ", ".join(["?"] * len(value))
            conditions.append(f"{field} IN ({placeholders})")
            params.extend(value)
        else:
            conditions.append(f"{field} {operator} ?")
            params.append(value)

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    return where_clause, params


def translate_json_to_sql(query: Dict[str, Any]) -> tuple[str, List[Any]]:
    """
    Translate a structured JSON query object into an executable SQL query.
    """
    _validate_query_structure(query)

    table = query["table"]
    operation = query["operation"]
    params = []
    allowed_columns = ALLOWED_TABLES[table]

    if operation == "search":
        select_fields = query.get("select_fields", ["*"])

        # Security: Validate selected fields
        for field in select_fields:
            if field != "*" and field not in allowed_columns:
                raise ValueError(f"Disallowed field '{field}' for table '{table}'")

        fields_str = ", ".join(select_fields)

        filters = query.get("filters")
        where_clause, where_params = _build_where_clause(filters, table)
        params.extend(where_params)

        limit = query.get("limit")
        limit_clause = ""
        if limit is not None:
            try:
                limit_clause = f" LIMIT {int(limit)}"
            except (ValueError, TypeError):
                raise ValueError("LIMIT must be an integer.")

        order_by_clause = ""
        if "order_by" in query:
            # Basic ORDER BY support for safety
            order_field = query["order_by"].get("field")
            order_direction = query["order_by"].get("direction", "ASC").upper()
            if order_field in allowed_columns and order_direction in ["ASC", "DESC"]:
                order_by_clause = f" ORDER BY {order_field} {order_direction}"

        sql = f"SELECT {fields_str} FROM {table}{where_clause}{order_by_clause}{limit_clause};"

    elif operation == "aggregate":
        aggregation_field = query.get("aggregation_field")
        aggregation_type = query.get("aggregation_type", "COUNT").upper()

        if not aggregation_field:
            raise ValueError("Aggregation operation requires 'aggregation_field'")

        # Security: Validate aggregation field and type
        if aggregation_field != "*" and aggregation_field not in allowed_columns:
            raise ValueError(
                f"Disallowed aggregation field '{aggregation_field}' for table '{table}'"
            )
        if aggregation_type not in ALLOWED_AGGREGATIONS:
            raise ValueError(f"Disallowed aggregation type: {aggregation_type}")

        filters = query.get("filters")
        where_clause, where_params = _build_where_clause(filters, table)
        params.extend(where_params)

        group_by = query.get("groupBy")
        group_by_clause = ""
        if group_by:
            if group_by in allowed_columns:
                group_by_clause = f" GROUP BY {group_by}"
            else:
                raise ValueError(f"Disallowed GROUP BY field: {group_by}")

                sql = (
                    f"SELECT {aggregation_type}({aggregation_field}) FROM "
                    f"{table}{where_clause}{group_by_clause};"
                )

    else:
        raise ValueError(f"Unsupported operation type: {operation}")

    return sql, params
