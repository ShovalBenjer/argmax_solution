"""
Function-Calling Handler for Query Generation

This module implements a sophisticated natural language to structured query
translation system using function-calling LLMs. It bridges the gap between
user-friendly natural language questions and precise database operations.

The FunctionCallingHandler class provides:
- Natural language question parsing
- Structured JSON query generation
- Database schema context integration
- LLM-based query optimization
- Comprehensive error handling

The system uses a predefined JSON schema that maps to SQL operations,
ensuring consistent and safe database queries while maintaining the
flexibility of natural language input.

Key Features:
- Schema-driven query generation
- Context-aware database understanding
- Robust error handling and fallback
- Async support for high-throughput scenarios
- Comprehensive logging and debugging

Example:
    >>> handler = FunctionCallingHandler()
    >>> question = "What are the calories for chicken breast?"
    >>> json_query = await handler.generate_json_query(question)
    >>> print(json_query)
    {'operation': 'search', 'table': 'nutrition_facts', ...}
"""

import json
import logging
from typing import Any, Dict

from llm_client import LLMClient

logger = logging.getLogger(__name__)


class FunctionCallingHandler:
    """
    Handles translation of natural language questions to structured JSON queries.

    This class implements a sophisticated pipeline that converts user-friendly
    natural language questions into precise, structured JSON queries that can
    be safely executed against the database. It uses LLM-based function calling
    with comprehensive schema context to ensure accurate translations.

    The handler maintains a detailed understanding of the database schema and
    provides the LLM with sufficient context to generate appropriate queries.
    It includes robust error handling and supports both simple and complex
    query patterns.

    Attributes:
        llm_client: LLM client for query generation
        schema_context: Database schema context for LLM prompts
        json_schema: Structured JSON schema definition

    Example:
        >>> handler = FunctionCallingHandler()
        >>> question = "Find high-protein foods with less than 10g carbs"
        >>> result = await handler.generate_json_query(question)
        >>> if "error" not in result:
        ...     print(f"Generated query: {result['operation']} on {result['table']}")
    """

    def __init__(self):
        """
        Initialize the function calling handler with LLM client and schema context.

        Sets up the LLM client for query generation and builds comprehensive
        database schema context that will be provided to the LLM for accurate
        query generation.
        """
        self.llm_client = LLMClient()
        self.schema_context = self._build_schema_context()
        self.json_schema = self._get_json_schema()

    def _get_json_schema(self) -> str:
        """
        Define the structured JSON schema for LLM query generation.

        Returns a comprehensive JSON schema that defines the structure
        and constraints for generated queries. This schema ensures that
        LLM outputs are consistent and can be safely converted to SQL.

        Returns:
            str: JSON string containing the query schema definition

        Example:
            >>> schema = handler._get_json_schema()
            >>> print(schema)
            '{"operation": "search | aggregate", ...}'
        """
        schema = {
            "operation": "search | aggregate",
            "table": "nutrition_facts | vegan_ontology | unit_conversions",
            "select_fields": ["field1", "field2"],
            "filters": [
                {
                    "field": "column_name",
                    "operator": "= | != | > | < | >= | <= | LIKE | IN",
                    "value": "string | number | list",
                }
            ],
            "aggregation_type": "AVG | COUNT | SUM | MIN | MAX",
            "aggregation_field": "column_name",
            "groupBy": "column_name",
            "limit": "integer",
        }
        return json.dumps(schema, indent=2)

    def _build_schema_context(self) -> str:
        """
        Build comprehensive database schema context for LLM prompts.

        Creates a detailed description of the database schema that helps
        the LLM understand the available tables, columns, and their purposes.
        This context is crucial for generating accurate and meaningful queries.

        Returns:
            str: Formatted schema context for LLM prompts

        Example:
            >>> context = handler._build_schema_context()
            >>> print("nutrition_facts" in context)
            True
        """
        context = """
DATABASE SCHEMA CONTEXT:

1. `nutrition_facts` table:
   - Columns: `name`, `calories`, `carbohydrate_g`, `fiber_g`, `protein_g`,
     `total_fat_g`, `cholesterol_mg`, `vitamin_b12_mcg`
   - Description: Contains nutritional information per 100g for various food items.

2. `vegan_ontology` table:
   - Columns: `term`, `aliases`, `is_explicitly_non_vegan`, `description`
   - Description: A knowledge base of vegan and non-vegan terms.
     `is_explicitly_non_vegan` is a boolean.

3. `unit_conversions` table:
   - Columns: `unit`, `factor`, `type`
   - Description: Converts common cooking units to grams.
"""
        return context

    async def generate_json_query(self, question: str) -> Dict[str, Any]:
        """
        Generate a structured JSON query from a natural language question.

        This method implements the core translation logic, converting user
        questions into structured JSON queries that can be safely executed
        against the database. It uses LLM-based function calling with
        comprehensive schema context to ensure accurate translations.

        The method includes robust error handling and provides detailed
        logging for debugging query generation issues.

        Args:
            question: Natural language question to convert to structured query

        Returns:
            Dict containing the generated JSON query or error information

        Raises:
            ValueError: If LLM returns an error response
            TypeError: If LLM returns unexpected data type
            json.JSONDecodeError: If LLM response cannot be parsed as JSON

        Example:
            >>> question = "What are the calories for chicken breast?"
            >>> result = await handler.generate_json_query(question)
            >>> if "error" not in result:
            ...     print(f"Operation: {result['operation']}")
            ...     print(f"Table: {result['table']}")
            ... else:
            ...     print(f"Error: {result['error']}")
        """
        prompt = f"""You are an expert at converting user questions into structured JSON queries.
Your task is to populate the following JSON schema to answer the user's question.

DATABASE SCHEMA:
{self.schema_context}

JSON QUERY FORMAT:
{self.json_schema}

User Question: "{question}"

Based on the user's question, generate a JSON object that matches the specified format.
Ensure all values are valid JSON. For `LIKE` operations, include '%' wildcards in the value.
For `IN` operations, the value must be a list of strings or numbers.
Respond with ONLY the generated JSON object, nothing else.
"""
        model_name = "arctic-text2sql:latest"

        if not self.llm_client.is_model_available(model_name):
            logger.error(f"Model '{model_name}' is not available for function calling.")
            return {"error": f"Model not available: {model_name}"}

        try:
            result = await self.llm_client.query_async(
                model_name, prompt, as_json=True, timeout=20.0
            )

            if "error" in result:
                raise ValueError(result.get("error", "Unknown LLM error"))

            # The result from the LLM should already be a dict if as_json=True worked
            if isinstance(result, dict):
                logger.info(f"Successfully generated JSON query for: '{question}'")
                return result
            else:
                raise TypeError(f"LLM returned unexpected type: {type(result)}")

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            logger.error(
                f"Failed to generate or parse JSON query for '{question}': {e}"
            )
            return {"error": f"Failed to generate valid JSON: {e}"}


if __name__ == "__main__":
    # Example Usage and Testing
    import asyncio

    async def main():
        """
        Example usage and testing of the FunctionCallingHandler.

        Demonstrates various query generation scenarios including simple
        searches, aggregations, and complex filtering operations.
        """
        handler = FunctionCallingHandler()

        # Test 1: Simple Search
        question1 = "What are the calories for chicken breast?"
        print(f"\n1. Testing: '{question1}'")
        json_query1 = await handler.generate_json_query(question1)
        print(json.dumps(json_query1, indent=2))

        # Test 2: Aggregation
        question2 = "What is the average protein content for items with more than 50g of protein?"
        print(f"\n2. Testing: '{question2}'")
        json_query2 = await handler.generate_json_query(question2)
        print(json.dumps(json_query2, indent=2))

        # Test 3: Complex Search
        question3 = (
            "Find non-vegan items from the ontology that are either milk or cheese"
        )
        print(f"\n3. Testing: '{question3}'")
        json_query3 = await handler.generate_json_query(question3)
        print(json.dumps(json_query3, indent=2))

    # Note: This test requires the Arctic model to be loaded and available in Ollama.
    # If the model is not running, the tests will fail with an error message.
    asyncio.run(main())
