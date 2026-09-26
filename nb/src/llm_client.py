"""
Unified LLM Client for Ollama Integration

This module provides a consistent interface for interacting with Ollama-based
language models across the diet classification system. It supports both
synchronous and asynchronous operations with proper error handling and
JSON response parsing.

The LLMClient class abstracts the complexity of Ollama API interactions,
providing a simple interface for model queries, model discovery, and
connection management. It includes automatic fallback mechanisms when
Ollama is unavailable.

Key Features:
- Synchronous and asynchronous query support
- Robust JSON parsing and validation
- Model availability checking and discovery
- Connection health monitoring
- Graceful error handling and logging
"""

import asyncio
import json
import re
from typing import Any, Dict, Optional

try:
    import ollama
except ImportError:
    # logging.warning("Ollama not available, LLM functionality will be limited")
    ollama = None

from config import app_config
from loguru import logger  # Use loguru for enhanced logging

# Configure professional logging
# logger = logging.getLogger(__name__)


def _robust_json_parser(text: str) -> Dict[str, Any]:
    """
    Parses JSON from LLM output, handling common formatting issues.
    - Extracts JSON from markdown code blocks.
    - Removes trailing commas.
    """
    logger.debug(
        f"Attempting to robustly parse JSON from text (first 100 chars): {text[:100]}..."
    )
    # Find JSON within markdown code blocks (e.g., ```json ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if match:
        extracted_text = match.group(1)
        logger.debug("Extracted text from markdown code block.")
    else:
        extracted_text = text
        logger.debug("No markdown code block found, using raw text.")

    # Remove trailing commas before closing brackets/braces
    cleaned_text = re.sub(r",\s*([}\]])", r"\1", extracted_text)
    if cleaned_text != extracted_text:
        logger.debug("Removed trailing commas from JSON.")

    try:
        parsed_json = json.loads(cleaned_text)
        logger.debug("Successfully parsed JSON.")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to decode JSON response after cleaning: {cleaned_text[:500]}... Error: {e}"
        )
        raise ValueError(f"Invalid JSON response from LLM: {e}") from e


class LLMClient:
    """
    Unified client for Ollama LLM interactions.
    """

    def __init__(self, host: Optional[str] = None):
        """
        Initialize the LLM client with Ollama connection.
        """
        self.host = host or app_config.OLLAMA_URL
        self.client = None
        self.async_client = None
        logger.info(
            f"LLMClient initialized. Attempting to connect to Ollama at: {self.host}"
        )
        self._init_clients()

    def _init_clients(self):
        """
        Initialize Ollama client connections.
        """
        if not ollama:
            logger.warning(
                "Ollama library not installed, LLM features disabled. Please install 'ollama'."
            )
            return

        try:
            self.client = ollama.Client(host=self.host)
            self.async_client = ollama.AsyncClient(host=self.host)
            # Test connection
            self.client.list()
            logger.success(
                f"LLM client successfully connected to Ollama at {self.host}"
            )
        except Exception as e:
            logger.error(
                (
                    f"Failed to initialize or connect to Ollama at {self.host}. LLM features will "
                    f"be disabled. Error: "
                    f"{e}"
                )
            )
            self.client = None
            self.async_client = None

    def list_models(self) -> list:
        """
        List available models from Ollama service.
        """
        logger.info(f"Attempting to list models from Ollama at {self.host}")
        if not self.client:
            logger.warning("Ollama client not available, cannot list models.")
            return []
        try:
            models = self.client.list()
            # Handle different response formats
            if isinstance(models, dict):
                listed_models = models.get("models", [])
                logger.debug(
                    f"Listed models (dict format): {[m.get('name') for m in listed_models]}"
                )
                return listed_models
            elif isinstance(models, list):
                logger.debug(
                    f"Listed models (list format): {[m.get('name') for m in models]}"
                )
                return models
            elif hasattr(models, "models"):  # Handle ListResponse object
                listed_models = models.models
                logger.debug(
                    f"Listed models (ListResponse object): {[m.model for m in listed_models]}"
                )
                return listed_models
            else:
                logger.warning(
                    (
                        f"Unexpected response format from Ollama list(): {type(models)}. "
                        f"Returning empty "
                        f"list."
                    )
                )
                return []
        except Exception as e:
            logger.error(
                (
                    f"Failed to list models from Ollama at {self.host}: {e}. Attempting fallback "
                    f"via "
                    f"requests."
                )
            )
            # Try direct API call as fallback
            try:
                import requests

                response = requests.get(f"{self.host}/api/tags")
                response.raise_for_status()  # Raise an exception for HTTP errors
                data = response.json()
                fallback_models = data.get("models", [])
                logger.info(
                    (
                        f"Successfully listed models via fallback API call: "
                        f"{[m.get('name') for m in fallback_models]}"
                    )
                )
                return fallback_models
            except Exception as fallback_error:
                logger.error(
                    f"Fallback API call also failed to list models: {fallback_error}"
                )
            return []

    async def query_async(
        self, model: str, prompt: str, as_json: bool = True, timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Execute asynchronous query to Ollama model with robust JSON parsing.
        """
        logger.info(
            (
                f"Sending async query to model '{model}' (JSON expected: {as_json}) with timeout "
                f"{timeout}s. Prompt (first 100 chars): "
                f"{prompt[:100]}..."
            )
        )
        if not self.async_client:
            logger.error("Ollama async client not available. Cannot execute query.")
            return {"error": "Ollama async client not available"}

        try:
            response = await asyncio.wait_for(
                self.async_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    format="json" if as_json else "",
                ),
                timeout=timeout,
            )

            content = response["message"]["content"]

            logger.debug(
                f"Raw LLM response content (first 500 chars): {content[:500]}..."
            )  # Log first 500 chars for brevity

            if as_json:
                try:
                    parsed_content = _robust_json_parser(content)
                    logger.success("Successfully parsed JSON response.")
                    return parsed_content
                except ValueError as e:
                    logger.error(f"Failed to parse JSON from LLM response. Error: {e}")
                    return {"error": str(e), "raw_content": content}

            logger.success("Successfully received non-JSON LLM response.")
            return {"content": content}

        except asyncio.TimeoutError:
            logger.error(f"Async query timeout for model {model} after {timeout}s.")
            return {"error": f"Query timeout after {timeout}s"}
        except Exception as e:
            logger.error(f"Async query failed for model {model}. Error: {e}")
            return {"error": str(e)}

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available on the Ollama server.
        """
        logger.info(f"Checking availability of model: '{model_name}'")
        models = self.list_models()
        if not models:
            logger.warning(
                f"No models found on Ollama server when checking for '{model_name}'."
            )
            return False

        # Filter out None values from the list of model names
        # Handle both dict objects and Model objects
        available_model_names = []
        for m in models:
            if hasattr(m, "model"):  # Model object
                available_model_names.append(m.model)
            elif isinstance(m, dict) and m.get("name"):  # Dict object
                available_model_names.append(m.get("name"))

        # Debug logging
        logger.debug(f"Currently available models for check: {available_model_names}")

        # Check for exact match first
        if model_name in available_model_names:
            logger.success(
                f"Exact match found for model: '{model_name}'. Model is available."
            )
            return True

        # Check for partial match with more flexible logic
        model_base = model_name.split(":")[0]  # Remove tag if present
        model_base = model_base.split("/")[-1]  # Get last part after slash

        logger.debug(f"Searching for base model: '{model_base}'")

        for name in available_model_names:
            name_base = name.split(":")[0]  # Remove tag if present
            name_base = name_base.split("/")[-1]  # Get last part after slash

            # Check if the base names match
            if name_base.startswith(model_base) or model_base.startswith(name_base):
                logger.success(
                    f"Partial match found: '{name}' matches '{model_name}'. Model is available."
                )
                return True

        logger.warning(
            f"Model '{model_name}' (or its base) not found among available models."
        )
        return False


# Global LLM client instance
llm_client = LLMClient()
