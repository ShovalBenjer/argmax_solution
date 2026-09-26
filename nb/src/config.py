"""
Centralized configuration management for all services.

This module provides a unified configuration system for the diet classification
pipeline, managing service URLs, API keys, database paths, and operational
parameters. It uses environment variables with sensible defaults and provides
validation capabilities.

The configuration supports both development and production environments
through environment variable overrides. All sensitive data (API keys) are
loaded from environment variables for security.

Example:
    >>> from config import app_config
    >>> print(app_config.OLLAMA_URL)
    'http://localhost:11434'
    >>> print(app_config.KETO_CARBS_THRESHOLD)
    10.0
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from decouple import config
except ImportError:
    # Fallback if decouple not available
    def config(key, default=None, cast=str):
        """Fallback configuration function when python-decouple is not available.

        Args:
            key: Environment variable name to retrieve
            default: Default value if environment variable is not set
            cast: Type conversion function (str, int, float, bool)

        Returns:
            The environment variable value converted to the specified type
        """
        value = os.environ.get(key)
        if value is None:
            return default
        return cast(value)


class Config:
    """
    Centralized configuration class for all services.

    This class manages all configuration parameters for the diet classification
    system, including service endpoints, API credentials, database paths,
    and operational thresholds. It implements a singleton-like pattern where
    all configuration is accessed through the global `app_config` instance.

    Attributes:
        OPENSEARCH_URL: URL for OpenSearch service endpoint
        OLLAMA_URL: URL for Ollama LLM service endpoint
        MLFLOW_TRACKING_URI: URI for MLflow experiment tracking
        HUGGING_FACE_HUB_TOKEN: API token for Hugging Face Hub access
        GOOGLE_API_KEY: API key for Google services
        PROJECT_ROOT: Root directory of the project
        DB_PATH: Path to the SQLite knowledge graph database
        RAW_DATA_DIR: Directory containing raw CSV data files
        EVAL_DATA_DIR = PROJECT_ROOT / "eval_data"
        KETO_CARBS_THRESHOLD: Maximum carbohydrate threshold for keto classification
        RPM_LIMIT: Rate limiting for requests per minute
        TPM_LIMIT: Token per minute limit for LLM requests
        BATCH_SIZE: Default batch size for data processing operations
        REDIS_HOST: Hostname for Redis service
        REDIS_PORT: Port for Redis service
        REDIS_DB: Database number for Redis
        QWEN_MAX_CONTEXT_TOKENS: Maximum tokens for Qwen context window
        ARCTIC_TIMEOUT: Timeout for Arctic operations
        QWEN_TIMEOUT: Timeout for Qwen operations
    """

    def __init__(self):
        logger.info("Initializing configuration...")

        # Service URLs - Support both OLLAMA_URL and OLLAMA_HOST for Docker compatibility
        self.OPENSEARCH_URL: str = config(
            "OPENSEARCH_URL", default="http://localhost:9200"
        )
        logger.debug(f"OPENSEARCH_URL: {self.OPENSEARCH_URL}")

        # Check for OLLAMA_HOST (used in Docker Compose), then OLLAMA_URL, then default
        _ollama_host = config("OLLAMA_HOST", default=None)
        _ollama_url = config("OLLAMA_URL", default=None)
        if _ollama_host and "://" not in _ollama_host:
            self.OLLAMA_URL: str = f"http://{_ollama_host}:11434"
            logger.debug(
                f"OLLAMA_URL set from OLLAMA_HOST (no schema): {self.OLLAMA_URL}"
            )
        elif _ollama_host:
            self.OLLAMA_URL: str = _ollama_host
            logger.debug(f"OLLAMA_URL set from OLLAMA_HOST: {self.OLLAMA_URL}")
        elif _ollama_url:
            self.OLLAMA_URL: str = _ollama_url
            logger.debug(f"OLLAMA_URL set from OLLAMA_URL: {self.OLLAMA_URL}")
        else:
            self.OLLAMA_URL: str = "http://localhost:11434"
            logger.debug(f"OLLAMA_URL defaulted to: {self.OLLAMA_URL}")

        self.MLFLOW_TRACKING_URI: str = config(
            "MLFLOW_TRACKING_URI", default="http://localhost:5000"
        )
        logger.debug(f"MLFLOW_TRACKING_URI: {self.MLFLOW_TRACKING_URI}")

        # Redis Configuration
        self.REDIS_HOST: str = config("REDIS_HOST", default="localhost")
        self.REDIS_PORT: int = config("REDIS_PORT", default=6379, cast=int)
        self.REDIS_DB: int = config("REDIS_DB", default=0, cast=int)
        logger.debug(
            (
                f"REDIS_HOST: {self.REDIS_HOST}, REDIS_PORT: {self.REDIS_PORT}, REDIS_DB: "
                f"{self.REDIS_DB}"
            )
        )

        # API Keys (sensitive, log only presence)
        self.HUGGING_FACE_HUB_TOKEN: Optional[str] = config(
            "HUGGING_FACE_HUB_TOKEN", default=None
        )
        if self.HUGGING_FACE_HUB_TOKEN:
            logger.debug("HUGGING_FACE_HUB_TOKEN is set.")
        self.GOOGLE_API_KEY: Optional[str] = config("GOOGLE_API_KEY", default=None)
        if self.GOOGLE_API_KEY:
            logger.debug("GOOGLE_API_KEY is set.")

        # Project paths
        self.PROJECT_ROOT = Path(__file__).resolve().parent
        self.DB_PATH = self.PROJECT_ROOT / "data" / "knowledge_graph.db"
        self.RAW_DATA_DIR = self.PROJECT_ROOT / "raw_data"
        self.EVAL_DATA_DIR = self.PROJECT_ROOT / "eval_data"
        logger.debug(f"PROJECT_ROOT: {self.PROJECT_ROOT}")
        logger.debug(f"DB_PATH: {self.DB_PATH}")
        logger.debug(f"RAW_DATA_DIR: {self.RAW_DATA_DIR}")
        logger.debug(f"EVAL_DATA_DIR: {self.EVAL_DATA_DIR}")

        # Classification thresholds
        self.KETO_CARBS_THRESHOLD: float = config(
            "KETO_CARBS_THRESHOLD", default=10.0, cast=float
        )
        logger.debug(f"KETO_CARBS_THRESHOLD: {self.KETO_CARBS_THRESHOLD}")

        # LLM Context Limits
        self.QWEN_MAX_CONTEXT_TOKENS: int = config(
            "QWEN_MAX_CONTEXT_TOKENS", default=1500, cast=int
        )
        logger.debug(f"QWEN_MAX_CONTEXT_TOKENS: {self.QWEN_MAX_CONTEXT_TOKENS}")

        # New timeout configurations
        self.ARCTIC_TIMEOUT: float = config("ARCTIC_TIMEOUT", default=30.0, cast=float)
        self.QWEN_TIMEOUT: float = config("QWEN_TIMEOUT", default=30.0, cast=float)
        logger.debug(
            f"ARCTIC_TIMEOUT: {self.ARCTIC_TIMEOUT}s, QWEN_TIMEOUT: {self.QWEN_TIMEOUT}s"
        )

        # Rate limiting
        self.RPM_LIMIT: int = config("RPM_LIMIT", default=10, cast=int)
        self.TPM_LIMIT: int = config("TPM_LIMIT", default=200000, cast=int)
        logger.debug(f"RPM_LIMIT: {self.RPM_LIMIT}, TPM_LIMIT: {self.TPM_LIMIT}")

        # Batch processing
        self.BATCH_SIZE: int = config("BATCH_SIZE", default=25, cast=int)
        logger.debug(f"BATCH_SIZE: {self.BATCH_SIZE}")

    @property
    def REDIS_URL(self) -> str:
        logger.debug(
            f"Constructing REDIS_URL: redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        )
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @classmethod
    def validate(cls) -> bool:
        logger.info("Validating required configuration paths...")
        required_paths = [cls.RAW_DATA_DIR]
        for path in required_paths:
            if not path.exists():
                logger.error(f"Required path does not exist: {path}")
                raise FileNotFoundError(f"Required path does not exist: {path}")
            logger.success(f"Required path exists: {path}")
        logger.success("Configuration validation successful.")
        return True


# Global config instance
app_config = Config()
