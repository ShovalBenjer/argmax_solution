#!/usr/bin/env python3
import os

"""
Redis Cache Manager for Diet Classification System

This module provides a centralized Redis caching solution optimized for storing
classification results and factual contexts. It implements a cache-first approach
with graceful fallback when Redis is unavailable.
"""

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

import json
import logging
import time
from typing import Any, Dict, Optional

from config import app_config

# Configure professional logging
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Thread-safe Redis cache manager with automatic fallback capabilities.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._redis_client = None
        self._is_connected = False
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not installed. Caching is disabled.")
            return

        try:
            self._redis_client = redis.StrictRedis(
                host=app_config.REDIS_HOST,
                port=app_config.REDIS_PORT,
                db=app_config.REDIS_DB,
                password=os.environ.get("REDIS_PASSWORD"),
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._redis_client.ping()
            self._is_connected = True
            logger.info(
                f"Redis cache manager connected to {app_config.REDIS_HOST}:{app_config.REDIS_PORT}"
            )
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching is disabled.")
            self._redis_client = None
            self._is_connected = False

    def is_available(self) -> bool:
        """Check if Redis is available and responsive."""
        return self._is_connected and self._redis_client is not None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Generic get method for retrieving and decoding JSON from cache."""
        if not self.is_available():
            return None
        try:
            cached_data = self._redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except (redis.exceptions.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve or decode cache key '{key}': {e}")
            return None

    def set(self, key: str, value: Dict[str, Any], ttl: int):
        """Generic set method for encoding and storing JSON in cache."""
        if not self.is_available():
            return
        try:
            serialized_data = json.dumps(value)
            self._redis_client.setex(key, ttl, serialized_data)
        except (redis.exceptions.RedisError, TypeError) as e:
            logger.error(f"Failed to set cache key '{key}': {e}")

    def get_ingredient_classification(
        self, ingredient_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached classification result for a single ingredient.
        """
        cache_key = f"ingredient_classification:{ingredient_key}"
        cached_wrapper = self.get(cache_key)
        if cached_wrapper:
            logger.debug(f"Cache hit for ingredient: {ingredient_key}")
            return cached_wrapper.get("classification_result")
        logger.debug(f"Cache miss for ingredient: {ingredient_key}")
        return None

    def set_ingredient_classification(
        self, ingredient_key: str, result: Dict[str, Any], ttl: int = 604800
    ):
        """
        Cache a classification result for a single ingredient. TTL defaults to 7 days.
        """
        cache_key = f"ingredient_classification:{ingredient_key}"
        # Wrap the result in a consistent structure with metadata
        cache_data = {
            "classification_result": result,
            "cached_at": time.time(),
            "cache_version": "1.0",
        }
        self.set(cache_key, cache_data, ttl)
        logger.debug(f"Cached classification for ingredient: {ingredient_key}")


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return CacheManager()
