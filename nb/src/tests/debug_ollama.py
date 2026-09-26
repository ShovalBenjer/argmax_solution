#!/usr/bin/env python3
"""
Debug script to check Ollama API responses and model availability.
"""

import json

import requests
from config import app_config


def debug_ollama():
    """Debug Ollama connection and model listing."""
    print("🔍 Debugging Ollama Connection...")
    print("=" * 50)

    # Check Ollama URL
    ollama_url = app_config.OLLAMA_URL
    print(f"Ollama URL: {ollama_url}")

    # Test basic connection
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            data = response.json()
            print(f"Raw Response: {json.dumps(data, indent=2)}")

            models = data.get("models", [])
            print(f"\nFound {len(models)} models:")
            for i, model in enumerate(models, 1):
                print(f"{i}. Model data: {model}")
                name = model.get("name", "No name")
                size = model.get("size", "No size")
                modified = model.get("modified_at", "No date")
                print(f"   Name: {name}")
                print(f"   Size: {size}")
                print(f"   Modified: {modified}")
                print()
        else:
            print(f"Error response: {response.text}")

    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # Test the list endpoint
    print("\n" + "=" * 50)
    print("Testing /api/list endpoint...")
    try:
        response = requests.get(f"{ollama_url}/api/list", timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"List Response: {json.dumps(data, indent=2)}")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    debug_ollama()
