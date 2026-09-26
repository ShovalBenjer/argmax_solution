import json
import os
import sys
from pathlib import Path

import requests

# Add the project root to the Python path to allow importing config
project_root = Path(__file__).resolve().parent.parent  # Go up one level to /src
sys.path.append(str(project_root))

try:
    from config import app_config
except ImportError:
    print("Warning: Could not import app_config. Using fallback Ollama URL.")

    class FallbackConfig:
        OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")

    app_config = FallbackConfig()


def get_existing_models() -> list:
    """Gets a list of names of existing models in Ollama."""
    try:
        response = requests.get(f"{app_config.OLLAMA_URL}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m.get("name") for m in models if m.get("name")]
    except requests.exceptions.RequestException:
        return []  # Return empty list if Ollama is not reachable


def create_model_from_file(model_name: str, modelfile_content: str):
    """
    Generic function to create a model in Ollama and stream the status.
    """
    api_url = f"{app_config.OLLAMA_URL}/api/create"
    payload = {"name": model_name, "modelfile": modelfile_content}

    print(f"Creating model '{model_name}'...")
    print("-" * 40)

    try:
        response = requests.post(api_url, json=payload, stream=True, timeout=600)
        response.raise_for_status()

        success = True
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    # Handle both string and integer status values
                    if isinstance(status, str):
                        status_display = status.strip()
                    else:
                        status_display = str(status)
                    print(f"\r> Status: {status_display}", end="", flush=True)
                    if "error" in data:
                        print(f"\nERROR: {data['error']}")
                        print(
                            (
                                "Hint: This often means the Ollama server cannot access the file "
                                "path in the 'FROM' "
                                "instruction."
                            )
                        )
                        success = False
                except json.JSONDecodeError:
                    print(f"\n[RAW]: {line.decode(errors='ignore')}")

        print("\n" + "-" * 40)
        if success and "error" not in data:
            print(f"Successfully processed model '{model_name}'.")
            return True
        else:
            print(f"Failed to create model '{model_name}'.")
            return False

    except requests.exceptions.RequestException as e:
        print(
            (
                f"\nAPI Error: Failed to connect to Ollama at {api_url}. Is Ollama running and "
                f"accessible?"
            )
        )
        print(f"Details: {e}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return False


def load_qwen_model(base_path: Path, existing_models: list):
    """Load Qwen model into Ollama."""
    model_name = "qwen/qwen3-0.6b-gguf:q8_0"
    if model_name in existing_models:
        print(f"Model '{model_name}' already exists. Skipping.")
        return True

    model_file = base_path / "qwen3-0.6b-gguf" / "qwen3-0.6b-base-q8_0.gguf"

    print(f"Attempting to load Qwen model from: {model_file.resolve()}")
    if not model_file.exists():
        print("ERROR: Qwen model file not found at the specified path!")
        return False

    # Use relative path for Docker container
    modelfile_content = """FROM ./qwen3-0.6b-gguf/qwen3-0.6b-base-q8_0.gguf

TEMPLATE "<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 4096
"""
    return create_model_from_file(model_name, modelfile_content)


def load_arctic_model(base_path: Path, existing_models: list):
    """Load Arctic Text2SQL model into Ollama."""
    model_name = "arctic-text2sql:latest"  # Use a standard tag
    if model_name in existing_models:
        print(f"Model '{model_name}' already exists. Skipping.")
        return True

    # Let's try to find the file with a more flexible approach
    arctic_dir = base_path / "Arctic-Text2SQL-R1-7B-GGUF"
    if not arctic_dir.exists():
        print(f"ERROR: Directory for Arctic model not found: {arctic_dir.resolve()}")
        return False

    found_files = list(arctic_dir.glob("*.gguf"))
    if not found_files:
        print(f"ERROR: No .gguf files found in {arctic_dir.resolve()}")
        return False

    model_file = found_files[0]  # Use the first GGUF file found in the directory

    print(f"Found and attempting to load Arctic model from: {model_file.resolve()}")

    # Use relative path for Docker container
    modelfile_content = """FROM ./Arctic-Text2SQL-R1-7B-GGUF/Arctic-Text2SQL-R1-7B.Q2_K.gguf

TEMPLATE "{{ .Prompt }}"

PARAMETER stop ";"
PARAMETER num_ctx 8192
"""
    return create_model_from_file(model_name, modelfile_content)


def list_models_summary():
    """List available models in Ollama."""
    print("\n--- Verifying Available Models in Ollama ---")
    existing_models = get_existing_models()
    if not existing_models:
        print("  Could not connect to Ollama or no models are installed.")
    else:
        for model in existing_models:
            print(f"  - Found: {model}")
    print("-" * 42)


if __name__ == "__main__":
    print("--- Starting Model Loader for Ollama ---")

    # The base path where the model directories (qwen3-0.6b-gguf, etc.) are located.
    # This should be the directory where the script is located (src/models)
    models_base_path = Path(__file__).parent

    # Get a list of models that are already loaded to avoid re-creating them.
    print("Checking for existing models...")
    existing_models = get_existing_models()
    if not existing_models:
        print(
            "Could not reach Ollama to check for existing models. Will attempt to create them."
        )

    qwen_success = load_qwen_model(models_base_path, existing_models)
    arctic_success = load_arctic_model(models_base_path, existing_models)

    # Final status check
    list_models_summary()

    if qwen_success and arctic_success:
        print("\nSUCCESS: All required models are available in Ollama!")
    else:
        print(
            "\nWARNING: One or more models could not be loaded. Please check the errors above."
        )
