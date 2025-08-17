import requests
import random
import time

OLLAMA_URL = "http://127.0.0.1:11434"

def list_models():
    resp = requests.get(f"{OLLAMA_URL}/api/tags")
    resp.raise_for_status()
    data = resp.json()
    models = [m["name"] for m in data.get("models", [])]
    print("Available models:", models)
    return models

def load_model(model_name):
    print(f"Requesting to load model: {model_name}")
    resp = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model_name})
    resp.raise_for_status()
    print("Pull request sent, waiting for model to load...")

    # Poll for model availability
    for _ in range(20):
        time.sleep(2)
        models = list_models()
        if model_name in models:
            print(f"Model '{model_name}' loaded successfully.")
            return True
    print(f"Model '{model_name}' did not load in time.")
    return False

def main():
    models = list_models()
    if not models:
        print("No models found on Ollama.")
        return
    model_name = random.choice(models)
    print(f"Randomly selected model: {model_name}")
    load_model(model_name)

if __name__ == "__main__":
    main()
