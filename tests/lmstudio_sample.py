import requests
import random
import time

LMSTUDIO_URL = "http://127.0.0.1:1234"

def list_models():
    resp = requests.get(f"{LMSTUDIO_URL}/v1/models")
    resp.raise_for_status()
    data = resp.json()
    models = [m["id"] for m in data.get("data", [])]
    print("Available models:", models)
    return models

def load_model(model_name):
    print(f"Simulating load for model: {model_name}")
    # LM Studio loads models via UI or API call; here we just check for availability
    for _ in range(10):
        time.sleep(2)
        models = list_models()
        if model_name in models:
            print(f"Model '{model_name}' is available in LM Studio.")
            return True
    print(f"Model '{model_name}' did not appear in LM Studio in time.")
    return False

def main():
    models = list_models()
    if not models:
        print("No models found in LM Studio.")
        return
    model_name = random.choice(models)
    print(f"Randomly selected model: {model_name}")
    load_model(model_name)

if __name__ == "__main__":
    main()
