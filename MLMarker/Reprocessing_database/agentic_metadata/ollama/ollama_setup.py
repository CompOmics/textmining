"""Code to setup an Ollama server with a specified model and settings"""

import subprocess
import requests
import time
import os

OLLAMA_EXE = os.path.join(os.environ["LOCALAPPDATA"], "Programs", "Ollama", "ollama.exe")

def is_ollama_running(base_url: str) -> bool:
    """Check if Ollama server is already running by sending a request to the base URL."""
    
    try:
        response = requests.get(base_url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    
def setup_ollama_server(base_url: str, model: str, env_vars: dict, ):    
    # Check if Ollama is already running and stop
    is_running = is_ollama_running("http://localhost:11434")
    if is_running:
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    # Start new Ollama server with config params
    env = os.environ.copy()
    env["OLLAMA_HOST"] = base_url.replace("http://", "")
    env["OLLAMA_FLASH_ATTENTION"] = env_vars.get("OLLAMA_FLASH_ATTENTION", "0")  
    env["OLLAMA_KV_CACHE_TYPE"] = env_vars.get("OLLAMA_KV_CACHE_TYPE", "f16")
    
    process = subprocess.Popen(
                                [OLLAMA_EXE, "serve"],
                                env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                )
    
    for attempt in range(15):
        time.sleep(1)
        if is_ollama_running("http://localhost:11434"):
            break
    else:
        raise RuntimeError("Failed to start Ollama server after 15 seconds.")

    # Pull model if not already downloaded
    response = requests.get(f"{base_url}/api/tags")
    installed_models = [m["name"] for m in response.json().get("models", [])]

    if model not in installed_models:
        print(f"Pulling {model}...")
        requests.post(f"{base_url}/api/pull", json={"name": model}, timeout=600)
    else:
        print(f"ollama with {model} is ready.")

if __name__ == "__main__":
    setup_ollama_server(
        base_url="http://localhost:11434",
        model="gurubot/Qwen3.5-35B-A3B-GGUF-unsloth-nothink:UD-Q4_K_XL",
        env_vars={
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q4_0",
        },
    )