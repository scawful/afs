#!/usr/bin/env python3
"""
LMStudio API Client for Model Evaluation
"""

import requests
import json
import time
from typing import Dict, Optional

class LMStudioClient:
    """Client for querying models running in LMStudio."""

    MODELS = {
        "zelda-majora": "http://localhost:5000/chat",
        "zelda-din": "http://localhost:5001/chat",
        "zelda-farore": "http://localhost:5002/chat",
        "zelda-veran": "http://localhost:5003/chat",
        "zelda-hylia": "http://localhost:5004/chat",
        "zelda-scribe": "http://localhost:5005/chat",
        "scawful-echo": "http://localhost:5006/chat",
        "scawful-memory": "http://localhost:5007/chat",
        "scawful-muse": "http://localhost:5008/chat"
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def query(self, model_name: str, prompt: str) -> Optional[Dict]:
        """Query a model and return structured response."""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        endpoint = self.MODELS[model_name]

        try:
            response = requests.post(
                endpoint,
                json={"prompt": prompt},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": "timeout", "model": model_name}
        except requests.exceptions.ConnectionError:
            return {"error": "connection_failed", "model": model_name}
        except Exception as e:
            return {"error": str(e), "model": model_name}

    def batch_query(self, model_name: str, prompts: list) -> list:
        """Query multiple prompts with a single model."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Querying {model_name}...", end=" ", flush=True)
            start = time.time()
            result = self.query(model_name, prompt)
            elapsed = time.time() - start
            result["elapsed"] = elapsed
            results.append(result)
            print(f"({elapsed:.2f}s)")

        return results

    def health_check(self) -> Dict[str, bool]:
        """Check which models are available."""
        status = {}
        for model_name, endpoint in self.MODELS.items():
            try:
                response = requests.get(
                    endpoint.replace("/chat", "/health"),
                    timeout=2
                )
                status[model_name] = response.status_code == 200
            except:
                status[model_name] = False

        return status

if __name__ == "__main__":
    import sys

    client = LMStudioClient()

    print("LMStudio Model Health Check")
    print("=" * 40)

    status = client.health_check()
    for model_name, is_healthy in status.items():
        symbol = "✓" if is_healthy else "✗"
        print(f"{symbol} {model_name}")

    print("")
    if all(status.values()):
        print("All models are ready!")
    else:
        print("Some models are not available. Check LMStudio.")
