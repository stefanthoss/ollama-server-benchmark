import sys
from typing import List

import requests


class OllamaServer:
    def __init__(self, address, logger):
        self.address = address
        self.logger = logger

    def get_version(self) -> str:
        try:
            response = requests.get(f"{self.address}/api/version")
            response.raise_for_status()
            return response.json()["version"]
        except requests.exceptions.RequestException as e:
            self.logger.error("Error occurred while connecting to Ollama server", error=str(e))
            sys.exit(1)

    def get_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.address}/api/tags")
            response.raise_for_status()
            return [obj['name'] for obj in response.json()["models"]]
        except requests.exceptions.RequestException as e:
            self.logger.error("Could not load model list from Ollama server", error=str(e))
            sys.exit(1)

    def unload_models(self):
        try:
            response = requests.get(f"{self.address}/api/ps")
            response.raise_for_status()
            json_response = response.json()
            running_models = [obj["name"] for obj in json_response["models"]]
            if len(running_models) > 0:
                self.logger.debug(f"Stopping these models on Ollama server: {running_models}")
                for model in running_models:
                    payload = {
                        "model": model,
                        "keep_alive": 0,
                    }
                    response = requests.post(
                        f"{self.address}/api/generate", json=payload
                    )
                    response.raise_for_status()
            self.logger.info("No models running on Ollama server")
        except requests.exceptions.RequestException as e:
            self.logger.error(
                "Could not unload running models from Ollama server", error=str(e)
            )
            sys.exit(1)
