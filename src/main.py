import argparse
import csv
import sys
from itertools import product

import requests
import structlog
import yaml

from server_response import ServerResponse

LOGGER = structlog.get_logger()


def main(ollama_server, config_file, output_csv, skip_unloading):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if (
        not all(key in config for key in ["models", "prompts"])
        or len(config["models"]) < 1
        or len(config["prompts"]) < 1
    ):
        raise ValueError(
            "Config YAML file must contain at least one model and one prompt"
        )

    # Check Ollama server version
    try:
        response = requests.get(f"{ollama_server}/api/version")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        LOGGER.error("Error occurred while connecting to Ollama server", error=str(e))
        sys.exit(1)
    LOGGER.info(f"Ollama server version: {response.json()['version']}")

    # Check that all models exist on server
    try:
        response = requests.get(f"{ollama_server}/api/tags")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Could not load model list from Ollama server", error=str(e))
        sys.exit(1)
    json_response = response.json()
    local_models = [obj['name'] for obj in json_response["models"]]
    LOGGER.debug(f"Models available on Ollama server: {local_models}")
    for model in config["models"]:
        if model not in local_models:
            LOGGER.error(f"Did not find model {model} on Ollama server")
            sys.exit(1)
    LOGGER.info(f"All configured models available on Ollama server")

    # Unload all running models
    if not skip_unloading:
        try:
            response = requests.get(f"{ollama_server}/api/ps")
            response.raise_for_status()
            json_response = response.json()
            running_models = [obj['name'] for obj in json_response["models"]]
            if len(running_models) > 0:
                LOGGER.debug(f"Models running on Ollama server: {running_models}")
                for model in running_models:
                    payload = {
                        "model": model,
                        "keep_alive": 0,
                    }
                    response = requests.post(f"{ollama_server}/api/generate", json=payload)
                    response.raise_for_status()
            LOGGER.info("No models running on Ollama server")
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"Could not unload running models from Ollama server", error=str(e))
            sys.exit(1)

    result_cols = [
        "model",
        "prompt",
        "created_at",
        "total_duration",
        "load_duration",
        "eval_duration",
        "eval_rate",
    ]
    results = [result_cols]

    # Request prompts
    for model, prompt in product(config["models"], config["prompts"]):
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": 0,
            }
            response = requests.post(f"{ollama_server}/api/generate", json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            LOGGER.error(
                "Error occurred while sending prompt request to Ollama server",
                error=str(e),
            )
            sys.exit(1)

        try:
            json_response = response.json()
            response_obj = ServerResponse(json_response, prompt)
            LOGGER.debug(
                f"Received response from {response_obj.model} in {response_obj.total_duration}s ({response_obj.eval_rate} tokens/s)."
            )
        except (ValueError, KeyError) as e:
            LOGGER.error("Cannot parse response from Ollama server", error=str(e))
            sys.exit(1)

        result = [getattr(response_obj, key) for key in result_cols]
        results.append(result)

    # Generate output
    with open(output_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python script for evaluating the performance of Ollama servers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--ollama_server", required=True, help="Ollama server HTTP address"
    )
    parser.add_argument(
        "-c", "--config_file", default="benchmark.yml", help="Path to config YAML file"
    )
    parser.add_argument(
        "-o", "--output_csv", default="results.csv", help="Path to output CSV file"
    )
    parser.add_argument('--skip-unloading', action='store_true')
    args = parser.parse_args()

    main(args.ollama_server, args.config_file, args.output_csv,  args.skip_unloading)
