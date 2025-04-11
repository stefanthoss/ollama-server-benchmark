import argparse
import csv
import sys
from itertools import product

import structlog
import yaml

from ollama_server import OllamaServer

LOGGER = structlog.get_logger()


def main(
    server: OllamaServer,
    config_file: str,
    output_csv: str,
    benchmark_num: int,
    skip_unloading: bool,
):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if (
        not all(key in config for key in ["models", "prompts"])
        or not config["models"]
        or not config["prompts"]
    ):
        raise ValueError(
            "Config YAML file must contain at least one model and one prompt"
        )

    LOGGER.info(f"Ollama server version: {server.get_version()}")

    # Check that all models exist on server
    local_models = server.get_models()
    LOGGER.debug(f"Models available on Ollama server: {local_models}")
    for model in config["models"]:
        if model not in local_models:
            LOGGER.error(f"Did not find model {model} on Ollama server")
            sys.exit(1)
    LOGGER.info("All configured models available on Ollama server")

    if not skip_unloading:
        server.unload_models()

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
    LOGGER.info("Starting benchmark")
    for n, model, prompt in product(
        range(benchmark_num), config["models"], config["prompts"]
    ):
        response_obj = server.generate_response(model, prompt)
        LOGGER.debug(
            f"Run {(n + 1)}: Received response from {response_obj.model} in {response_obj.total_duration}s ({response_obj.eval_rate} tokens/s)"
        )

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
    parser.add_argument("--server", required=True, help="Ollama server HTTP address")
    parser.add_argument(
        "--config", default="benchmark.yml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--output", default="results.csv", help="Path to output CSV file"
    )
    parser.add_argument(
        "--num", default=1, type=int, help="Number of times to repeat the benchmark"
    )
    parser.add_argument(
        "--skip-unloading",
        action="store_true",
        help="Skip unloading all running models on the server",
    )
    args = parser.parse_args()

    server = OllamaServer(args.server, LOGGER)

    main(
        server,
        args.config,
        args.output,
        args.num,
        args.skip_unloading,
    )
