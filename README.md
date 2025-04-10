# Ollama Server Benchmark
Python script for evaluating the performance of Ollama servers using the Ollama API

## Usage

```shell
uv run src/main.py --help
uv run src/main.py --config benchmark.yml --output results.csv --server http://ollama.example.com:11434 --num 1
```

## Development

```shell
uv tool run ruff check
uv tool run ruff format
```
