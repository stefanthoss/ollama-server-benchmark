# Ollama Server Benchmark
Python script for evaluating the performance of Ollama servers using the Ollama API

## Usage

```shell
uv run src/main.py -h
uv run src/main.py -c benchmark.yml -o results.csv -s http://ollama.example.com:11434
```

## Development

```shell
uv tool run ruff check
uv tool run ruff format
```
