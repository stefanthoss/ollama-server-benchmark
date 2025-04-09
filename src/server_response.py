MAX_STR_PRINT_LENGTH = 30


class ServerResponse:
    def __init__(self, response_data, prompt):
        self.prompt = prompt
        for key, value in response_data.items():
            if isinstance(value, str):
                setattr(self, key, value)
            elif isinstance(value, int):
                # All duration fields are using nanoseconds
                if key.endswith("_duration"):
                    value = value / 1e9
                setattr(self, key, value)

        # Calculated fields, see https://github.com/ollama/ollama/blob/main/benchmark/server_benchmark_test.go
        self.eval_rate = self.eval_count / self.eval_duration
        self.prompt_eval_rate = self.prompt_eval_count / self.prompt_eval_duration

    def __str__(self):
        attrs = []
        for key, value in vars(self).items():
            if isinstance(value, str) and len(value) > MAX_STR_PRINT_LENGTH:
                value = value[:MAX_STR_PRINT_LENGTH] + "..."
            attrs.append(f"{key}={value}")
        return f"ServerResponse({', '.join(attrs)})"
