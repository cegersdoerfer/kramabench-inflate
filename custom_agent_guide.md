# Implementing a Custom Agent for KramaBench

## Interface

Your agent must inherit from `benchmark.benchmark_api.System` (defined in `benchmark/benchmark_api.py`) and implement two methods:

### `process_dataset(dataset_directory)`

Called once at startup with the path to the input data directory (e.g. `data/astronomy/input/`). Use this to load, parse, or index the dataset files your agent will need. You **must** set `self.dataset_directory = dataset_directory`.

### `serve_query(query, query_id, subset_files) -> Dict`

Called once per benchmark task. Arguments:

| Argument | Type | Description |
|---|---|---|
| `query` | `str` | Natural language question |
| `query_id` | `str` | Task identifier (e.g. `"astronomy-easy-1"`) |
| `subset_files` | `List[str]` | Relevant filenames, or `[]` for "use all files" |

Must return a dict with this structure:

```python
{
    "explanation": {"answer": <value>},  # required — evaluated against ground truth
    "pipeline_code": "<python code>",    # required — the code that produces the answer
    "token_usage": 0,                    # optional
    "token_usage_input": 0,              # optional
    "token_usage_output": 0,             # optional
}
```

The `"answer"` value can be a string, number, list, or dict depending on the task.

## Minimal Example

```python
# systems/my_agent.py
from benchmark.benchmark_api import System
from typing import List, Dict, Any
import os
import pandas as pd

class MyAgent(System):
    def __init__(self, name="MyAgent", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.output_dir = kwargs.get("output_dir", ".")
        self.dataset = {}

    def process_dataset(self, dataset_directory):
        self.dataset_directory = dataset_directory
        for root, _, files in os.walk(dataset_directory):
            for fname in files:
                path = os.path.join(root, fname)
                rel = os.path.relpath(path, dataset_directory)
                if fname.endswith(".csv"):
                    try:
                        self.dataset[rel] = pd.read_csv(path)
                    except Exception:
                        pass

    def serve_query(self, query, query_id="", subset_files=None):
        # Your logic here — call an LLM, run generated code, look up data, etc.
        answer = "placeholder"
        code = "# generated code"

        return {
            "explanation": {"answer": answer},
            "pipeline_code": code,
            "token_usage": 0,
        }
```

## Registration

Add your class to `systems/__init__.py`:

```python
from .my_agent import MyAgent
```

## Running

```bash
python evaluate.py --sut MyAgent --workload astronomy --verbose
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--sut` | required | Your system class name |
| `--workload` | `legal` | Domain to benchmark (`astronomy`, `legal`, `environment`, etc.) |
| `--num_workers` | `8` | Parallel task workers |
| `--run_subtasks` | `false` | Also evaluate subtasks |
| `--use_system_cache` | `false` | Skip `serve_query` if a cached response exists |
| `--no_pipeline_eval` | `false` | Skip LLM-based code evaluation (saves API calls) |
| `--verbose` | `false` | Enable verbose logging |

## Parallelization

The benchmark runs tasks in parallel (`--num_workers`). If your agent holds unpicklable objects (LLM clients, DB connections, etc.), implement `__getstate__` and `__setstate__`:

```python
def __getstate__(self):
    state = self.__dict__.copy()
    state.pop("llm_client", None)  # remove unpicklable object
    return state

def __setstate__(self, state):
    self.__dict__.update(state)
    self.llm_client = create_client()  # reinitialize after unpickling
```

## Common Patterns

**Code-generation agent** (like the built-in DS-GURU baseline): send the query and data context to an LLM, ask it to generate Python code, execute the code, extract the answer from stdout, and retry on errors.

**RAG agent**: index dataset files in `process_dataset`, retrieve relevant chunks in `serve_query`, pass them to an LLM alongside the query.

**Direct analysis**: load DataFrames in `process_dataset`, write domain-specific logic in `serve_query` to compute answers programmatically.

## Output

Results are written to:

```
results/<YourClassName>/<domain>_measures_<timestamp>.csv   # per-task metrics
results/aggregated_results.csv                              # summary
```

Scratch files can be written to the `output_dir` passed via `kwargs` in `__init__`.
