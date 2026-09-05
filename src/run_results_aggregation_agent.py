import json
from pathlib import Path

from agents.results_aggregation_agent import (
    ResultsAggregationAgent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
)

BENCHMARK_RESULTS_PATH = (
    EXPERIMENT_FOLDER
    / "benchmark_results.json"
)

FEATURE_RESULTS_PATH = (
    EXPERIMENT_FOLDER
    / "feature_engineering_results.json"
)


def load_json_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required result file was not found:\n"
            f"{file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():
    benchmark_results = load_json_file(
        BENCHMARK_RESULTS_PATH
    )

    feature_results = load_json_file(
        FEATURE_RESULTS_PATH
    )

    agent = ResultsAggregationAgent(
        output_folder=EXPERIMENT_FOLDER,
    )

    agent.run(
        benchmark_results=benchmark_results,
        feature_results=feature_results,
    )


if __name__ == "__main__":
    main()