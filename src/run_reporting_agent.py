import json
from pathlib import Path

from agents.reporting_agent import ReportingAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
)

REPORTS_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

AGGREGATED_RESULTS_PATH = (
    EXPERIMENTS_FOLDER
    / "aggregated_results.json"
)


def load_aggregated_results():
    if not AGGREGATED_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Aggregated results were not found. "
            "Run run_results_aggregation_agent.py first."
        )

    with AGGREGATED_RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():
    aggregated_results = (
        load_aggregated_results()
    )

    agent = ReportingAgent(
        reports_folder=REPORTS_FOLDER,
        experiments_folder=EXPERIMENTS_FOLDER,
    )

    agent.run(
        aggregated_results=aggregated_results,
    )


if __name__ == "__main__":
    main()