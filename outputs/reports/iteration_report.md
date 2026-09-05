# Recursive AutoML Experiment Report

## Current champion

- Model: Extra Trees - Tuned
- Cross-validation RMSE: 0.4831
- Champion source: HyperparameterTuningAgent
- Protected test used: False

## Experiment history

| Iteration | Agent | Candidate | RMSE | Decision |
|---:|---|---|---:|---|
| 0 | ModelBenchmarkAgent | Random Forest | 0.5199 | Promoted |
| 1 | FeatureEngineeringAgent | Random Forest - Domain Features | 0.5195 | Rejected |
| 2 | HyperparameterTuningAgent | Random Forest - Tuned | 0.5199 | Rejected |
| 3 | AlternativeModelAgent | Extra Trees | 0.4881 | Promoted |
| 4 | HyperparameterTuningAgent | Extra Trees - Tuned | 0.4831 | Promoted |

## Next controlled experiment

- Experiment type: stop_and_review
- Model or models: Extra Trees - Tuned
- Reason: The approved model comparison and Extra Trees tuning experiments are complete. Human review is required before evaluating the final test set.
- Protected test access: False

## Governance rules

- The final test set remains untouched.
- Failed experiments remain in the history.
- A candidate must improve RMSE by at least 0.5% before promotion.
- Only parameters from the approved configuration may be tested.
