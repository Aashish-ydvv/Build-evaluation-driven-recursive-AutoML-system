# Evaluation-Driven Recursive AutoML

A controlled multi-agent machine-learning system that proposes, evaluates, accepts or rejects experiments while protecting an untouched final test set.

## Problem statement

The project predicts building heating load from eight architectural design variables.

The objective is not only to train one model. The system maintains a champion model, tests controlled improvements and promotes a new candidate only when it passes a predefined cross-validation threshold.

## Dataset

- Dataset: UCI Energy Efficiency dataset
- Total rows: 768
- Input features: 8
- Target: heating_load
- Task type: regression
- Missing values: 0
- Duplicate rows: 0

## System components

1. **Model Benchmark Agent:** compares approved baseline model families.
2. **Feature Engineering Agent:** evaluates domain-informed transformations.
3. **Results Aggregation Agent:** validates and combines experiment results.
4. **Reporting Agent:** records the history and proposes the next controlled experiment.

Specialized execution components perform approved hyperparameter and alternative-model experiments.

## Evaluation methodology

- Development data: 614 rows
- Protected test data: 154 rows
- Development/test split: 80/20
- Cross-validation: shuffled five-fold CV
- Primary selection metric: mean RMSE
- Promotion threshold: minimum 0.5% improvement
- Final test evaluations: 1

The test set was not supplied to the experiment agents. It was evaluated once after model selection stopped.

## Experiment history

| Iteration | Agent | Candidate | RMSE | Decision |
|---:|---|---|---:|---|
| 0 | ModelBenchmarkAgent | Random Forest | 0.5199 | Promoted |
| 1 | FeatureEngineeringAgent | Random Forest - Domain Features | 0.5195 | Rejected |
| 2 | HyperparameterTuningAgent | Random Forest - Tuned | 0.5199 | Rejected |
| 3 | AlternativeModelAgent | Extra Trees | 0.4881 | Promoted |
| 4 | HyperparameterTuningAgent | Extra Trees - Tuned | 0.4831 | Promoted |

## Final champion

- Model: Extra Trees - Tuned
- Cross-validation RMSE: 0.4831
- Parameters:
  - Number of trees: 400
  - Maximum depth: 12
  - Minimum samples per leaf: 2
  - Maximum features: 1.0

## Final protected-test results

| Metric | Value |
|---|---:|
| MAE | 0.3291 |
| RMSE | 0.4832 |
| R2 | 0.9978 |

The cross-validation RMSE and test RMSE were nearly identical, indicating consistent performance on the protected holdout.

## Feature importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | overall_height | 0.5025 |
| 2 | roof_area | 0.2471 |
| 3 | glazing_area | 0.0847 |
| 4 | relative_compactness | 0.0559 |
| 5 | surface_area | 0.0544 |
| 6 | wall_area | 0.0493 |
| 7 | glazing_distribution | 0.0059 |
| 8 | orientation | 0.0003 |

Tree-based feature importance describes how this model used the variables. It does not establish that a feature causes heating-load changes.

## Error analysis

- Mean residual: -0.0549
- Median absolute error: 0.1619
- Maximum absolute error: 1.4403

A small negative mean residual indicates slight average overprediction, but the value is close to zero.

## Diagnostic plots

![Final model diagnostics](../figures/final_model_diagnostics.png)

## Governance safeguards

- The final test set was isolated before experiments.
- Experiment agents received only development data.
- Failed experiments were preserved.
- Only approved model families were executed.
- Promotion required a predefined improvement.
- Duplicate history updates were prevented.
- Test evaluation required explicit human approval.
- The test set was evaluated only once.

## Limitations

- The dataset contains simulated building configurations rather than measurements from operating buildings.
- The dataset contains only 768 rows and a limited collection of design variables.
- Climate, occupancy, construction materials and equipment schedules are not represented.
- Feature importance may be affected by strongly correlated input variables.
- The result should not replace professional building-energy simulation.

## Saved artifacts

- Final model: `outputs/models/extra_trees_final.joblib`
- Final metrics: `outputs/final_results/final_test_metrics.json`
- Test predictions: `outputs/final_results/final_test_predictions.csv`
- Feature importance: `outputs/final_results/feature_importance.csv`
- Diagnostic figure: `outputs/figures/final_model_diagnostics.png`

## Reproducibility statement

All dataset splitting, model training and cross-validation operations use fixed random states. The experiment history records both accepted and rejected candidates.
