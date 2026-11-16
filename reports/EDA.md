# Exploratory data analysis

Original rows: **151,445**; cleaned rows: **71,890**.
Target median: **$138,791**; 10–90 percentile: **$67,500–$250,000**.

## Missing values (before cleaning)

- `work_year`: 0
- `experience_level`: 0
- `employment_type`: 0
- `job_title`: 0
- `salary`: 0
- `salary_currency`: 0
- `salary_in_usd`: 0
- `employee_residence`: 0
- `remote_ratio`: 0
- `company_location`: 0
- `company_size`: 0

## Median salary by experience level

- `EN`: $85,500
- `MI`: $122,000
- `SE`: $156,160
- `EX`: $190,000

## Feature importance

Permutation importance uses Ridge(alpha=1), the `temporal: <=2023 / 2024 / >=2025` split, and 2,000 held-out rows with two shuffles per feature. Values show the increase in MAE when each feature is shuffled; larger positive values indicate greater predictive contribution for this fitted model.

- `job_title`: MAE increase $7,463 ± $254
- `experience_level`: MAE increase $4,733 ± $438
- `employee_residence`: MAE increase $3,400 ± $121
- `company_location`: MAE increase $3,072 ± $177
- `employment_type`: MAE increase $307 ± $47
- `remote_ratio`: MAE increase $67 ± $9
- `company_size`: MAE increase $-6 ± $3

These values describe predictive association, not causation. Correlated features can share or mask importance.

## Limitations

- Survey/sample selection bias; cross-country salary levels may not be comparable.
- No individual identifier; repeated profiles may exist. Exact duplicate records are removed.
- `salary`, `salary_currency`, and `salary_in_usd` are never features.
- These associations are descriptive, not causal.
