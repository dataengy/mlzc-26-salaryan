# Problem definition and scope

## Prediction task

Estimate indicative gross annual compensation in USD for a role described by job title, seniority, employment type, employee residence, remote ratio, company location, and company size. This is a supervised regression task: `salary_in_usd` is the target.

## Intended use

The estimate is an educational, dataset-level benchmark for exploring salary patterns. It is not a salary offer, a negotiation recommendation, an estimate of an individual's value, or a statement about Georgia's B2B market. The dataset is a historical global survey/index and may not represent a particular labor market or current compensation.

## Inputs and leakage controls

The seven inference features are listed in `model_utils.py` and validated by the API request schema. `salary`, `salary_currency`, and `salary_in_usd` must never be model inputs. `work_year` is used only to create temporal evaluation partitions; it is not included in the prediction request or feature matrix.

## Success measures

The project compares candidates using validation MAE and RMSE, chooses the lowest validation MAE, then evaluates that fixed choice once on a held-out test set. MAE is expressed in USD and is easier to interpret as a typical absolute error; RMSE emphasizes larger errors. See `models/metrics.json` for the generated run values.

## Risks and limitations

- Selection and reporting bias in the source data.
- Geographic pay levels and currency conversions may not be comparable across countries or years.
- Job title and location categories can have sparse or shifting coverage.
- Similar records may represent repeated respondents; exact deduplication cannot identify them.
- The model omits important determinants such as company, industry, actual responsibilities, and negotiation context.

The service returns an estimate with a disclaimer and must not be used for hiring, compensation, or other individual decisions.
