# AI / Data Job Salary Predictor — ML Zoomcamp 2026 Midterm

**Problem.** Predict indicative gross annual compensation in USD from role, seniority, employment type, employee/company geography, remote ratio, and company size. Use as a statistical benchmark; **not** a guaranteed salary offer or a claim about Georgia B2B market rates. This is an independent educational project.

**Source.** aijobs.net Global AI/ML/Data Science Salary Index: <https://github.com/foorilla/ai-jobs-net-salaries>, CC0. Source snapshot downloaded by `scripts/download.py`. `data/source.json` records URL, byte count and SHA256.

## Run locally (Python 3.12 recommended)

This repo uses `uv` and a local `.venv`. After installing `uv`, run:

```bash
make setup
make data
make eda
make train
make test
make serve
```

In another terminal:

```bash
curl -fsS http://localhost:8080/health
curl -fsS -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d @sample-request.json
```

The trained artifact `models/model.joblib` is committed with the project so a
fresh clone contains everything needed to build and run the Docker image. To
retrain it from the source data, run `make data` followed by `make train`.
Then build and start the container:

```bash
docker build -t mlzc-salary .
docker run --rm -p 8080:8080 mlzc-salary
```

**Docker verification:** with the container running, check both endpoints:

```bash
curl -fsS http://localhost:8080/health
curl -fsS -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d @sample-request.json
```

`models/model.joblib` is tracked in Git and included in the image. The original
CSV need not be committed; the download script fetches it. Training metrics and
EDA outputs are generated locally and are not required to serve predictions.

## Data and leakage controls

Input fields: `experience_level`, `employment_type`, `job_title`, `employee_residence`, `remote_ratio`, `company_location`, `company_size`. Never use `salary`, `salary_currency`, or `salary_in_usd` to predict `salary_in_usd` (direct/derived target leakage). `work_year` is used only to split the observations: <=2023 for training, 2024 validation, >=2025 test if these years have enough records; deterministic random fallback documented in `models/metrics.json`. Model fitting may use up to 50k reproducibly sampled rows to keep runtime short.

## Model selection

- DummyRegressor(median) is the baseline.
- Ridge (two regularization settings) and RandomForestRegressor (two max-depth settings) are compared on validation MAE/RMSE.
- The winner is refitted on train+validation and **only then** evaluated on held-out test data.
- `models/metrics.json` contains the actual metrics and model choice; do not invent them.
- `reports/EDA.md` and `reports/salary_histogram.png` contain EDA artifacts.

This is a survey-based historical dataset, likely with strong geographic/sample-selection bias. Do not make a job offer or individual compensation decision from the model.

## Deployment: Google Cloud Run (optional, requires billing)

```bash
# Must have local model.joblib and configured gcloud CLI / active project
PROJECT_ID="YOUR_GCP_PROJECT"
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy mlzc-salary --source . --region europe-west1 \
  --allow-unauthenticated --memory 1Gi --max-instances 1 --min-instances 0
# Then curl the issued service URL with sample-request.json
```

Set a billing budget and verify permissions before cloud deployment. `--allow-unauthenticated` makes the example prediction service publicly accessible: deploy no private data and use rate/spending limits. If deployment is blocked, document why and keep verified local Docker evidence.

## Submission checklist

- [ ] Screenshots and EDA narrative generated from actual dataset.
- [ ] `models/metrics.json` contains real (not synthetic) model metrics.
- [ ] `pytest -q` passes and public Docker build works from clean Git clone.
- [ ] `/predict` tested with `sample-request.json`.
- [ ] Deploy URL / video / screenshot and instructions added if applicable.
- [ ] Repo public; exact commit SHA recorded.
- [ ] Project link and commit SHA submitted at https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm before 2026-11-16 23:00 UTC.
- [ ] Complete three assigned peer reviews by 2026-11-23 23:00 UTC.

## References

- Course requirements: https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/projects
- Course project page: https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm
