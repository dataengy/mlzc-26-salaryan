.PHONY: setup kernel run-jupyter data eda train smoke unit integration test e2e serve docker-build docker-run docker-check

setup:
	uv venv --python 3.12 --allow-existing .venv
	uv pip install --python .venv/bin/python -r settings/requirements.txt

kernel: setup
	uv pip install --python .venv/bin/python ipykernel notebook
	.venv/bin/python -m ipykernel install --user --name mlzoomcamp-salary --display-name "Python (ML Zoomcamp Salary)"

run-jupyter: kernel
	MPLCONFIGDIR=.run/matplotlib .venv/bin/python -m jupyter notebook notebooks/salary-prediction.ipynb

data:
	PYTHONPYCACHEPREFIX=.run/python-cache PYTHONPATH=src .venv/bin/python scripts/download.py

eda:
	PYTHONPYCACHEPREFIX=.run/python-cache PYTHONPATH=src .venv/bin/python -m mlzc_26_salaryan.eda

train:
	PYTHONPYCACHEPREFIX=.run/python-cache PYTHONPATH=src .venv/bin/python -m mlzc_26_salaryan.train

smoke:
	PYTHONPYCACHEPREFIX=.run/python-cache PYTHONPATH=src .venv/bin/python -m pytest -c settings/pytest.ini --rootdir=. -q tests/smoke

unit:
	PYTHONPYCACHEPREFIX=.run/python-cache PYTHONPATH=src .venv/bin/python -m pytest -c settings/pytest.ini --rootdir=. -q tests/unit

integration:
	PYTHONPYCACHEPREFIX=.run/python-cache PYTHONPATH=src .venv/bin/python -m pytest -c settings/pytest.ini --rootdir=. -q tests/integration

e2e:
	PYTHONPYCACHEPREFIX=.run/python-cache PYTHONPATH=src .venv/bin/python -m pytest -c settings/pytest.ini --rootdir=. -q -m e2e tests/e2e

test: smoke unit integration

serve:
	PYTHONPYCACHEPREFIX=.run/python-cache PYTHONPATH=src .venv/bin/python -m uvicorn mlzc_26_salaryan.app:app --reload --port 8080

docker-build:
	docker build -f deploy/Dockerfile -t mlzc-salary .

docker-run:
	docker run --rm -p 8080:8080 mlzc-salary

docker-check: docker-build
	@set -eu; \
	name=mlzc-salary-check-$$$$; \
	trap 'docker rm -f "$$name" >/dev/null 2>&1 || true' EXIT INT TERM; \
	docker run -d --name "$$name" -p 18080:8080 mlzc-salary >/dev/null; \
	ready=; \
	for attempt in $$(seq 1 30); do \
		if curl -fsS http://localhost:18080/health; then ready=1; break; fi; \
		sleep 1; \
	done; \
	if [ -z "$$ready" ]; then echo 'Health check failed' >&2; exit 1; fi; \
	curl -fsS -X POST http://localhost:18080/predict \
		-H 'Content-Type: application/json' \
		-d @data/examples/sample-request.json
