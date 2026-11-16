.PHONY: setup data eda train test serve docker-build docker-run

setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -r requirements.txt

data:
	uv run --python .venv/bin/python scripts/download.py

eda:
	uv run --python .venv/bin/python eda.py

train:
	uv run --python .venv/bin/python train.py

test:
	uv run --python .venv/bin/python -m pytest -q

serve:
	uv run --python .venv/bin/python -m uvicorn app:app --reload --port 8080

docker-build:
	docker build -t mlzc-salary .

docker-run:
	docker run --rm -p 8080:8080 mlzc-salary
