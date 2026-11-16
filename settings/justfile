set shell := ["bash", "-cu"]

setup:
    python -m pip install -r requirements.txt

data:
    python scripts/download.py

eda:
    python eda.py

train:
    python train.py

test:
    pytest -q

serve:
    uvicorn app:app --reload --port 8080

docker-build:
    docker build -t mlzc-salary .

docker-run:
    docker run --rm -p 8080:8080 mlzc-salary
