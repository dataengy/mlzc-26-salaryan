FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
COPY settings/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY settings/config.yml ./settings/config.yml
COPY src/mlzc_26_salaryan ./src/mlzc_26_salaryan
COPY models/model.joblib ./models/model.joblib
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn mlzc_26_salaryan.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
