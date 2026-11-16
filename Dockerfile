FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/mlzoomcamp_salary_minimal ./src/mlzoomcamp_salary_minimal
COPY models/model.joblib ./models/model.joblib
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn mlzoomcamp_salary_minimal.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
