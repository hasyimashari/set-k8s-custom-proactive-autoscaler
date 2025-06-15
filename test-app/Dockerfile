FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir flask prometheus_client

COPY test-app.py .

EXPOSE 5000

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

CMD ["python", "test-app.py"]
