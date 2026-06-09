FROM python:3.11-slim

WORKDIR /app

RUN groupadd --system arl && useradd --system --gid arl --create-home --home-dir /home/arl arl

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[serving,prometheus]" \
    && mkdir -p /app/results/sidecar \
    && chown -R arl:arl /app

USER arl

ENV PYTHONUNBUFFERED=1
ENV ARL_CONFIG=/app/configs/serving_pilot_fraud_torch.yaml

EXPOSE 8080 9091

CMD ["python3", "scripts/run_serve.py", "--config", "configs/serving_pilot_fraud_torch.yaml", "--host", "0.0.0.0", "--port", "8080"]
