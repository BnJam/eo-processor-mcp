FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY eo_processor_mcp/ eo_processor_mcp/

RUN pip install --no-cache-dir .

ENTRYPOINT ["eo-processor-mcp"]
