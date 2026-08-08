# Multi-stage build: builder compiles the Rust extension in eo-processor
# (PyPI has no manylinux cp311/cp312 wheel yet), runtime ships a slim image.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Rust toolchain — required to build the eo-processor sdist from PyPI.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy only what hatchling needs to build the wheel.
COPY pyproject.toml README.md LICENSE ./
COPY eo_processor_mcp/ eo_processor_mcp/

# Install into a separate prefix so the runtime stage can copy only what's needed.
RUN pip install --no-cache-dir --prefix=/install .

# Runtime stage: no Rust, no build tools.
FROM python:3.12-slim

COPY --from=builder /install/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /install/bin/eo-processor-mcp /usr/local/bin/eo-processor-mcp

ENTRYPOINT ["eo-processor-mcp"]