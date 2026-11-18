# The ResolveLane API. The local embedding model is part of the image, so the container starts
# and recognizes issues without a network connection, as the local tier is meant to.
#
#   docker compose up --build        (with the web interface; see compose.yaml)

FROM python:3.14.10-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv
WORKDIR /app

# Dependencies before the code, so changing the code does not reinstall them. The package is
# installed in editable mode with an empty stand-in: the application finds its skills/ and
# evals/ directories next to src/, so the code must run from /app/src, not site-packages.
COPY pyproject.toml ./
RUN mkdir -p src/resolvelane && touch src/resolvelane/__init__.py \
 && pip install -e '.[local-embedding,mcp]'

# The local embedding model, downloaded once at build time.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/models')"


FROM python:3.14.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    RESOLVELANE_MODEL_CACHE_DIR=/opt/models \
    RESOLVELANE_DATABASE=/app/data/resolvelane.sqlite3 \
    RESOLVELANE_SKILLS_DIR=/app/skills

RUN useradd --create-home --uid 1000 resolvelane
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=resolvelane:resolvelane /opt/models /opt/models

WORKDIR /app
COPY pyproject.toml mcp_servers.example.json ./
COPY skills ./skills
COPY evals ./evals
COPY src ./src
# Only the case database is written at run time.
RUN mkdir data && chown resolvelane:resolvelane data
USER resolvelane

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "resolvelane.api:app", "--host", "0.0.0.0", "--port", "8000"]
