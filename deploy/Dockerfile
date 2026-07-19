# Image for Hugging Face Spaces (SDK: docker, app_port: 7860).
#
# Local development stays on uv -- this file exists only because a Space needs a
# container. It is the one place Docker appears in the project.
#
# The FASTA path needs three native tools. BLAST+ and HMMER come from Debian
# rather than the 282 MB tarball and a source build that install_tools.sh uses on
# a laptop; only AMRFinderPlus has no package and is fetched as a binary.
#
# AMRFinderPlus v4 links against GLIBCXX_3.4.32 (gcc 13+). Debian trixie has
# gcc 14, so v4 runs here even though the same binary fails on Ubuntu 22.04.

FROM python:3.12-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/opt/amrfinder:$PATH" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ncbi-blast+ hmmer curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# AMRFinderPlus binaries, newest release that runs on this base.
ARG AMR_VERSION=4.2.7
RUN mkdir -p /opt/amrfinder \
    && curl -fsSL "https://github.com/ncbi/amr/releases/download/amrfinder_v${AMR_VERSION}/amrfinder_binaries_v${AMR_VERSION}.tar.gz" \
       | tar -xz -C /opt/amrfinder \
    && /opt/amrfinder/amrfinder --version

# The AMR database is ~230 MB and versioned. Baking it in keeps the Space's cold
# start fast and pins the annotation to one database version, which matters:
# training and serving on different databases shifts the feature space.
RUN amrfinder -u && amrfinder --version && ls /opt/amrfinder/data

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY app/ app/
COPY config/ config/
COPY models/ models/
COPY data/demo/ data/demo/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# Spaces runs as a non-root user; uploads and reports are written under /app.
RUN mkdir -p /app/data/store && chmod -R 777 /app/data

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=7860", \
     "--server.headless=true", "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=50"]
