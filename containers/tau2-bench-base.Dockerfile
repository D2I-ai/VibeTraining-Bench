# syntax=docker/dockerfile:1
# Fixed Tau2 dependency image shared by Tau3 task environments.
ARG PYTHON_BASE_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
FROM ${PYTHON_BASE_IMAGE}

# The build entrypoint normalizes VTB_DOMESTIC_NETWORK_MODE to 0 or 1.
ARG VTB_DOMESTIC_NETWORK_MODE=0
ARG PIP_INDEX_URL=
ARG TAU2_BENCH_COMMIT
ARG LITELLM_VERSION=1.83.0
ARG MCP_VERSION=1.25.0
ARG FASTMCP_VERSION=3.0.0

LABEL org.opencontainers.image.title="vibetrainbench Tau2 base" \
      org.opencontainers.image.revision="${TAU2_BENCH_COMMIT}" \
      org.vibetrainbench.litellm.version="${LITELLM_VERSION}" \
      org.vibetrainbench.mcp.version="${MCP_VERSION}" \
      org.vibetrainbench.fastmcp.version="${FASTMCP_VERSION}"

ENV TAU2_BENCH_ROOT=/opt/tau2-bench \
    TAU2_DATA_DIR=/opt/tau2-bench/data

WORKDIR ${TAU2_BENCH_ROOT}
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

# The default path is upstream PyPI. Domestic mode changes only this build layer;
# an explicit PIP_INDEX_URL always has precedence. Tau2 1.0.1 declares
# litellm<1.82.7, but Harbor's Tau3 agent intentionally installs 1.83.0 before every
# trial. Apply that tested override separately from the Tau2 resolver transaction.
RUN if [ -n "$PIP_INDEX_URL" ]; then \
        index_url="$PIP_INDEX_URL"; \
    elif [ "$VTB_DOMESTIC_NETWORK_MODE" = "1" ]; then \
        index_url="https://repo.huaweicloud.com/repository/pypi/simple"; \
    else \
        index_url="https://pypi.org/simple"; \
    fi \
    && echo "[tau2-image] domestic_network_mode=$VTB_DOMESTIC_NETWORK_MODE pip_index=$index_url" \
    && python3 -m pip install --no-cache-dir --index-url "$index_url" \
        ".[knowledge]" \
        "mcp==${MCP_VERSION}" \
        "fastmcp==${FASTMCP_VERSION}" \
    && python3 -m pip install --no-cache-dir --no-deps --index-url "$index_url" \
        "litellm==${LITELLM_VERSION}"

# Harbor's Tau3 adapter runs an exact pip install before each trial. These versions are
# already present, and no-index turns future drift into a local error rather than a hidden
# PyPI dependency during evaluation.
ENV PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
