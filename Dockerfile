# Sandbox for running EvaLoop with API models (e.g. via OpenRouter).
#
# Model-generated code is executed by EvalPlus, whose guards are Unix-only and are
# explicitly "NOT a security sandbox". This image runs everything as an unprivileged
# user in a container that sees only the code and a mounted results directory.
# See "Running in Docker" in README.md for the run command.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only torch: the model modules import it, but API models never use it.
# The base image's pip rejects the PyTorch index's wheel names, so upgrade it first;
# torch's own dependencies come from PyPI.
RUN pip install --upgrade "pip==24.2" && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple "torch==2.6.0"

WORKDIR /app
COPY requirements.txt ./
# evalplus and openai are pinned to the versions this fork was tested against
RUN pip install -r requirements.txt "evalplus==0.3.1" "openai==3.42.1" "pytest==8.3.3" "ruff"

RUN useradd --create-home --uid 1000 app && mkdir -p /app/results /app/logs && chown app:app /app /app/results /app/logs
COPY --chown=app:app . .
RUN pip install --no-deps -e . && pip freeze > /app/pip-freeze.txt

USER app
# Download MBPP+ and compute its ground truth at build time, so runs are offline-reproducible
RUN python -c "from evaloop.evaluation.py_testing import load_mbpp_dataset; p, _ = load_mbpp_dataset(); print(len(p), 'MBPP+ tasks')"

ENTRYPOINT ["evaloop"]
CMD ["--help"]
