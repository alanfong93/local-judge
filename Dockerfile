FROM python:3.13-slim AS application

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY docs/ ./docs/

RUN python -m pip install --no-cache-dir ".[container]"
RUN groupadd --gid 10001 localjudge \
    && useradd --uid 10001 --gid localjudge --create-home --shell /usr/sbin/nologin localjudge

FROM application AS test

COPY tests/ ./tests/
RUN python -m pip install --no-cache-dir ".[dev]"
RUN python -m pytest

FROM application AS runtime

ENV HOME=/home/localjudge
USER localjudge
EXPOSE 8000
ENTRYPOINT ["python", "-m", "local_judge.deployment"]
CMD ["http"]
