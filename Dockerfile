# The web app, pinned to the Python it was tested on.
#
# Built and smoke-tested by GitHub Actions on every push (.github/workflows/
# ci.yml), and deployed from the same file by Render. What CI proves is what
# ships -- there is no second recipe that could drift from this one.
#
#   docker build -t nifty-iv .
#   docker run -p 8000:8000 nifty-iv

FROM python:3.12-slim

# Fail fast and log straight through, rather than buffering output that then
# vanishes if the container is killed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Requirements before the code, deliberately. Docker caches each step, so this
# ordering means editing main.py does not reinstall pandas and scipy --
# rebuilds drop from minutes to seconds. Copying everything first would
# invalidate the cache on every single change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# An ordinary user, not root. If anything in here is ever compromised, root
# inside a container is a far better starting point for an attacker than a
# user who owns nothing but /app.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Shell form on purpose, so ${PORT} expands. Render assigns a port at runtime
# and an app that hardcodes 8000 never answers. The default keeps
# `docker run -p 8000:8000` working locally.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
