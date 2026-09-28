FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/fluoroscopy-alert-takeover

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests
RUN pip install --no-cache-dir -e .

# Offline causal estimation is the only stage the container is sized for: the manuscript
# reports 18.4 min per 10,000 procedures on 8 CPU cores (Table A1), so the default image
# carries no accelerator runtime. The perception route is trained outside the image.
ENV OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

ENTRYPOINT ["python", "-m", "fluoroscopy_alert_takeover.cli.estimate"]
