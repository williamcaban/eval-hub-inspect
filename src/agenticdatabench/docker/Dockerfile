FROM python:3.12-slim

# System deps: build tools (for statsmodels/scipy source builds on some
# architectures) + runtime libs for OpenCV and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Prefer binary wheels; fall back to source build only when necessary
# Install in order: heavy numerical stack first so layers are reused
RUN pip install --no-cache-dir --prefer-binary \
    "pandas>=2.0.0" \
    "numpy>=1.26.0" \
    "scipy>=1.12.0" \
    "scikit-learn>=1.5.0" \
    "statsmodels>=0.14.0" \
    "joblib>=1.3.0"

RUN pip install --no-cache-dir --prefer-binary \
    "fuzzywuzzy>=0.18.0" \
    "python-Levenshtein>=0.25.0" \
    "Pillow>=12.3.0" \
    "opencv-python-headless>=4.8.1.78" \
    matplotlib \
    nbformat \
    jsonlines

# Non-root user for security
RUN useradd -m -u 1001 agent
USER agent
