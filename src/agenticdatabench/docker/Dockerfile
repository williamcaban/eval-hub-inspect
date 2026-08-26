FROM python:3.12-slim

# Install system deps needed by OpenCV, scipy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install data science stack matching the optional deps in pyproject.toml
RUN pip install --no-cache-dir \
    pandas>=2.0.0 \
    numpy>=1.26.0 \
    "scikit-learn>=1.5.0" \
    "fuzzywuzzy>=0.18.0" \
    "python-Levenshtein>=0.25.0" \
    "scipy>=1.12.0" \
    "Pillow>=12.3.0" \
    "statsmodels>=0.14.0" \
    "joblib>=1.3.0" \
    "opencv-python-headless>=4.8.1.78" \
    matplotlib \
    nbformat \
    jsonlines

# Non-root user for security
RUN useradd -m -u 1001 agent
USER agent
