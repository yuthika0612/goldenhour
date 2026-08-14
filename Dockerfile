# Golden Hour — production image.
#
# tesseract-ocr is a system binary, not a Python package, so a plain
# "Python runtime" host (Render's native buildpack, etc.) cannot install
# it. Docker is used specifically so OCR-first actually works in
# production, not only on a developer's laptop where tesseract happens to
# already be installed.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Render sets $PORT at runtime; server.py already reads it.
CMD ["python", "web/server.py"]
