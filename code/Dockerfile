FROM python:3.11-slim
RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get install libpq-dev -y \
    && rm -rf /var/lib/apt/lists/*
ADD . /
RUN pip install --no-cache-dir .
CMD ["python", "valorization/api/main.py"]
