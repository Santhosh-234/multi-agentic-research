
FROM python:3.11-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .
RUN pip install --upgrade pip


RUN pip install --no-cache-dir -r requirements.txt


COPY production_api.py .


EXPOSE 8000


CMD ["uvicorn", "production_api:app", "--host", "0.0.0.0", "--port", "8000"]