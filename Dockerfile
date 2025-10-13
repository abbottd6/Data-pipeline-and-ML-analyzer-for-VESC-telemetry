FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["streamlit", "run", "application/app.py", "--server.address=0.0.0.0", "--server.port=7860"]
