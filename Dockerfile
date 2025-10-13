FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads /app/tmp_processed && \
    cp -r .streamlit/* /home/appuser/ && \
    chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 7860
CMD ["streamlit", "run", "application/app.py", "--server.addess=0.0.0.0", "--server.port=7860"]
