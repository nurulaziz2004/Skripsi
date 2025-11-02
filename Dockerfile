FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir psycopg2-binary redis flask
CMD ["python", "utama_realtime.py"]