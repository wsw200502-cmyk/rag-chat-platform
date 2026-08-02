FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY models /app/models
EXPOSE 8000
CMD ["python", "agent_api.py"]
