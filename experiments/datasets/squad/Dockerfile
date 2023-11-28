FROM python:3.8
WORKDIR /app
ADD requirements.txt .
RUN pip install -r requirements.txt
ADD . .
ENTRYPOINT ["python3", "/app/process_squad.py"]