FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
#1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /app
ADD requirements.txt ./
RUN pip install -r requirements.txt
ADD train.py graph.py evaluate.py ./
ADD bayeformers bayeformers/
ENTRYPOINT ["python3"]