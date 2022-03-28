FROM python:3.7

COPY data data
COPY syslevel syslevel
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY reproducibility/run.sh run.sh

CMD ["bash", "run.sh"]
