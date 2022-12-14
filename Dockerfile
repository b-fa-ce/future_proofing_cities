# FROM python:3.10.8-slim
FROM tensorflow/tensorflow:2.10.0

COPY modules /modules
COPY requirements_prod.txt /requirements.txt
COPY setup.py /setup.py
COPY README.md /README.md

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN pip install .

RUN python setup.py install

RUN rm -rf *.egg-info

CMD uvicorn  modules.api.fast_api:app --host 0.0.0.0  --port $PORT
