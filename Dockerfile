FROM tensorflow/tensorflow:2.11.0

COPY modules /modules
COPY data/training_outputs /data/training_outputs

COPY data/processed_data/Paris/Paris_full.csv /data/processed_data/Paris/Paris_full.csv
COPY data/processed_data/Paris/Paris_full.csv /data/processed_data/Berlin/Berlin_full.csv


COPY requirements_prod.txt /requirements.txt
COPY setup.py /setup.py
COPY README.md /README.md

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN pip install .

RUN python setup.py install

RUN rm -rf *.egg-info

CMD uvicorn  modules.api.fast_api:app --host 0.0.0.0  --port $PORT
