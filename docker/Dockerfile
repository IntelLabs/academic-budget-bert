FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.12-cuda11.3.1
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /opt/ml/code
