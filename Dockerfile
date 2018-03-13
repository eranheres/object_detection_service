FROM tensorflow/tensorflow:1.4.0-gpu
RUN apt-get update && apt-get install -y protobuf-compiler python-pil python-lxml python-tk git
RUN pip install jupyter matplotlib

RUN git clone https://github.com/tensorflow/models.git
WORKDIR models/research/
COPY patch/all.patch .
RUN patch -p0 -i all.patch
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH=$PYTHONPATH:/notebooks/models/research:/notebooks/models/research/slim
RUN python setup.py sdist
RUN (cd slim && python setup.py sdist)

RUN mkdir /usr/mnt
