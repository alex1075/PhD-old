FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04
#CMD nvidia-smi

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get install -y git

EXPOSE 22
#COPY requirements.txt requirements.txt
#RUN  pip install -r requirements.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]
