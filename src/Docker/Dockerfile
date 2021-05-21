FROM python:3.7

RUN apt-get update && \
	apt-get install -y git

RUN mkdir /app
WORKDIR /app

COPY ./master/CBCT_seg CBCT_seg

COPY ./model_docker /app/model

RUN pip3 install -r /app/CBCT_seg/src/py/requirements.txt

# RUN wget https://github.com/celiale/OAI/archive/refs/tags/v2.0.zip
# RUN unzip v2.0.zip
# RUN mv OAI-2.0 OAI
# RUN rm -rf v2.0.zip \
# 	rm -rf OAI-2.0

# WORKDIR /app/OAI/python
# RUN pip3 install -r ./requirements.txt
# RUN wget https://github.com/celiale/OAI/releases/download/v2.0/Models.zip
# RUN unzip Models.zip

# docker run --rm -v '/Users/luciacev-admin/Documents/MandSeg/example':/app/scans mandseg:latest bash /app/CBCT_seg/src/sh/main_prediction.sh --dir_src /app/CBCT_seg --dir_input /app/scans --dir_model /app/model --model_name MandSeg_256_FullDA --num_epoch 40