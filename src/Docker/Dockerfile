FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-get update && \
	apt-get install -y git \
	libx11-6 \
	libgl1 \
	libopengl0 \
	libegl1 \
	wget

RUN mkdir /app
WORKDIR /app

RUN wget https://github.com/RomainUSA/CBCT_seg/archive/refs/tags/v3.1.1.zip
RUN unzip v3.1.1.zip
RUN mv CBCT_seg-3.1.1/src src
RUN rm -rf v3.1.1.zip \
	rm -rf CBCT_seg-3.1.1

RUN pip3 install -r /app/src/requirements.txt

RUN mkdir /app/model
WORKDIR /app/model

RUN wget https://github.com/RomainUSA/CBCT_seg/releases/download/v3.1.1/MandSeg_Final_35.hdf5
RUN wget https://github.com/RomainUSA/CBCT_seg/releases/download/v3.1.1/RCSeg_Final_50.hdf5

WORKDIR /app

RUN mkdir data data/preproc data/predicted 

RUN chmod +x src/sh/*.sh
ENV PATH=/app/src/sh:$PATH