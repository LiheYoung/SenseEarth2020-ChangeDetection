FROM nvidia/cuda:10.2-runtime-ubuntu18.04
FROM python:3.7
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD CUDA_VISIBLE_DEVICES=0 python3 test.py
