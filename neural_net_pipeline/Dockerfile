FROM python:3.7.2
LABEL authors="SamuraX <https://github.com/MatMatias>"
USER root
WORKDIR /app
COPY ./ ./

# install necessary python dependecies
RUN pip install nltk skipthoughts certifi==2022.5.18.1 tensorflow==1.13.2 numpy scipy matplotlib ipython jupyter pandas sympy nose torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# install punkt training model
RUN python -m nltk.downloader punkt
