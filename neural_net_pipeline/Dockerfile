FROM python:3.8.15
LABEL authors="SamuraX <https://github.com/MatMatias>"
USER root
WORKDIR /app
COPY ./ ./

# install necessary python dependecies
RUN pip install nltk scikit-learn skipthoughts certifi==2022.5.18.1 tensorflow numpy scipy matplotlib ipython jupyter pandas sympy nose torch torchvision tensorflow_hub torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# install punkt training model
RUN python -m nltk.downloader punkt
