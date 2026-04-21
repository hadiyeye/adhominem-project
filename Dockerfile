FROM continuumio/miniconda3:latest

# System tools (fastText needs a compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential g++ make \
    p7zip-full \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Create a Python 3.6 env for this old project
RUN conda create -n adhominem -y python=3.6 && conda clean -afy

# Install Python deps inside the env
RUN /bin/bash -lc "conda activate adhominem && \
    conda config --add channels conda-forge && \
    conda install -y fasttext && \
    pip install --upgrade 'pip<22' setuptools wheel && \
    pip install \
      tensorflow==1.15.5 \
      spacy==2.3.2 \
      textacy==0.8.0 \
      numpy==1.18.5 \
      scipy==1.4.1 \
      pandas==1.0.5 \
      scikit-learn==0.20.4 \
      bs4==0.0.1 && \
    python -m spacy download en_core_web_lg"

# Always start with the env activated
SHELL ["/bin/bash", "-lc"]
RUN echo "conda activate adhominem" >> ~/.bashrc

CMD ["/bin/bash"]

