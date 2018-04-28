FROM nvidia/cuda:9.1-cudnn7-runtime
LABEL maintainer="nykergoto@gmail.com"

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git zsh sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# userの追加
RUN groupadd -g 1000 developer

# user:penguin, password:highway
RUN useradd -g developer -G sudo -m -s /bin/bash penguin
RUN echo "penguin:highway" | chpasswd

WORKDIR /home/penguin/
USER penguin

# install miniconda and activate
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh && \
    echo "export PATH=${PATH}:${HOME}/miniconda/bin" >> .bash_profile


ENV PATH ${PATH}:/home/penguin/miniconda/bin
RUN conda update -y conda

# install pytorch for cuda9.1
RUN conda install -y \
    pytorch \
    torchvision \
    cuda91 \
    -c pytorch

RUN conda install -y \
    numpy \
    jupyter \
    notebook \
    pillow \
    matplotlib

CMD ["/bin/bash"]