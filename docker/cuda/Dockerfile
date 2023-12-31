################################################# Image #################################################
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

############################################# Date and Time #############################################
ENV TZ="America/Sao_Paulo"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* 

RUN apt-get update -y && apt-get install -y locales curl wget tar sudo git apt-utils \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

############################################# DEPENDENCIES ##############################################
RUN apt-get install -y gcc g++ graphviz vim python3 python3-pip python3-dev automake make \
    build-essential cmake llvm-dev clang cython3 python-is-python3 libedit-dev libxml2-dev libtinfo-dev \
    python3-setuptools

RUN pip3 install numpy==1.24.4 decorator scipy pytest psutil typing_extensions tornado cloudpickle \ 
    'xgboost>=1.1.0' synr mxnet 
RUN pip3 install transformers torch==1.13.1 onnx==1.15.0

########################################### INSTALL TVM #################################################
RUN cd $HOME && git clone -b v0.13.0 --recursive https://github.com/apache/tvm 
RUN cd $HOME/tvm && mkdir -p build && \
    wget https://raw.githubusercontent.com/lac-dcc/DropletSearch/main/docker/cuda/config.cmake && \
    wget https://raw.githubusercontent.com/lac-dcc/bennu/main/src/run_state.cc && \
    wget https://raw.githubusercontent.com/lac-dcc/bennu/main/src/run_state.h && \
    mv config.cmake $HOME/tvm/build && \
    mv run_state.* $HOME/tvm/src/auto_scheduler/ && \ 
    cd build && cmake .. && make -j4 

RUN echo "export PYTHONPATH=/root/tvm/python:/python:" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.0/compat/" >> ~/.bashrc

########################################### WORKDIR #####################################################
WORKDIR /root/bennu/
ENTRYPOINT ["/bin/bash"]
