FROM ubuntu:22.04
RUN apt update && apt install -y --no-install-recommends \
    libopenblas-openmp-dev liblapack-dev libscalapack-mpi-dev libelpa-dev libfftw3-dev libcereal-dev \
    libxc-dev libgtest-dev libgmock-dev libbenchmark-dev python3-numpy libparmetis-dev libsuperlu-dist-dev \
    bc cmake git g++ make bc time sudo unzip vim wget gfortran

ENV GIT_SSL_NO_VERIFY=true TERM=xterm-256color \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 OMPI_MCA_btl_vader_single_copy_mechanism=none

RUN git clone https://github.com/llohse/libnpy.git && \
    cp libnpy/include/npy.hpp /usr/local/include && \
    rm -r libnpy

RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip \
        --no-check-certificate --quiet -O libtorch.zip && \
    unzip -q libtorch.zip -d /opt  && rm libtorch.zip

RUN cp -r /usr/include/superlu-dist/* /usr/include && \
    wget https://bitbucket.org/berkeleylab/pexsi/downloads/pexsi_v2.0.0.tar.gz --no-check-certificate --quiet && \
    tar -xzf pexsi_v2.0.0.tar.gz && \
    cd pexsi_v2.0.0 && \
    cmake -B build -DPEXSI_ENABLE_FORTRAN=OFF -DPEXSI_ENABLE_OPENMP=ON && \
    cmake --build build -j`nproc` && \
    cmake --install build && \
    cd .. && rm -rf pexsi_v2.0.0

ENV CMAKE_PREFIX_PATH=/opt/libtorch/share/cmake

ADD https://api.github.com/repos/deepmodeling/abacus-develop/git/refs/heads/develop /dev/null

RUN git clone https://github.com/Flying-dragon-boxing/abacus-develop.git --depth 1 && \
    cd abacus-develop && \
    cmake -B build -DENABLE_DEEPKS=ON -DENABLE_LIBXC=ON -DENABLE_LIBRI=ON -DENABLE_PEXSI=ON && \
    cmake --build build -j`nproc` && \
    cmake --install build && \
    rm -rf build && \
    cd .. 
    #&& rm -rf abacus-develop
