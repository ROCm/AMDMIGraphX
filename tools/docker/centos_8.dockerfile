FROM centos:latest

ARG PREFIX=/usr/local

# Support multiarch
RUN rpm --target i386

RUN cd /etc/yum.repos.d/
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

# Add rocm repository
RUN sh -c 'echo -e "[rocm]\nname=rocm\nbaseurl=http://repo.radeon.com/rocm/centos8/5.2.5/main/\nenabled=1\ngpgcheck=1\ngpgkey=http://repo.radeon.com/rocm/rocm.gpg.key" > /etc/yum.repos.d/rocm.repo'

RUN yum clean all && \
    yum install -y wget \
    dnf-plugins-core \
    clang \
    epel-release

#For CentOS 8.3/8.4
RUN sh -c 'echo yum config-manager --set-enabled PowerTools'

# Install dependencies
RUN dnf update  -y && dnf makecache --refresh
RUN dnf groupinstall -y "Development Tools" --setopt=group_package_types=mandatory,default,optional

RUN dnf --enablerepo=powertools install -y --nobest \
    yum-utils \
    git-clang-format \
    doxygen \
    lcov \
    glibc-all-langpacks \
    libffi-devel \
    xz-devel \
    bzip2-devel \
    python38 \
    python38-devel \
    rocm-device-libs \
    hip-base \
    numactl \
    miopen-hip \
    miopen-hip-devel \
    rocblas \
    rocblas-devel \
    hipfft \
    rocthrust \
    rocrand \
    hipsparse \
    rccl \
    rccl-devel \
    rocm-smi-lib \
    rocm-dev \
    roctracer-dev \
    hipcub  \
    hipblas  \
    hipify-clang \
    half \
    openssl-devel


RUN alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 60 && alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 50 && update-alternatives --auto python3
RUN python3 --version && ls -la /usr/bin

# add this for roctracer dependancies
RUN pip3.8 install CppHeaderParser

# Workaround broken rocm packages
RUN ln -s /opt/rocm-* /opt/rocm
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf
RUN echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf
RUN ldconfig

#RUN localedef -c -f UTF-8 -i en_US en_US.UTF-8
#RUN export LC_ALL=en_US.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

#update pip
RUN pip3.8 install --upgrade pip 

# Install dependencies
ADD dev-requirements.txt /dev-requirements.txt
ADD requirements.txt /requirements.txt
ADD rbuild.ini /rbuild.ini

COPY ./tools/install_prereqs_centos.sh /
RUN /install_prereqs_centos.sh /usr/local / && rm /install_prereqs_centos.sh
RUN test -f /usr/local/hash || exit 1

# Install yapf
RUN pip3.8 install yapf==0.28.0

# Install doc requirements
ADD doc/requirements.txt /doc-requirements.txt
RUN pip3.8 install -r /doc-requirements.txt

# Download real models to run onnx unit tests
ENV ONNX_HOME=/.onnx
COPY ./tools/download_models.sh /
RUN /download_models.sh && rm /download_models.sh

# Install latest ccache version
RUN cget -p $PREFIX install facebook/zstd@v1.4.5 -X subdir -DCMAKE_DIR=build/cmake
RUN cget -p $PREFIX install ccache@v4.1 -DENABLE_TESTING=OFF
RUN cget -p /opt/cmake install kitware/cmake@v3.24.3

COPY ./test/onnx/.onnxrt-commit /

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=main
ARG ONNXRUNTIME_COMMIT

RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    cd onnxruntime && \
    if [ -z "$ONNXRUNTIME_COMMIT" ] ; then git checkout $(cat /.onnxrt-commit) ; else git checkout ${ONNXRUNTIME_COMMIT} ; fi && \
    /bin/sh /onnxruntime/dockerfiles/scripts/install_common_deps.sh


ADD tools/build_and_test_onnxrt.sh /onnxruntime/build_and_test_onnxrt.sh

RUN cget -p /usr/local install ROCmSoftwarePlatform/rocMLIR@acb727b348086b58a7f261b32c0e4f0686a4c0ee -DBUILD_MIXR_TARGET=On -DLLVM_ENABLE_ZSTD=Off -DLLVM_ENABLE_THREADS=Off

ENV MIOPEN_FIND_DB_PATH=/tmp/miopen/find-db
ENV MIOPEN_USER_DB_PATH=/tmp/miopen/user-db
ENV LD_LIBRARY_PATH=$PREFIX/lib

# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1
ENV ASAN_OPTIONS=detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
RUN ln -s /opt/rocm/llvm/bin/llvm-symbolizer /usr/bin/llvm-symbolizer

