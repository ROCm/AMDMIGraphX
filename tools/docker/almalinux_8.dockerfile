# AlmaLinux 8 image for building MIGraphX manylinux_2_28 compatible RPM packages.
#
# Mirrors hip-clang.docker but targets the el8 ROCm RPM repositories so that
# `make package` (CPack) emits RPMs usable across EL8/RHEL8/Rocky/AlmaLinux
# derivatives (glibc 2.28 -> manylinux_2_28).
FROM almalinux:8.10

ARG PREFIX=/usr/local

# Add ROCm 7.1.1 RPM repository for RHEL 8 / AlmaLinux 8
RUN printf '%s\n' \
        '[ROCm-7.1.1]' \
        'name=ROCm 7.1.1' \
        'baseurl=https://repo.radeon.com/rocm/rhel8/7.1.1/main' \
        'enabled=1' \
        'priority=50' \
        'gpgcheck=1' \
        'gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key' \
        > /etc/yum.repos.d/rocm.repo && \
    rpm --import https://repo.radeon.com/rocm/rocm.gpg.key

# Enable EPEL + PowerTools (CRB on EL8) so build dependencies resolve
RUN dnf install -y dnf-plugins-core epel-release && \
    dnf config-manager --set-enabled powertools && \
    dnf clean all

# Base build, packaging and language tooling.  rpm-build / rpmdevtools are
# required so that `make package` (CPack RPM generator) can emit RPMs.
# We install BOTH system gcc/gcc-c++ AND gcc-toolset-13:
#   * System gcc (8.5) gives ROCm's clang++ a default GCC toolchain it can
#     auto-discover under /usr/lib/gcc/x86_64-redhat-linux/8/ so that
#     `-lstdc++` resolves at link time (lld doesn't search gcc-toolset paths).
#   * gcc-toolset-13 supplies a modern C++17/20-capable libstdc++ that we
#     prefer for the host compiles via a ROCm clang config file below.
RUN dnf install -y --nobest \
        cmake \
        gcc \
        gcc-c++ \
        libstdc++-devel \
        gcc-toolset-13 \
        gcc-toolset-13-libatomic-devel \
        gdb \
        git \
        make \
        pkgconfig \
        which \
        wget \
        curl \
        file \
        rpm-build \
        rpmdevtools \
        redhat-rpm-config \
        zlib-devel \
        openssl-devel \
        numactl-devel \
        libgfortran \
        glibc-langpack-en \
        python3.12 \
        python3.12-devel \
        python3.12-pip && \
    dnf clean all

# Make python3.12 the default `python3` / `pip3`
RUN alternatives --set python3 /usr/bin/python3.12 && \
    alternatives --set python /usr/bin/python3.12 || true

# Activate gcc-toolset-13 for all subsequent layers and at container runtime
ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64 \
    MANPATH=/opt/rh/gcc-toolset-13/root/usr/share/man \
    PCP_DIR=/opt/rh/gcc-toolset-13/root

# Install ROCm runtime + devel packages used by MIGraphX
RUN dnf install -y --nobest \
        rocm-device-libs \
        hip-devel \
        miopen-hip \
        rocblas && \
    dnf clean all

# Workaround broken rocm packages
RUN ln -s /opt/rocm-* /opt/rocm
RUN echo "/opt/rocm/lib"      > /etc/ld.so.conf.d/rocm.conf
RUN echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf
RUN ldconfig

# Workaround broken miopen cmake files (note: lib64 on EL, not lib/x86_64-linux-gnu)
RUN sed -i 's,;/usr/lib64/librt.so,,g' /opt/rocm/lib/cmake/miopen/miopen-targets.cmake

# Point ROCm's clang/clang++ at gcc-toolset-13 so it picks up the newer
# libstdc++ headers and link library.  Without this, clang scans /usr for a
# gcc install and falls back to the EL8 system gcc 8.5 libstdc++.
# A clang config file alongside the binary is read on every invocation.
RUN printf '%s\n' '--gcc-toolchain=/opt/rh/gcc-toolset-13/root/usr' \
        | tee /opt/rocm/llvm/bin/clang.cfg \
              /opt/rocm/llvm/bin/clang++.cfg \
              /opt/rocm/llvm/bin/clang-cpp.cfg \
        > /dev/null

# Workaround for distributions running cmake < 3.25 (EL8 ships cmake 3.20)
RUN sed -i -e 's/^block/if(COMMAND block)\nblock/g' \
           -e 's/^endblock/endblock\(\)\nendif/g' \
           /opt/rocm/lib/cmake/hipblaslt/hipblaslt-config.cmake

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install yapf
RUN pip3 install yapf==0.28.0

# Install doc requirements
ADD docs/sphinx/requirements.txt /doc-requirements.txt
RUN pip3 install -r /doc-requirements.txt

# Install dependencies via rbuild (handled by install_prereqs.sh which now has
# a dnf branch for RHEL-like distros)
ADD dev-requirements.txt /dev-requirements.txt
ADD requirements.txt /requirements.txt
ADD rbuild.ini /rbuild.ini

COPY ./tools/install_prereqs.sh /
COPY ./tools/requirements-py.txt /
RUN /install_prereqs.sh /usr/local / && rm /install_prereqs.sh && rm /requirements-py.txt
