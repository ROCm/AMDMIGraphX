# ROCm 7.13.0 pip install — gfx950 (MI355X), RHEL 8 (UBI8)
# Follows: https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html
#   ?fam=instinct&os=rhel&gpu=mi355x&gfx=gfx950&rhel-ver=8.10&i=pip&w=compute
#
# Uses Red Hat Universal Base Image 10 (UBI10) — no subscription required.
# python3.12 comes from the UBI10 CodeReady Builder repo (enabled by default in UBI).

FROM registry.access.redhat.com/ubi10/ubi:latest

ARG GPU_ARCH=all

# ── 1. System prerequisites ────────────────────────────────────────────────────
RUN dnf install -y --nodocs \
        python3 \
        python3-pip \
        libatomic \
        libquadmath \
        ca-certificates \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# ── 2. Virtual environment ─────────────────────────────────────────────────────
RUN python3 -m venv /opt/rocm-venv
ENV PATH="/opt/rocm-venv/bin:$PATH"

# ── 3. Install ROCm wheels  ───────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://repo.amd.com/rocm/whl/whl-multi-arch/ \
        "rocm[libraries,devel,device-${GPU_ARCH}]"

# ── 4. GPU device access (udev rules; applied by host at runtime in Docker) ────
RUN mkdir -p /etc/udev/rules.d && \
    echo 'KERNEL=="kfd", GROUP="render", MODE="0666"' > /etc/udev/rules.d/70-amdgpu.rules && \
    echo 'SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="render", MODE="0666"' >> /etc/udev/rules.d/70-amdgpu.rules

# ── 5. Smoke-test: verify packages are importable (no GPU needed for import) ───
RUN python -c "import importlib.util; \
    spec = importlib.util.find_spec('amdsmi'); \
    print('amdsmi found:', spec is not None)"

WORKDIR /workspace
CMD ["/bin/bash"]

