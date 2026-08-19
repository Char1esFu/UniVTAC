# OS layer only; the env is built by setup-univtac-env.sh on first start
# (libuipc/cuRobo compile for the native GPU arch, invisible to `docker build`).
# No CUDA base image needed: nvcc 12.4 and gcc 11.4 come from conda.
FROM ubuntu:22.04

ARG USERNAME=charlesfu
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y --no-install-recommends \
        sudo ca-certificates \
        git git-lfs make cmake build-essential pkg-config \
        wget curl zip unzip tar vim nano micro \
        pipx mesa-utils gawk \
        libglu1-mesa libxt6 \
        locales \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Non-root user matching the host uid, so the workspace keeps clean ownership.
RUN (groupadd --gid ${USER_GID} ${USERNAME} 2>/dev/null || true) \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /usr/bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# conda lives on a named volume, so source it lazily rather than via `conda init`.
RUN printf '%s\n' \
      '' \
      'if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then' \
      '    . "$HOME/miniforge3/etc/profile.d/conda.sh"' \
      '    [ -f "$HOME/miniforge3/etc/profile.d/mamba.sh" ] && . "$HOME/miniforge3/etc/profile.d/mamba.sh"' \
      '    [ -d "$HOME/miniforge3/envs/UniVTAC" ] && conda activate UniVTAC' \
      'fi' \
      >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
WORKDIR /home/${USERNAME}
