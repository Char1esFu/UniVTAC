# UniVTAC devcontainer image.
# Stable layer only: OS + apt + locale + non-root user + miniforge.
# The heavy, GPU-dependent installs (Isaac Sim / Isaac Lab / TacEx / libuipc /
# cuRobo) are NOT here — they need a running container with the GPU visible and
# are run once via .devcontainer/setup-univtac-env.sh.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ARG USERNAME=charlesfu
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive

# --- system packages (mirrors the distrobox setup, minus ROS) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y --no-install-recommends \
        sudo ca-certificates \
        git git-lfs make cmake build-essential \
        wget curl zip unzip tar vim nano micro \
        pipx mesa-utils gawk \
        libglu1-mesa libxt6 \
        locales \
    && rm -rf /var/lib/apt/lists/*

# --- locale (en_US.UTF-8) ---
RUN locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# --- non-root user matching the host uid, so the bind-mounted workspace keeps
#     clean ownership; passwordless sudo to mirror the distrobox experience ---
RUN (groupadd --gid ${USER_GID} ${USERNAME} 2>/dev/null || true) \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /usr/bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# --- shell init that survives image rebuilds. conda lives on a named volume
#     (~/miniforge3) and vcpkg under another (~/deps), so we source them lazily
#     instead of relying on `conda init` writing into an ephemeral home. ---
RUN printf '%s\n' \
      '' \
      '# --- UniVTAC devcontainer shell init ---' \
      '[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ] && source "$HOME/miniforge3/etc/profile.d/conda.sh"' \
      '[ -f "$HOME/miniforge3/etc/profile.d/mamba.sh" ] && source "$HOME/miniforge3/etc/profile.d/mamba.sh"' \
      >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
WORKDIR /home/${USERNAME}
