#!/usr/bin/env bash
# Builds the UniVTAC environment per docs/Installation.md. Run by
# postCreateCommand; safe to re-run, each stage stamps ~/deps/.state and is
# skipped afterwards.
#
#     UNIVTAC_FORCE=tacex_uipc bash .devcontainer/setup-univtac-env.sh   # redo one stage
#     UNIVTAC_FORCE=all        bash .devcontainer/setup-univtac-env.sh   # redo everything
set -Eeo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="$HOME/miniforge3"
DEPS_DIR="$HOME/deps"
VCPKG_DIR="$DEPS_DIR/vcpkg"
STATE_DIR="$DEPS_DIR/.state"
ENV_NAME="UniVTAC"

mkdir -p "$DEPS_DIR" "$STATE_DIR"
exec > >(tee -a "$STATE_DIR/setup.log") 2>&1

stage() {
    local name="$1"; shift
    local force="${UNIVTAC_FORCE:-}"
    if [ -f "$STATE_DIR/$name.done" ] && [[ ",$force," != *",all,"* && ",$force," != *",$name,"* ]]; then
        echo "--- $name: done, skipping"
        return 0
    fi
    printf '\n\033[1;36m>>> %s\033[0m\n' "$name"
    "$@"
    touch "$STATE_DIR/$name.done"
}

activate_env() {
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
}

s_miniforge() {
    mkdir -p "$CONDA_DIR"
    local installer="$CONDA_DIR/miniforge.sh"
    wget -qO "$installer" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash "$installer" -b -u -p "$CONDA_DIR"
    rm -f "$installer"
    # shellcheck disable=SC1091
    source "$CONDA_DIR/bin/activate"
    conda init --all
    conda config --set auto_activate_base false
    mamba shell init --shell bash --root-prefix "$CONDA_DIR"
}

s_conda_env() {
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda env list | grep -qE "^\s*${ENV_NAME}\s" || conda create -n "$ENV_NAME" python=3.10 -y
}

s_repo() {
    git config --global --add safe.directory "$REPO_DIR"
    git config --global --add safe.directory "$REPO_DIR/third_party/TacEx"
    git lfs install --skip-repo
    git -C "$REPO_DIR/third_party/TacEx" lfs pull
}

s_isaacsim() {
    activate_env
    pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    pip install --upgrade pip
    pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
    # flatdict 4.0.1 breaks on setuptools >= 81; the pin has to stay.
    pip install "setuptools<81" wheel
    pip install --no-build-isolation flatdict==4.0.1
    pip install transforms3d
}

s_vcpkg() {
    [ -d "$VCPKG_DIR/.git" ] || git clone https://github.com/microsoft/vcpkg.git "$VCPKG_DIR"
    "$VCPKG_DIR/bootstrap-vcpkg.sh" -disableMetrics
}

# TacEx core, from the modified source bundled in third_party/.
s_tacex_core() {
    activate_env
    cd "$REPO_DIR/third_party/TacEx"
    ./tacex.sh -i
}

# libuipc build deps: CMake 3.26 / GCC 11.4 / CUDA 12.4, all in-env.
s_uipc_deps() {
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda env update -n "$ENV_NAME" --file "$REPO_DIR/third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml"
    conda install -n "$ENV_NAME" -c conda-forge -y sysroot_linux-64=2.34 ffmpeg
}

# Builds libuipc + python bindings. CMAKE_CUDA_ARCHITECTURES unset means
# "native", so this needs the GPU.
s_tacex_uipc() {
    activate_env
    cd "$REPO_DIR/third_party/TacEx"
    pip install -e source/tacex_uipc -v
}

s_isaaclab() {
    activate_env
    [ -d "$DEPS_DIR/IsaacLab/.git" ] || git clone https://github.com/isaac-sim/IsaacLab "$DEPS_DIR/IsaacLab"
    cd "$DEPS_DIR/IsaacLab"
    git fetch --tags
    git checkout v2.1.1
    # Pin first: isaaclab.sh would otherwise pull an incompatible sb3.
    pip install "stable-baselines3==2.7.0"
    ./isaaclab.sh -i
}

s_curobo() {
    activate_env
    [ -d "$DEPS_DIR/curobo/.git" ] || git clone https://github.com/NVlabs/curobo.git "$DEPS_DIR/curobo"
    cd "$DEPS_DIR/curobo"
    git fetch --tags
    git checkout v0.7.8
    pip install -e . --no-build-isolation
}

s_extras() {
    activate_env
    pip install huggingface_hub
}

stage miniforge   s_miniforge
stage conda_env   s_conda_env
stage repo        s_repo
stage isaacsim    s_isaacsim
stage vcpkg       s_vcpkg
export CMAKE_TOOLCHAIN_FILE="$VCPKG_DIR/scripts/buildsystems/vcpkg.cmake"
stage tacex_core  s_tacex_core
stage uipc_deps   s_uipc_deps
stage tacex_uipc  s_tacex_uipc
stage isaaclab    s_isaaclab
stage curobo      s_curobo
stage extras      s_extras

cat <<'DONE'

>>> UniVTAC environment ready (conda activate UniVTAC happens in new shells).

    python -c "import isaacsim, isaaclab, tacex, tacex_uipc, curobo; print('ok')"

    Dataset (~131 GB) is not downloaded automatically:  bash data/download.sh
DONE
