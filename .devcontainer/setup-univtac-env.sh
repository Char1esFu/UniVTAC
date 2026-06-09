#!/usr/bin/env bash
# One-time heavy setup for the UniVTAC conda env. Run INSIDE the devcontainer:
#
#     bash .devcontainer/setup-univtac-env.sh
#
# Mirrors docs/Installation.md. Idempotent-ish: re-running skips an already
# installed miniforge / env / cloned repo, but pip/build steps will re-run.
# Everything it installs lands on the persisted named volumes (~/miniforge3,
# ~/deps), so it survives container rebuilds.
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="$HOME/miniforge3"
DEPS_DIR="$HOME/deps"
ENV_NAME="UniVTAC"

mkdir -p "$DEPS_DIR"

# vcpkg is installed in step 5; unset any inherited value so cmake doesn't
# pick up a non-existent toolchain file during the IsaacLab install (step 3).
unset CMAKE_TOOLCHAIN_FILE

# --- 0. miniforge3 (installs into the persisted volume on first run) ---------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo ">>> Installing miniforge3 into $CONDA_DIR"
    mkdir -p "$CONDA_DIR"
    installer="$CONDA_DIR/miniforge.sh"
    wget -O "$installer" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash "$installer" -b -u -p "$CONDA_DIR"
    rm -f "$installer"
    source "$CONDA_DIR/bin/activate"
    conda init --all
    conda config --set auto_activate_base false
    mamba shell init --shell bash --root-prefix "$CONDA_DIR"
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# --- 1. create the env -------------------------------------------------------
if ! conda env list | grep -qE "^[[:space:]]*${ENV_NAME}[[:space:]]"; then
    conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

cd "$REPO_DIR"
git -C third_party/TacEx lfs pull || true

# --- 2. Isaac Sim 4.5 (pip) --------------------------------------------------
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# flatdict needs to be built against an older setuptools, then restored
pip install "setuptools<81" wheel
pip install --no-build-isolation flatdict==4.0.1
pip install "setuptools==82.0.1"
pip install transforms3d

# --- 3. Isaac Lab 2.1.1 ------------------------------------------------------
if [ ! -d "$DEPS_DIR/IsaacLab" ]; then
    git clone https://github.com/isaac-sim/IsaacLab "$DEPS_DIR/IsaacLab"
fi
cd "$DEPS_DIR/IsaacLab"
git fetch --tags
git checkout v2.1.1
./isaaclab.sh --install

# --- 4. TacEx [Core] (modified, bundled source) ------------------------------
cd "$REPO_DIR/third_party/TacEx"
./tacex.sh -i

# --- 5. libuipc build deps: vcpkg + CUDA 12.4 toolchain (in-env) -------------
if [ ! -d "$DEPS_DIR/vcpkg" ]; then
    git clone https://github.com/microsoft/vcpkg.git "$DEPS_DIR/vcpkg"
fi
"$DEPS_DIR/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
export CMAKE_TOOLCHAIN_FILE="$DEPS_DIR/vcpkg/scripts/buildsystems/vcpkg.cmake"

conda env update -n "$ENV_NAME" --file "$REPO_DIR/third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml"
conda install -n "$ENV_NAME" -c conda-forge sysroot_linux-64=2.34 ffmpeg -y

# --- 6. tacex_uipc (builds libuipc + python bindings) ------------------------
cd "$REPO_DIR/third_party/TacEx"
pip install -e source/tacex_uipc -v

# --- 7. cuRobo 0.7.8 ---------------------------------------------------------
if [ ! -d "$DEPS_DIR/curobo" ]; then
    git clone https://github.com/NVlabs/curobo.git "$DEPS_DIR/curobo"
fi
cd "$DEPS_DIR/curobo"
git fetch --tags
git checkout v0.7.8
pip install -e . --no-build-isolation

# --- 8. dataset --------------------------------------------------------------
pip install huggingface_hub
# Dataset is ~131 GB; download manually when needed:
#   cd "$REPO_DIR" && bash data/download.sh

echo ""
echo ">>> UniVTAC environment ready.  Activate with:  conda activate UniVTAC"
echo "    To rebuild tacex_uipc manually, first run:"
echo "    export CMAKE_TOOLCHAIN_FILE=\"$DEPS_DIR/vcpkg/scripts/buildsystems/vcpkg.cmake\""
