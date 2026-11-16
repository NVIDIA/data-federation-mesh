<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Installation

DFM is distributed as a set of Python packages:

| Package | Description |
|---------|-------------|
| `nv-dfm-core` | Core framework — Pipeline API, execution engine, code generation, and CLI |
| `nv-dfm-lib-common` | Shared schemas and utilities used across adapter libraries |
| `nv-dfm-lib-weather` | Weather and climate data adapters (ECMWF, GFS, HRRR, SFNO, cBottle) |

`nv-dfm-core` is the only required package. The adapter libraries are optional and can be installed separately depending on your use case. See [Adapter Libraries](../../adapter_libraries/index.md) for more details.

**Python requirements:** core and common support Python 3.10–3.12; the weather library and full source workspace require Python 3.12. Use Python 3.12 to follow all tutorials.

## Install From PyPI

```bash
# Install core framework only
pip install nv-dfm-core

# Install weather adapters (includes core and common packages)
pip install nv-dfm-lib-weather
```

## Install From Source Code

DFM uses [uv](https://docs.astral.sh/uv/) to manage dependencies.

```{tip}
If you don't have `uv` installed, follow the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
```

```{admonition} Prerequisites
:class: warning

On Debian/Ubuntu systems, make sure the following system packages are installed:

    sudo apt update && sudo apt install -y curl gcc
```

Clone the repository:
```bash
git clone https://github.com/NVIDIA/data-federation-mesh.git
cd data-federation-mesh
```

Install all workspace packages:
```bash
uv sync --all-packages
```

Or install a single package:
```bash
uv sync --package nv-dfm-core
```

To run the tutorials, add the `tutorials` extra:
```bash
uv sync --all-packages --extra tutorials
```

```{note}
This is a multi-package workspace. Each `uv sync` invocation reconfigures the virtual environment to match exactly the requested set of packages — syncing for a single package will remove dependencies that are not required by that package. Use `--all-packages` when you need the full workspace available.
```

After your chosen synchronization command, activate the environment before running `dfm` commands directly:

```bash
source .venv/bin/activate
```

```{note}
The `tutorials` extra does not include a Jupyter frontend. The Weather Federation tutorials are `.ipynb` notebooks — open them with your own Jupyter environment (JupyterLab, Jupyter Notebook, or the VS Code / Cursor notebook editor). Select the kernel from this checkout's `.venv`. To launch JupyterLab from the project environment, install it first, e.g. `uv pip install jupyterlab && uv run --no-sync jupyter lab`. A later `uv sync` can remove this separately installed frontend; reinstall it afterward if needed.
```

## Installing the Weather Data Library

The `nv-dfm-lib-weather` package contains adapters for loading and processing weather data from different sources.
This library relies on [Earth2Studio](https://nvidia.github.io/earth2studio/v/0.16.0/userguide/about/install.html) and its dependencies.

```{admonition} Experimental
:class: warning
`nv-dfm-lib-weather` is **experimental**. It is provided as a collection of examples and starting points for building your own adapters. Before any production use, review and extensively test all adapters in your target environment.
```

To install `nv-dfm-lib-weather` without AI models:
```bash
pip install nv-dfm-lib-weather
```

Or from source:
```bash
uv sync --package nv-dfm-lib-weather
```

:::{note}
The base weather install (without the AI model extras below) needs **no GPU and no CUDA toolkit** — it installs on any machine. PyTorch is pulled in as a prebuilt wheel (no compilation). GPU prerequisites only apply to the AI model extras; see [AI Model Adapters](#ai-model-adapters-sfno-cbottle) below.
:::

### AI Model Adapters (SFNO, cBottle)

The SFNO and cBottle adapters perform GPU-accelerated AI inference and need extra setup beyond the base installation. Their model kernels are compiled from source against the pinned PyTorch build, so a set of **system-level** components must be present beforehand; the required Python packages are then installed differently depending on whether you use `uv` (from source) or `pip` (published wheels).

:::{admonition} System prerequisites (required for both `pip` and `uv`)
:class: warning

Install and configure these **before** installing the AI model extras:

1. **NVIDIA GPU** — A CUDA-capable GPU with compute capability ≥ 8.9 and ≥ 40 GB of GPU memory is recommended. See [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) for a list of supported hardware.

2. **NVIDIA GPU Drivers** — Install the appropriate driver for your GPU and OS. See the [NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html).

3. **CUDA Toolkit** — CUDA 12.9 is the tested and recommended version. It provides the `nvcc` compiler used to build the model kernels against PyTorch. See [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads).

4. **Host C/C++ build tools** — Install a compiler toolchain compatible with your CUDA toolkit. For the Linux GPU setup, Debian/Ubuntu users can install `build-essential` and `ninja-build` with `sudo apt install -y build-essential ninja-build`. See the [CUDA 12.9 Linux system requirements](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html#system-requirements) for supported distributions and compiler versions.
:::

The Python packages — **PyTorch**, **Earth2Studio**, and the non-PyPI model dependencies (`makani`, `earth2grid`, `cbottle`, `torch-harmonics`) — are then handled per install method, as described in the note below the install commands.

Install the AI model extras. Pick the extra(s) you need — to get both models, install `all` (or combine `cbottle` and `sfno`):

```bash
pip install "nv-dfm-lib-weather[cbottle]"        # cBottle (Climate in a Bottle) model adapters
pip install "nv-dfm-lib-weather[sfno]"           # SFNO (Spherical Fourier Neural Operator) adapters
pip install "nv-dfm-lib-weather[cbottle,sfno]"   # both models
pip install "nv-dfm-lib-weather[all]"            # all optional AI model dependencies
```

Or from source. Choose **one** command — `uv sync` uses exact mode, so each invocation prunes anything the target doesn't need (running them in sequence would just undo the previous one). To get both models, combine the extras in a single command or use `all`:

```bash
uv sync --package nv-dfm-lib-weather --extra cbottle                 # cBottle only
uv sync --package nv-dfm-lib-weather --extra sfno                    # SFNO only
uv sync --package nv-dfm-lib-weather --extra cbottle --extra sfno    # both models
uv sync --package nv-dfm-lib-weather --extra all                     # all AI model extras
```

:::{note}
`uv sync --package nv-dfm-lib-weather ...` resolves the shared workspace environment to exactly what that package (plus the chosen extras) needs, so it prunes unrelated workspace items such as the root `dev`/`docs` dependency groups and the `tutorials` extra if they were installed. If you want the AI extras **on top of** a full workspace install, use `--all-packages` instead, e.g. `uv sync --all-packages --extra all`.
:::

:::{note}
**Installing from source with `uv`:** PyTorch (the cu129 build on Linux) and the non-PyPI dependencies (`makani`, `earth2grid`, `cbottle`, `torch-harmonics`) are resolved and built **automatically** from the git/index sources configured in the workspace. Beyond the system prerequisites above, there is nothing extra to install.

**Installing the published wheels with `pip`:** you must install these Python packages manually first:

- **PyTorch** with CUDA support matching your CUDA toolkit. See [pytorch.org](https://pytorch.org/) for platform-specific commands, e.g.:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    ```
- **Earth2Studio model-specific dependencies** (not on PyPI), per the [Earth2Studio 0.16.0 Installation Guide](https://nvidia.github.io/earth2studio/v/0.16.0/userguide/about/install.html#optional-dependencies):
    - **SFNO** requires [`makani`](https://nvidia.github.io/earth2studio/v/0.16.0/userguide/about/install.html#sfno) (from git).
    - **cBottle** requires [`earth2grid` and `cbottle`](https://nvidia.github.io/earth2studio/v/0.16.0/userguide/about/install.html#cbottle) (from git, with `--no-build-isolation`).

Earth2Studio itself is pulled in automatically by `nv-dfm-lib-weather` in both cases.
:::

