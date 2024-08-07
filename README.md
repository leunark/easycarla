# easycarla

This project uses the **Carla** simulator to generate training data for autonomous driving models.\
It consists out of multiple modules that can be used even independently for other use cases. All code was vectorized for better performance.\
The resulting output is a **Kitti Dataset**. 

## Showcase

![Showcase](docs/showcase.gif)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

### Prerequisites

Ensure your system meets the following requirements:
- Windows 7 or later (64-bit)
- A modern GPU (NVIDIA is recommended for CUDA support)
- At least 50 GB of free disk space

### Install
1. **Download Carla**:
- Go to the [Carla GitHub releases page](https://github.com/carla-simulator/carla/releases) and download the latest release for Windows.
- Extract the downloaded zip file to a directory of your choice.

2. **Python**: Install Python **3.10** from the [official website](https://www.python.org/). Ensure you add Python to the system PATH during installation / Activate your environment. \
**Using `venv`**
- Open another Command Prompt and create a virtual environment:
    ```shell
    python -m venv carla-env
    ```
- Activate the virtual environment:
    ```shell
    carla-env\Scripts\activate
    ```

**Using `conda`**
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if you don't have it installed.
- Create a new conda environment:
    ```shell
    conda create --name carla-env python=3.10
    ```
- Activate the conda environment:
    ```shell
    conda activate carla-env
    ```

3. **Install this package**:
- Install carla package in development mode:
    ```shell
    pip install -e .
    ```

## Usage
1. **Run Carla Simulator**:
- Open a Command Prompt.
- Run `CarlaUE4.exe` to start the simulator:
    ```shell
    .\CarlaUE4.exe
    ```
    ***Note: Carla can be run in headless mode with `.\CarlaUE4.exe -RenderOffScreen`***

2. **Start Client**:
- Run the main script to setup the client:
    ```shell
    python src/easycarla/main.py
    ```
    or simply run `easycarla` with optional command line arguments (s. `examples/run.cmd`).



