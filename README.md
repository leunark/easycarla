# Carla Training Data Generator

This project uses the Carla simulator to generate training data for autonomous driving models. Carla provides a flexible and scalable platform for generating a wide range of driving scenarios and capturing various sensor data.

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

2. **Python**: Install Python from 3.7 to **3.10** from the [official website](https://www.python.org/). Ensure you add Python to the system PATH during installation / Activate your environment. \
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
        conda create --name carla-env python=3.7
        ```
    - Activate the conda environment:
        ```shell
        conda activate carla-env
        ```

3. **Install requirements**:
    - Install carla package:
        ```shell
        pip install carla
        ```
    - Install other required dependencies:
        ```shell
        pip install -r requirements.txt
        ```

## Usage

1. **Run Carla Simulator**:
   - Open a Command Prompt.
   - Run `CarlaUE4.exe` to start the simulator:
     ```shell
     path\to\Carla\WindowsNoEditor\CarlaUE4.ex
     ```
     ***Note: Carla can be run in headless mode with `CarlaUE4.exe -RenderOffScreen`***

2. **Start Client**:
    - Run `CarlaUE4.exe` with the `-RenderOffScreen` flag to start the simulator:
     ```shell
     python ...py
     ```


