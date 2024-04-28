# Topo-DDA
Topo-DDA is a topology optimization code based on DDA that runs on GPU. The 
ultimate goal of this project is to efficiently do electromagnetic full-wave
simulation to be able to model complex structures at large scales in the field
of topology optimization.

This project contains a C++ / CUDA library, with Python bindings through
pybind11.

## Installation
We strongly recommend developing within a conda environment. You can use the
following commands to construct and activate a conda environment.

```
conda create --name environment_name
conda activate environment_name
```

### Download the repository and dependencies
We use git submodules to manage most of our dependencies. To automatically
download these modules alongside the Topo-DDA model repository, clone the
repository using:

```
git clone --recurse-submodules
```

If you forgot to use the `--recurse-submodules` flag, you can always just
download the submodules afterward using:

```
git submodule update --init --recursive
```

### Download the CUDA dependencies

If using Anaconda, run the following command within your conda environment.

```
conda install cuda -c nvidia
```

If not using Anaconda, you can install the CUDA toolkit using a package manager
of your choice or by downloading directly from NVIDIA (not recommended).

### Download the Python dependencies

The next step is to install Python and the Python dependencies. If using
Anaconda,

```
conda install python
pip install scipy
pip install numpy
pip install pytest
```

## Build the python bindings

We use CMake to automatically configure the library, so you should not have to
manually specify any paths during the build and install process. To build the
bindings, run the following command:

```
pip install .
```

## Run the python tests

Due to a limitation of the C++ library, we currently need to run the python
tests from within the  `python_tests` subdirectory.

```
cd python_tests
python -m pytest . -s
```
