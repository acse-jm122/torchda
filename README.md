# DeepDA
Use deep learning in data assimilation workflow.

[![flake8](https://github.com/acse-jm122/irp-acse-jm122/actions/workflows/flake8-format-test.yml/badge.svg)](https://github.com/acse-jm122/irp-acse-jm122/actions/workflows/flake8-format-test.yml)
[![pytest](https://github.com/acse-jm122/irp-acse-jm122/actions/workflows/pytest-conda.yml/badge.svg)](https://github.com/acse-jm122/irp-acse-jm122/actions/workflows/pytest-conda.yml)

Package Dependencies
--------------------
* Python 3.10 or later
* PyTorch (Recommend 2.0 or later)

Installation
------------
* Build a `conda` environment, run:
```
conda env create -f environment.yml
```
then activate it with:
```
conda activate DeepDA
```

* Install PyTorch with `pip`, run:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```
then at the root folder of this repo, run:
```
pip3 install .
```

* Install `deepda` from source, run:
```
pip3 install git+https://github.com/acse-jm122/deepda.git
```
