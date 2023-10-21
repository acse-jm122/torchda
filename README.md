# DeepDA
Use deep learning in data assimilation workflow.

[![flake8](https://github.com/acse-jm122/irp-acse-jm122/actions/workflows/flake8-format-test.yml/badge.svg)](https://github.com/acse-jm122/irp-acse-jm122/actions/workflows/flake8-format-test.yml)
[![pytest](https://github.com/acse-jm122/irp-acse-jm122/actions/workflows/pytest-conda.yml/badge.svg)](https://github.com/acse-jm122/irp-acse-jm122/actions/workflows/pytest-conda.yml)
[![License](https://img.shields.io/github/license/acse-jm122/deepda
)](https://github.com/acse-jm122/deepda/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Tutorials (4 notebooks)
-----------------------
| **notebook** | **open in colab / kaggle** |
|:------------:|:-------------------------------------------------:|
| [EnKF Testing Demo on Lorenz 63 Model](https://github.com/acse-jm122/deepda/blob/main/examples/lorenz_enkf.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-jm122/deepda/blob/main/examples/lorenz_enkf.ipynb) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/acse-jm122/deepda/blob/main/examples/lorenz_enkf.ipynb) |
| [LSTM on Lorenz 63 Model](https://github.com/acse-jm122/deepda/blob/main/examples/lorenz_lstm_example.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-jm122/deepda/blob/main/examples/lorenz_lstm_example.ipynb) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/acse-jm122/deepda/blob/main/examples/lorenz_lstm_example.ipynb) |
| [Testing Demo on Variational Methods](https://github.com/acse-jm122/deepda/blob/main/examples/variational_methods.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-jm122/deepda/blob/main/examples/variational_methods.ipynb) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/acse-jm122/deepda/blob/main/examples/variational_methods.ipynb) |
| [Testing Demo of Variational Methods on Shallow Water Model](https://github.com/acse-jm122/deepda/blob/main/examples/shallow_water_example/models.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-jm122/deepda/blob/main/examples/shallow_water_example/models.ipynb) [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/acse-jm122/deepda/blob/main/examples/shallow_water_example/models.ipynb) |

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
