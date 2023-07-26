# IRP2023
AI for science: learning the map function from observation space to hidden state space in data assimilation

Dependencies
------------
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