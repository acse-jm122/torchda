try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="deepda",
    version="0.0.1",
    author="Jingyang Min",
    description="Use deep learning in data assimilation",
    packages=["deepda"],
    install_requires=required,
    python_requires=">=3.10",
)
