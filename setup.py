from setuptools import setup, find_packages

setup(
    name="Multivariate Diffusion Models",
    version="1.0.0",
    url="https://github.com/rajesh-lab/MultivariateDiffusionModels",
    author="Raghav Singhal; Mark Goldstein",
    author_email="singhal.raghav@gmail.com",
    description="Learned multivariate diffusion models",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.22.3",
        "pytorch_lightning >= 1.4.9",
        "matplotlib >= 3.4.3",
        "torchmetrics >= 0.5.1",
        "torch >= 1.13.0",
        "torchvision >= 0.14.0",
        "wandb >= 0.12.21",
        "black >= 22.6.0",
        "torch-fidelity",
        "torch-ema",
        "torchdiffeq",
        "joblib",
        "seaborn",
    ],
)
